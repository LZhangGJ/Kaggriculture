#!/usr/bin/env python3
"""Deterministic tests for the sale overlay; no match-win or sandbox claims."""
from pathlib import Path
import copy, importlib.util, sys, unittest
R=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location("tested_sale", R/"main.py")
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def observation():
    farms=[]
    for _ in range(2):
        farms.append({"farmer":[4,4],"hands":[],"tiles":[[{} for c in range(10)] for r in range(10)]})
    return {"step":240,"player":0,"farms":farms,"private":{"shed":{"MELON":5,"WHEAT":3},"inventories":[{}]},"market":{"prices":{"MELON":160,"WHEAT":20,"MILK":100}}}

class Fake:
    def __init__(self,action):self.action=action
    def __call__(self,*args):return copy.deepcopy(self.action)

def overlay(action):
    a=object.__new__(m.SaleAgent);a.base=Fake(action)
    a.config={"r13_sale_mode":2,"r13_sale_from":0,"r13_sale_order":0};a.changed=0;a.added=0
    return a

def basic_action():return {"farmer":["PASS"],"hands":[],"market":[["BUY","WHEAT",2]]}

def mature(obs):obs["farms"][1]["tiles"][1][2]={"crop":"MELON","yield_units":10,"planted_day":0}

class OverlayTests(unittest.TestCase):
    def test_no_visible_product_preserves_action(self):
        obs=observation();act=basic_action();self.assertEqual(overlay(act)(obs),act)
    def test_mature_same_product_adds_sale_and_preserves_units(self):
        obs=observation();mature(obs);act=basic_action();out=overlay(act)(obs)
        self.assertEqual(out["market"],[["SELL","MELON",5],["BUY","WHEAT",2]])
        self.assertEqual(out["farmer"],act["farmer"]);self.assertEqual(out["hands"],act["hands"])
    def test_immature_melon_does_not_trigger(self):
        obs=observation();mature(obs);obs["farms"][1]["tiles"][1][2]["planted_day"]=1
        act=basic_action();self.assertEqual(overlay(act)(obs),act)
    def test_inputs_not_liquidated(self):
        obs=observation();obs["farms"][1]["tiles"][1][2]={"crop":"WHEAT","yield_units":5,"planted_day":0}
        act=basic_action();self.assertEqual(overlay(act)(obs),act)
    def test_current_drop_can_be_sold(self):
        obs=observation();mature(obs);obs["private"]["inventories"][0]={"MELON":7}
        act=basic_action();act["farmer"]=["DROP"]
        self.assertEqual(overlay(act)(obs)["market"][0],["SELL","MELON",12])
    def test_drop_capacity_shared_and_item_order_respected(self):
        obs=observation();obs["private"]["shed"]={"WHEAT":96};obs["private"]["inventories"][0]={"MILK":3,"MELON":4}
        act={"farmer":["DROP"],"hands":[],"market":[]}
        shed=m.SaleAgent.after_units(obs,act)
        self.assertEqual(shed,{"WHEAT":96,"MILK":3,"MELON":1})
    def test_pickup_reduces_available_stock(self):
        obs=observation();act={"farmer":["PICKUP","MELON",3],"hands":[],"market":[]}
        self.assertEqual(m.SaleAgent.after_units(obs,act)["MELON"],2)
    def test_existing_sale_enlarged_without_reordering(self):
        obs=observation();mature(obs);act=basic_action();act["market"].append(["SELL","MELON",2])
        self.assertEqual(overlay(act)(obs)["market"],[["BUY","WHEAT",2],["SELL","MELON",5]])
    def test_market_limit_keeps_all_original_orders(self):
        obs=observation();mature(obs);act=basic_action();act["market"]=[["BUY","WHEAT",1] for _ in range(10)]
        self.assertEqual(overlay(act)(obs),act)
    def test_native_rejects_trained_selector(self):
        a=m.create_agent()
        try:
            a.config["scenario"]=-1
            with self.assertRaises(ValueError):a.reset()
        finally:a.close()

if __name__=="__main__":
    # build.py forwards its compiler option; these runtime tests do not compile.
    if "--cxx" in sys.argv:
        i=sys.argv.index("--cxx");del sys.argv[i:i+2]
    unittest.main(verbosity=2)
