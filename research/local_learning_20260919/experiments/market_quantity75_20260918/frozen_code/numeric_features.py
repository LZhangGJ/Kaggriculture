"""Exact visible integer inputs, shared by offline data and live inference."""
import numpy as np
from features import ITEMS,CROPS,PRODUCTS,SHOPS,MAX_UNITS
GLOBAL_NAMES=("own_money","other_money","own_hires","other_hires","own_hands","other_hands")
GLOBAL_NAMES+=tuple("shed_"+x for x in ITEMS)+tuple("seeds_"+x for x in CROPS)
GLOBAL_NAMES+=tuple("market_inventory_"+x for x in PRODUCTS)+tuple("market_price_"+x for x in PRODUCTS)
GLOBAL_NAMES+=tuple("shop_count_"+x for x in SHOPS)
def exact_numbers(obs):
    seat=int(obs["player"]);own=obs["farms"][seat];other=obs["farms"][1-seat];private=obs["private"]
    values=[own["money"],other["money"],own["hires_today"],other["hires_today"],len(own["hands"]),len(other["hands"])]
    values += [private["shed"].get(x,0) for x in ITEMS]+[private["seeds"].get(x,0) for x in CROPS]
    values += [obs["market"]["inventory"].get(x,0) for x in PRODUCTS]+[obs["market"]["prices"].get(x,0) for x in PRODUCTS]
    values += [obs["town"]["unlocked_shops"].count(x) for x in SHOPS]
    g=np.asarray(values,dtype=np.float64);assert len(g)==len(GLOBAL_NAMES)==49
    inv=private["inventories"];assert len(inv)==1+len(own["hands"]) and len(inv)<=MAX_UNITS
    u=np.zeros((MAX_UNITS,len(ITEMS)),dtype=np.float64)
    u[:len(inv)]=[[a.get(x,0) for x in ITEMS] for a in inv]
    for a in (g,u):
        assert np.isfinite(a).all() and np.equal(a,np.rint(a)).all(),"Nonintegral visible numeric field"
        assert (a>=-(2**31)).all() and (a<2**31).all(),"int32 range exceeded"
    return g.astype(np.int32),u.astype(np.int32)
