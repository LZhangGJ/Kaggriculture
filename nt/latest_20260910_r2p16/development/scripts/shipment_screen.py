"""OFF fixture verification, then matched live tests of valuation-only lag."""
import json
import run_panel as panel
import economic_screen as screen
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
def main():
    panel.init_worker()
    pool=json.loads((HERE/'pool_all11.json').read_text())
    base={(r['opponent'],r['seed'],r['opponent_seat']):r for r in json.loads((HERE/'baseline_rows_all11.json').read_text())}
    receipt=HERE/'candidate_r2p3/OFF_CONFORMANCE.json'
    if not receipt.exists():
        checks=[]
        for opponent,offset in zip(pool[:4],[8,28,42,71]):
            for seat in (0,1):
                seed=2609110000+offset
                row=panel.game((opponent['id'],str(ROOT/opponent['working']),seed,seat,str(HERE/'candidate_r2p3/off_fixtures'),None,str(HERE/'candidate_r2p3/policy/shipment_off.so')))
                old=base[(opponent['id'],seed,seat)]
                assert not row['runtime_error'] and all(row[k]==old[k] for k in ['joint_action_sha256','r2_cash','opponent_cash','steps']),row
                checks.append(row)
        panel.save(receipt,dict(status='PASS',actions=719*len(checks),rows=checks))
    for name in ('half','one'):
        screen.OPTIONS['shipment_'+name]={}
        screen.run(['shipment_'+name],8,HERE/'shipment_screen8'/name,True,HERE/f'candidate_r2p3/policy/shipment_{name}.so')
if __name__=='__main__':main()
