from pathlib import Path
import argparse,json
import run_panel as panel
import economic_screen as screen
HERE=Path(__file__).resolve().parent

def main():
    p=argparse.ArgumentParser();p.add_argument('--seeds',type=int,default=8);p.add_argument('--sales',default='0,1');a=p.parse_args()
    assert 1<=a.seeds<=100
    audit=json.loads((HERE/'crop_clock_forecast_audit/RESULTS.json').read_text());assert audit['status']=='PASS' and audit['cases']==42
    for sale in map(int,a.sales.split(',')):
        assert sale in (0,1);name=f'cropclock1_sale{sale}';binary=HERE/'candidate_r2p11/policy'/(name+'.so')
        receipt=json.loads((HERE/'candidate_r2p11'/(name+'.BUILD.json')).read_text());assert panel.sha(binary)==receipt['binary_sha256']
        for p,h in receipt['sources'].items():assert panel.sha(HERE/'candidate_r2p11/policy'/p)==h
        screen.OPTIONS[name]={};screen.run([name],a.seeds,HERE/f'crop_clock_screen{a.seeds}'/name,True,binary)
if __name__=='__main__':main()
