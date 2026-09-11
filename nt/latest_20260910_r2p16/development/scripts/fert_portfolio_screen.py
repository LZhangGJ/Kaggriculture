from pathlib import Path
import argparse,json
import run_panel as panel
import economic_screen as screen
HERE=Path(__file__).resolve().parent
def main():
    p=argparse.ArgumentParser();p.add_argument('--seeds',type=int,default=8);args=p.parse_args();assert 1<=args.seeds<=100
    assert (HERE/'crop_clock_validation32/RESULTS.json').exists()
    audit=json.loads((HERE/'fert_portfolio_audit/RESULTS.json').read_text());assert audit['status']=='PASS' and audit['cases']==42
    assert json.loads((HERE/'candidate_r2p14/UNIT_TESTS.json').read_text())['status']=='PASS'
    binary=HERE/'candidate_r2p14/policy/fertportfolio1.so';receipt=json.loads((HERE/'candidate_r2p14/fertportfolio1.BUILD.json').read_text())
    assert panel.sha(binary)==receipt['binary_sha256']
    for p,h in receipt['sources'].items():assert panel.sha(HERE/'candidate_r2p14/policy'/p)==h
    name='fertportfolio1';screen.OPTIONS[name]={};screen.run([name],args.seeds,HERE/f'fert_portfolio_screen{args.seeds}'/name,True,binary)
if __name__=='__main__':main()
