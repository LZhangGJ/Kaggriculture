"""Run the released configuration against seven LIVE frozen opponents."""
from pathlib import Path
import argparse,json
from experiment import pool,run,NAMES,R
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--tag',required=True,help='New output directory name under runs/; never overwrites')
p.add_argument('--count',type=int,default=4,help='Independent seeds; each uses both seats against each opponent')
p.add_argument('--start',type=int,default=260909000)
p.add_argument('--threads',type=int,default=4)
p.add_argument('--config',type=Path,default=R/'policy/config.json')
p.add_argument('--opponents',default=','.join(NAMES[:7]))
p.add_argument('--baseline',choices=['auto','j7'])
p.add_argument('--trace',action='store_true')
a=p.parse_args();opps=[]
for name in a.opponents.split(','):
 i=int(name) if name.isdigit() else NAMES.index(name)
 if not 0<=i<7:raise ValueError('This acceptance entry requires one of the seven frozen opponents')
 opps.append(i)
cfg={} if a.baseline else json.loads(a.config.read_text())
run(pool(),a.tag,cfg,a.count,a.start,opps,a.baseline or 'triad',a.trace,a.threads)
