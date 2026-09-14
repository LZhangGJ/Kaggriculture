#!/usr/bin/env python3
import subprocess,json,datetime,sys
from pathlib import Path
b=Path('/mnt/data/r3_work');name=sys.argv[1];jobs=json.loads(Path(sys.argv[2]).read_text());results=[]
for job in jobs:
 if datetime.datetime.now(datetime.timezone.utc)>=datetime.datetime.fromisoformat('2026-09-13T20:40:45+00:00'):
  results.append({'not_started':job,'reason':'packaging deadline guard'});break
 rc=subprocess.call(['python',str(b/'tests/timed_run.py'),*job]);results.append({'job':job,'returncode':rc})
 (b/'logs'/f'{name}_queue.json').write_text(json.dumps(results,indent=2))
 if rc:break
