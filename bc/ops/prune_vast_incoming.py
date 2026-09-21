#!/home/keith/kaggriculture-rl-20260915/venv/bin/python
"""Retention for arena-9975 (Vast) /opt/arena-bench/.arena/incoming.

Deletes ONLY continuous/daily job bundles whose run is fully drained on the WRX
coordinator (every game resolved) AND whose remote mtime is older than --hours.
Never touches matchmaking-policy.json, bundles without a local manifest, or
undrained runs. Writes a JSON receipt per apply run. Owner: Claude campaign
maintenance 2026-09-20 (user-approved).
"""
import sys,json,subprocess,time,argparse,shlex
sys.path.insert(0,"/home/keith/kaggriculture-arena")
from pathlib import Path
from tools.arena import schedule
from tools.arena.continuous import read
ROOT=Path("/home/keith/kaggriculture-arena/.arena")
HOST="arena-9975"; INCOMING="/opt/arena-bench/.arena/incoming"
RECEIPTS=Path("/home/keith/kaggriculture-arena/research/incoming-prune")
def ssh(cmd,timeout=300):
    return subprocess.run(["ssh","-o","BatchMode=yes","-o","ConnectTimeout=15",HOST,cmd],capture_output=True,text=True,timeout=timeout)
def main():
    ap=argparse.ArgumentParser();ap.add_argument("--apply",action="store_true");ap.add_argument("--hours",type=float,default=48)
    a=ap.parse_args()
    out=ssh(f"cd {INCOMING} && stat -c '%n %s %Y' * ; df -B1 --output=avail / | tail -1")
    if out.returncode!=0: print("ssh failed",out.stderr[:300]); sys.exit(1)
    lines=out.stdout.strip().splitlines(); avail_before=int(lines[-1])
    now=time.time(); victims=[]; kept={"undrained":0,"young":0,"norun":0,"other":0}
    for line in lines[:-1]:
        p=line.split()
        if len(p)!=3 or not p[0].endswith(".zip"): kept["other"]+=1; continue
        name,size,mt=p[0],int(p[1]),int(p[2]); rid=name[:-4]
        mp=ROOT/"runs"/rid/"manifest.json"
        if not mp.exists(): kept["norun"]+=1; continue
        m=read(mp)
        drained=schedule.complete(ROOT,m) if m.get("kind")=="daily" else schedule.drained(ROOT,m)
        if not drained: kept["undrained"]+=1; continue
        if (now-mt)/3600<=a.hours: kept["young"]+=1; continue
        victims.append(dict(name=name,size=size,mtime=mt))
    total=sum(v["size"] for v in victims)
    print(f"candidates {len(victims)} bytes {total/1e9:.1f}G kept {kept} avail_before {avail_before/1e9:.1f}G apply={a.apply}")
    receipt=dict(at=now,host=HOST,incoming=INCOMING,hours=a.hours,apply=a.apply,kept=kept,avail_before=avail_before,candidates=victims)
    if a.apply and victims:
        names=[v["name"] for v in victims]; deleted=0; errors=[]
        for i in range(0,len(names),400):
            chunk=names[i:i+400]
            r=ssh(f"cd {INCOMING} && rm -f -- {' '.join(shlex.quote(n) for n in chunk)} && echo OK")
            if r.returncode==0 and "OK" in r.stdout: deleted+=len(chunk)
            else: errors.append(r.stderr[:300])
        after=ssh(f"df -B1 --output=avail / | tail -1; ls {INCOMING} | wc -l")
        receipt.update(deleted=deleted,errors=errors,after=after.stdout.split())
        print("deleted",deleted,"errors",errors,"after(avail,count)",after.stdout.split())
    RECEIPTS.mkdir(parents=True,exist_ok=True)
    fn=RECEIPTS/f"{time.strftime('%Y%m%dT%H%M%SZ',time.gmtime(now))}-{'apply' if a.apply else 'dryrun'}.json"
    json.dump(receipt,open(fn,"w"),indent=1); print("receipt",fn)
if __name__=="__main__": main()
