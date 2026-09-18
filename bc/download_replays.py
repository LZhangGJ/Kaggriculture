"""Download dated organizer ZIPs without extracting the replay corpus."""
import argparse,csv,hashlib,json,subprocess,zipfile
from pathlib import Path

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--index',type=Path,default=Path(__file__).parent/'metadata/organizer-index-20260917.csv')
    p.add_argument('--refresh-index',action='store_true')
    p.add_argument('--from-date',default='2026-08-15');p.add_argument('--through-date',default='2026-09-16')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    if a.refresh_index:
        dest=a.output/'index';dest.mkdir(exist_ok=True)
        subprocess.run(['kaggle','datasets','download','-d','kaggle/kaggriculture-episodes-index','-p',str(dest),'-q'],check=True)
        with zipfile.ZipFile(dest/'kaggriculture-episodes-index.zip') as z:
            candidates=[n for n in z.namelist() if Path(n).name=='manifest.csv']
            if len(candidates)!=1:raise ValueError('Expected one organizer manifest.csv')
            a.index=dest/'manifest.csv';a.index.write_bytes(z.read(candidates[0]))
    rows=[r for r in csv.DictReader(a.index.open(encoding='utf-8-sig')) if a.from_date<=r['date']<=a.through_date]
    if not rows:raise ValueError('No dates match the requested window')
    for row in rows:
        slug=row['daily_dataset_slug']
        if not slug.startswith('kaggriculture-episodes-') or '/' in slug:raise ValueError('Unexpected organizer slug')
        dest=a.output/row['date'];dest.mkdir(exist_ok=True);archive=dest/(slug+'.zip');receipt=dest/'receipt.json'
        if not archive.exists():subprocess.run(['kaggle','datasets','download','-d','kaggle/'+slug,'-p',str(dest),'-q'],check=True)
        with archive.open('rb') as f:sha=hashlib.file_digest(f,'sha256').hexdigest()
        if receipt.exists() and json.loads(receipt.read_text())['sha256']==sha:
            print(row['date'],'verified, reused',flush=True);continue
        with zipfile.ZipFile(archive) as z:
            bad=z.testzip()
            if bad:raise ValueError('CRC failure: '+bad)
            count=sum(n.endswith('.json') for n in z.namelist())
        receipt.write_text(json.dumps(dict(date=row['date'],dataset='kaggle/'+slug,sha256=sha,bytes=archive.stat().st_size,
            episode_files=count,index_episode_count=int(row['episode_count']),count_matches=count==int(row['episode_count'])),indent=2))
        print(row['date'],count,'replays',flush=True)

if __name__=='__main__':main()
