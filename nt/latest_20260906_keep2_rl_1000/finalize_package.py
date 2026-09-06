"""Generate a navigable report index and manifest after packaging/portable parity."""
from pathlib import Path
import json,hashlib,gzip,ast
P=Path(__file__).resolve().parent
def ignored(p):return any(x in ('build','runs','__pycache__','.venv')for x in p.relative_to(P).parts)
def payload():return sorted(p for p in P.rglob('*')if p.is_file()and not ignored(p)and p.name!='PACKAGE_MANIFEST.json')
def main():
    reports=sorted((P/'archive').rglob('*.md'))
    lines=['# RL全过程报告索引','','这些是原样归档报告；内部原机绝对路径只用于溯源。当前使用说明以根README为准。','']
    previous=None
    for f in reports:
        relative=f.relative_to(P);group=relative.parts[2]if len(relative.parts)>2 else 'other'
        if group!=previous:lines+=['','## '+group,''];previous=group
        lines.append('- ['+f.name+'](<'+relative.as_posix()+'>)')
    (P/'REPORT_INDEX_ZH.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    configs=games=0
    for p in P.rglob('games.json.gz'):
        with gzip.open(p,'rt',encoding='utf8')as f:games+=len(json.load(f))
        configs+=1
    models=json.loads((P/'MODEL_INDEX.json').read_text())
    for f in payload():
        assert f.stat().st_size<50*1024**2,f
        assert f.suffix not in ('.o','.so','.npz','.pyc'),f
        if f.suffix=='.py':ast.parse(f.read_text(encoding='utf-8-sig'),filename=str(f))
    def strip_includes(text):return '\n'.join(l for l in text.replace('\r\n','\n').splitlines()if not l.startswith('#include')).strip()
    for n in ('f3_policy.cpp','policy.cpp','f3_config.hpp','rl_common.hpp'):
        assert strip_includes((P/'src'/n).read_text())==strip_includes((P/'archive/frozen_runtime'/n).read_text()),n
    acceptance=dict(status='PASS',models=len(models),optimizer_checkpoints=sum(bool(m['pt'])for m in models),
        report_documents=len(reports),result_configurations=configs,result_games=games,
        portable_parity_games=812,policy_changes='include paths only',
        omitted=['rollout decision arrays','most intermediate weights/Adam states','replays','compiled objects/libraries','environments/caches'],
        historical_full_training_audit='Preserved original receipts; absent raw rollouts cannot be re-audited from this light package alone.')
    (P/'PACKAGE_ACCEPTANCE.json').write_text(json.dumps(acceptance,indent=2),encoding='utf8')
    files={str(p.relative_to(P)).replace('\\','/'):{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}for p in payload()}
    (P/'PACKAGE_MANIFEST.json').write_text(json.dumps(dict(count=len(files),bytes=sum(e['bytes']for e in files.values()),files=files),indent=2),encoding='utf8')
    print(json.dumps(acceptance,indent=2))
if __name__=='__main__':main()
