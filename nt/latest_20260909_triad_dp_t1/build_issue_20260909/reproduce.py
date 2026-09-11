"""Portable compiler diagnosis, isolated in a new temporary directory."""
import argparse,hashlib,json,pathlib,platform,subprocess,sys,tempfile,time,zipfile
ROOT=pathlib.Path(__file__).resolve().parent

def run(cmd,log=None,timeout=240):
    result=subprocess.run([str(x) for x in cmd],capture_output=True,text=True,timeout=timeout)
    if log:log.write_text(result.stdout+result.stderr,encoding='utf-8')
    if result.returncode:raise RuntimeError(f'Command failed ({result.returncode}): {cmd}\n{result.stderr[-8000:]}')
    return result.stdout

def minimal(work,compiler):
    rows={}
    for name,flags in [('o0',['-O0']),('o3',['-O3']),('nomodref',['-O3','-fno-ipa-modref'])]:
        binary=work/('minimal_'+name)
        cmd=[compiler,'-std=c++20',*flags,'-ffp-contract=off','-Wall','-Wextra',ROOT/'evidence/diagnosis/ipa_modref_repro.cpp','-o',binary]
        run(cmd,work/(name+'_build.log'))
        output=run([binary,'0'])
        rows[name]={'command':[str(x) for x in cmd],'output':output.strip(),'matches_expected':output.split()==['9','29','10','26','11','28']}
    assert rows['o0']['matches_expected'] and rows['nomodref']['matches_expected'],rows
    return {'variants':rows,'compiler_issue_reproduced':not rows['o3']['matches_expected']}

def step0(work,compiler):
    if platform.system()!='Linux' or platform.machine() not in ('x86_64','AMD64'):raise RuntimeError('Frozen shared libraries require Linux x86-64 / WSL2.')
    unpack=work/'first_step';unpack.mkdir()
    with zipfile.ZipFile(ROOT/'first_step_reproducer.zip') as z:
        for info in z.infolist():
            target=(unpack/info.filename).resolve()
            if not target.is_relative_to(unpack.resolve()) or ((info.external_attr>>16)&0o170000)==0o120000:raise ValueError('Unsafe archive member')
        z.extractall(unpack)
    package=unpack/'T1_Rebuild_Divergence_GPT_Review_20260909_v1'
    run([sys.executable,package/'verify_package.py'],work/'archive_verify.log')
    source=package/'source_release/policy'
    binary=work/'rebuilt_nomodref.so'
    cmd=[compiler,'-std=c++20','-O3','-DNDEBUG','-ffp-contract=off','-fPIC','-shared','-Wl,-Bsymbolic','-fno-ipa-modref',source/'bridge.cpp',source/'executor/vendor/simulator.cpp','-o',binary]
    run(cmd,work/'t1_build.log')
    report=work/'first_step_comparison.json'
    run([sys.executable,package/'reproduce_step0.py','--extra-binary',binary,'--out',report],work/'first_step.log',timeout=90)
    data=json.loads(report.read_text());expected=data['results']['supplier_original'];actual=data['results']['extra']
    assert data['same_wire_input'] and data['same_config'] and data['supplier_matches_frozen_trace'] and data['action_divergence_reproduced']
    assert actual['action']==expected['action'] and actual['debug']==expected['debug'],data
    return {'status':'PASS','command':[str(x) for x in cmd],'same_input':True,'same_config':True,'action_and_debug_match_supplier':True,'binary_sha256':hashlib.sha256(binary.read_bytes()).hexdigest(),'detailed_result':str(report)}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=('all','minimal','step0'),default='all')
    p.add_argument('--compiler',default='g++')
    p.add_argument('--out',type=pathlib.Path,help='Optional NEW JSON path; refuses overwrite')
    args=p.parse_args()
    if args.out and args.out.exists():raise FileExistsError(args.out)
    work=pathlib.Path(tempfile.mkdtemp(prefix='t1-build-issue-')).resolve()
    start=time.monotonic();result={'compiler':run([args.compiler,'--version']).splitlines()[0],'work_directory':str(work),'mode':args.mode}
    print('Diagnostic workspace:',work,flush=True)
    if args.mode in ('all','minimal'):
        result['minimal']=minimal(work,args.compiler)
        print('Minimal example complete',flush=True)
    if args.mode in ('all','step0'):result['step0']=step0(work,args.compiler)
    result.update(status='PASS',seconds=time.monotonic()-start,production_agent_modified=False)
    text=json.dumps(result,ensure_ascii=False,indent=2)+'\n'
    (work/'RESULT.json').write_text(text,encoding='utf-8')
    if args.out:
        with args.out.open('x',encoding='utf-8') as f:f.write(text)
    print(text,end='')
if __name__=='__main__':main()
