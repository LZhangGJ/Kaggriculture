"""Download notebook outputs as data. Never execute notebook code on the host."""
import argparse
import io
import json
import tarfile
import zipfile
from pathlib import Path, PurePosixPath

from .store import file_hash, write

LIMIT = 256 * 1024 * 1024
BRIDGE = '''import contextlib, inspect, json, os, random, runpy, sys
class Obj(dict):
    def __getattr__(self, key):
        try: return self[key]
        except KeyError: raise AttributeError(key)
def obj(x):
    if isinstance(x, dict): return Obj({k: obj(v) for k,v in x.items()})
    if isinstance(x, list): return [obj(v) for v in x]
    return x
random.seed(int(os.environ.get('ARENA_AGENT_SEED', '0')))
with contextlib.redirect_stdout(sys.stderr):
    scope = runpy.run_path('main.py', run_name='arena_policy')
    policy = scope.get('agent') or scope.get('my_agent')
    if not callable(policy): raise RuntimeError('No agent function in main.py')
    sig = inspect.signature(policy)
    accepts_config = len(sig.parameters) >= 2
for line in sys.stdin:
    request = json.loads(line)
    with contextlib.redirect_stdout(sys.stderr):
        observation = obj(request['observation'])
        action = policy(observation, obj(request['configuration'])) if accepts_config else policy(observation)
    print(json.dumps(action, allow_nan=False), flush=True)
'''


def safe_name(name):
    p = PurePosixPath(name)
    if p.is_absolute() or '..' in p.parts or '\\' in name or ':' in name:
        raise ValueError('Unsafe output path')
    return str(p)


def unpack(data):
    result = {}
    total = 0
    with tarfile.open(fileobj=io.BytesIO(data), mode='r:*') as tar:
        for member in tar:
            if member.isdir():
                continue
            if not member.isfile():
                raise ValueError('Archive links and special files forbidden')
            name = safe_name(member.name)
            total += member.size
            if total > LIMIT or name in result or len(result) >= 10000:
                raise ValueError('Oversized or duplicate archive entries')
            result[name] = tar.extractfile(member).read()
    if 'main.py' not in result:
        mains = [p for p in result if p.endswith('/main.py')]
        if len(mains) != 1:
            raise ValueError('No unambiguous main.py')
        prefix = mains[0][:-len('main.py')]
        result = {k[len(prefix):]: v for k,v in result.items() if k.startswith(prefix)}
    return result


def pack(files, destination):
    if '_arena_bridge.py' in files:
        raise ValueError('Reserved bridge filename')
    with zipfile.ZipFile(destination, 'w', compression=zipfile.ZIP_DEFLATED) as z:
        for name, data in sorted({**files, '_arena_bridge.py': BRIDGE.encode()}.items()):
            item = zipfile.ZipInfo(safe_name(name), date_time=(2026,1,1,0,0,0))
            item.external_attr = 0o100644 << 16
            item.compress_type = zipfile.ZIP_DEFLATED
            z.writestr(item, data)


def export(notebook, destination):
    import requests
    from kaggle.api.kaggle_api_extended import KaggleApi, ApiGetKernelRequest, ApiListKernelSessionOutputRequest
    api = KaggleApi()
    api.authenticate()
    owner, slug = notebook.split('/')
    with api.build_kaggle_client() as client:
        kernel = client.kernels.kernels_api_client
        query = ApiGetKernelRequest(); query.user_name = owner; query.kernel_slug = slug
        metadata = kernel.get_kernel(query).metadata
        if getattr(metadata, 'is_private', False):
            raise ValueError('Discovery only imports public notebooks')
        before = metadata.current_version_number
        request = ApiListKernelSessionOutputRequest(); request.user_name = owner; request.kernel_slug = slug; request.page_size = 100
        outputs = kernel.list_kernel_session_output(request)
        if outputs.next_page_token:
            raise ValueError('Output listing exceeds adapter limit')
        archives = [f for f in outputs.files if f.file_name.endswith('.tar.gz')]
        chosen = next((f for f in archives if f.file_name == 'submission.tar.gz'), None)
        if chosen is None and len(archives) == 1:
            chosen = archives[0]
        if chosen is None:
            chosen = next((f for f in outputs.files if f.file_name in ('main.py', 'submission.py')), None)
        if chosen is None:
            from .notebook_source import extract
            files=extract(api,notebook,kernel)
            if before != kernel.get_kernel(query).metadata.current_version_number:
                raise ValueError('Notebook changed during source extraction')
            destination=Path(destination).resolve();destination.parent.mkdir(parents=True,exist_ok=True)
            archive=destination.with_suffix('.zip');pack(files,archive);sha=file_hash(archive)
            version=f'{before}-source-{sha[:12]}'
            manifest=dict(name=slug,author=owner,version=version,run=['python','_arena_bridge.py'],sha256=sha,
                          origin=dict(kind='public',notebook=notebook,version=version))
            write(destination,dict(archive_path=str(archive),manifest=manifest))
            return manifest
        blocks = bytearray()
        with requests.get(chosen.url, stream=True, timeout=45) as response:
            response.raise_for_status()
            for block in response.iter_content(1024*1024):
                blocks.extend(block)
                if len(blocks) > LIMIT:
                    raise ValueError('Output exceeds size limit')
        after = kernel.get_kernel(query).metadata.current_version_number
        if before != after:
            raise ValueError('Notebook changed during export; retry next refresh')
    files = unpack(blocks) if chosen.file_name.endswith('.tar.gz') else {'main.py': bytes(blocks)}
    destination = Path(destination).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    archive = destination.with_suffix('.zip')
    pack(files, archive)
    # Hash pins the actual output even if a notebook edit has no new executed output.
    sha = file_hash(archive)
    version = f'{before}-output-{sha[:12]}'
    manifest = dict(name=slug, author=owner, version=version, run=['python','_arena_bridge.py'],
                    sha256=sha, origin=dict(kind='public',notebook=notebook,version=version))
    write(destination, dict(archive_path=str(archive),manifest=manifest))
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('notebook'); parser.add_argument('destination')
    args = parser.parse_args()
    print(json.dumps(export(args.notebook, args.destination)))
