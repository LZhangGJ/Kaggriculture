"""Extract supported public source formats as data, without executing cells."""
import ast
import base64
import copy
import hashlib
import json
import io
import zipfile
from pathlib import Path
import re
import tempfile
import zlib

import requests


def download(url, limit=16*1024*1024):
    data = bytearray()
    with requests.get(url, stream=True, timeout=45) as response:
        response.raise_for_status()
        for block in response.iter_content(65536):
            data.extend(block)
            if len(data)>limit:raise ValueError('Source dependency too large')
    return bytes(data)


def inflate(data):
    decoder=zlib.decompressobj()
    out=decoder.decompress(data,64*1024*1024)
    if not decoder.eof:raise ValueError('Compressed source exceeds limit or is incomplete')
    return out


def extract(api, notebook, kernel):
    with tempfile.TemporaryDirectory() as temp:
        api.kernels_pull(notebook, path=temp, metadata=False)
        paths=list(Path(temp).glob('*.ipynb'))
        if len(paths)!=1:raise ValueError('No unambiguous notebook source')
        cells=[''.join(c.get('source',[])) for c in json.loads(paths[0].read_text(encoding='utf-8'))['cells'] if c['cell_type']=='code']
    outputs=[c.split('\n',1)[1] for c in cells if c.split('\n',1)[0].strip()=='%%writefile main.py']
    if len(outputs)==1:
        data=outputs[0].replace('\r\n','\n').encode()
        expected=re.findall(r"EXPECTED_(?:MAIN_)?SHA256\s*=\s*['\"]([a-f0-9]{64})['\"]", '\n'.join(cells))
        if expected and hashlib.sha256(data).hexdigest() not in expected:
            raise ValueError('Notebook source hash mismatch')
        ast.parse(data)
        return {'main.py':data}
    if notebook!='ahmedberatozer/kaggriculture-v23-adaptive-routes-smart-sales':
        raise ValueError('Unsupported notebook source format; do not execute cells')
    # Reproduce this published, hash-pinned data assembly recipe explicitly.
    from kagglesdk.kernels.types.kernels_api_service import ApiDownloadKernelOutputRequest
    q=ApiDownloadKernelOutputRequest();q.owner_slug='thomastschinkel'
    q.kernel_slug='kaggriculture-95-5-win-rate-via-replay-routing';q.version_number=2;q.file_path='main.py'
    with kernel.download_kernel_output(q) as response:
        response.raise_for_status();data=bytearray()
        for chunk in response.iter_content(65536):
            data.extend(chunk)
            if len(data)>16*1024*1024:raise ValueError('Donor output too large')
    donor=bytes(data)
    if zipfile.is_zipfile(io.BytesIO(donor)):
        with zipfile.ZipFile(io.BytesIO(donor)) as z:
            info=z.getinfo('main.py')
            if info.file_size>16*1024*1024:raise ValueError('Donor source too large')
            donor=z.read(info)
    if hashlib.sha256(donor).hexdigest()!='8241246765098c50223d897739cb32076b71a303284d2e5e1c5f0fb495bb5a97':
        raise ValueError('v23 donor changed')
    assignment=next(n for n in ast.parse(donor).body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Tuple)
                    and [getattr(x,'id',None) for x in n.targets[0].elts]==['SCHEDULES','POLICY'])
    blob=next(n.value for n in ast.walk(assignment.value) if isinstance(n,ast.Constant) and isinstance(n.value,str))
    schedules,_=json.loads(inflate(base64.b85decode(blob)))
    template=None
    for cell in cells:
        if 'V23_TEMPLATE_B64 =' not in cell:continue
        node=next(n for n in ast.parse(cell).body if isinstance(n,ast.Assign) and getattr(n.targets[0],'id',None)=='V23_TEMPLATE_B64')
        encoded=''.join(ast.literal_eval(node.value).split()).replace('j1upVZZ0tyhW','j1upVZ0tyhW').replace('xHeLNAMGEliRp','xHeLNAMdliRp')
        raw=inflate(base64.b64decode(encoded,validate=True))
        if hashlib.sha256(raw).hexdigest()!='985d07220622767f8f90351c1dfd82c4b56c43407b3168692672e0435c26d841':
            raise ValueError('v23 template changed')
        template=raw.decode()
    if template is None:raise ValueError('Missing v23 template')
    license_text=download('https://www.apache.org/licenses/LICENSE-2.0.txt',65536).decode().replace('\r\n','\n')
    if hashlib.sha256(license_text.encode()).hexdigest()!='cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30':
        raise ValueError('License hash mismatch')
    payload={'base':copy.deepcopy(schedules[0]),'patches':{}}
    for branch in (1,2,3,4):
        tape=copy.deepcopy(schedules[branch]);tape[:144]=payload['base'][:144]
        if branch in (3,4):tape[144:288]=copy.deepcopy(schedules[2][144:288])
        payload['patches'][str(branch)]=[[i,a] for i,a in enumerate(tape) if a!=payload['base'][i]]
    blob=base64.b85encode(zlib.compress(json.dumps(payload,separators=(',',':')).encode(),9)).decode()
    notice='# SPDX-License-Identifier: Apache-2.0\n# v23 modifications: public production router, audited Python chassis, and build tooling.\n# Credits: thomastschinkel, yhay81, tetsutani; offline simulator: destbreso/nikital7.\n'
    notice+=''.join('# '+line+'\n' for line in license_text.splitlines())
    data=(notice+template.replace('__V23_ROUTE_BLOB__',repr(blob))).encode()
    if hashlib.sha256(data).hexdigest()!='6eb728a40cc55f7e497add6ea2946ce38b0389c24531a208219d48587435838e':
        raise ValueError('Reconstructed v23 source hash mismatch')
    return {'main.py':data,'LICENSE.txt':license_text.encode()}
