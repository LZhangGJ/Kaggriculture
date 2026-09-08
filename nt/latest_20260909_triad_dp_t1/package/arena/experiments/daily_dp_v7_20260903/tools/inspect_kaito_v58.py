"""Bounded static extraction of embedded source. Never executes the agent."""
from pathlib import Path
import ast
import base64
import hashlib
import json
import re
import zlib
from prepare_opponent_sources import put

EXP=Path(__file__).resolve().parents[1]
EXPECTED="b041058ec187a8d0a01edc0eab8de068b53deca3e6c1973faf74ace6916ddcb9"


def decode(node):
    found=[n.value for n in ast.walk(node) if isinstance(n,ast.Constant)
           and isinstance(n.value,(str,bytes)) and len(n.value)>1000]
    assert len(found)==1
    return zlib.decompress(base64.b85decode(found[0]))


def main():
    src=EXP/"opponents/kaito_v58/output/main.py";raw=src.read_bytes()
    assert hashlib.sha256(raw).hexdigest()==EXPECTED
    tree=ast.parse(raw)
    assigns={n.targets[0].id:n.value for n in tree.body if isinstance(n,ast.Assign)
             and len(n.targets)==1 and isinstance(n.targets[0],ast.Name)}
    out=EXP/"opponents/kaito_v58/inspection/static_bundle_v1"
    base=decode(assigns["_V51_BASE_SOURCE"])
    put(out/"v50_entry.py",base)
    base_tree=ast.parse(base)
    summary=dict(source_sha256=EXPECTED,base_bytes=len(base),
                 base_assignments=[dict(name=ast.unparse(n.targets[0]),kind=type(n.value).__name__,line=n.lineno)
                                   for n in base_tree.body if isinstance(n,ast.Assign)],
                 base_definitions=[dict(name=n.name,line=n.lineno,end=n.end_lineno)
                                   for n in base_tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))])
    put(out/"inspection.json",json.dumps(summary,indent=2).encode())
    print(json.dumps(summary),flush=True)
    layers=[]
    def extract(data,layer,depth=0):
        assert depth<12
        t=ast.parse(data)
        assignments={n.targets[0].id:n.value for n in t.body if isinstance(n,ast.Assign)
                     and len(n.targets)==1 and isinstance(n.targets[0],ast.Name)}
        item=dict(layer=layer,bytes=len(data),assignments=list(assignments),modules=[])
        for name,value in assignments.items():
            if name.endswith("_BASE_SOURCE"):
                child=decode(value);label=name.strip("_").lower()
                put(out/(label+".py"),child);extract(child,label,depth+1)
            elif name.endswith("_MODULES"):
                modules=json.loads(decode(value))
                for module,code in modules.items():
                    assert re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.]*",module)
                    assert isinstance(code,str)
                    dest=out/"modules"/(layer+"__"+module+".py")
                    put(dest,code.encode())
                    item["modules"].append(dict(name=module,bytes=len(code.encode()),lines=len(code.splitlines()),path=str(dest.relative_to(EXP))))
        # Suppress data literals but retain all executable functions verbatim.
        runtime=[]
        lines=data.decode().splitlines()
        for n in t.body:
            if isinstance(n,(ast.FunctionDef,ast.ClassDef,ast.Import,ast.ImportFrom)):
                runtime.extend(lines[n.lineno-1:n.end_lineno]);runtime.append("")
        put(out/(layer+"_runtime.py"),"\n".join(runtime).encode())
        layers.append(item)
    extract(base,"v50_entry")
    put(out/"layers_summary.json",json.dumps(layers,indent=2).encode())
    print(json.dumps(layers),flush=True)


if __name__=="__main__":main()
