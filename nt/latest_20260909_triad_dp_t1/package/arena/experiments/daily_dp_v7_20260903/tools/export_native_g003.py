"""Statically export G003 after proving it shares G001's execution semantics.

No downloaded Python code is executed. Reuse is permitted only with identical
AST definitions AND equivalent embedded execution-stack AST. Trees and every
route are exported from G003 itself, never from its example replay.
"""
import ast
import base64
import hashlib
import json
import zlib
from pathlib import Path

from prepare_opponent_sources import literal_unwrap, put, sha

EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]
G001 = ROOT / "research/team_mate/Kaggriculture_main_512631c/agents/route_clustering_switch_agent/main.py"
G003 = EXP / "opponents/g003/source/main.py"


def static_bundle(path):
    data = path.read_bytes()
    decoded = literal_unwrap(data)
    assert decoded is not None
    tree = ast.parse(decoded)
    definitions = {n.name: ast.dump(n, include_attributes=False) for n in tree.body
                   if isinstance(n, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef))}
    assignments = {n.targets[0].id: n.value for n in tree.body if isinstance(n, ast.Assign)
                   and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name)}
    def decompress(name):
        node = assignments[name]
        data = [n.value for n in ast.walk(node) if isinstance(n, ast.Constant) and isinstance(n.value, bytes)]
        assert len(data) == 1
        return zlib.decompress(base64.b85decode(data[0]))
    return dict(hash=sha(data), definitions=definitions, base=decompress("_S_BASE"),
                actions=json.loads(decompress("_S_ACTIONS")),
                metadata=ast.literal_eval(assignments["_S_METADATA"]),
                policy=ast.literal_eval(assignments["_S_POLICY_PAYLOAD"]),
                nash=ast.literal_eval(assignments["_S_NASH"]))


def main():
    old, new = static_bundle(G001), static_bundle(G003)
    assert old["hash"] == "9b4fdf7a7c92d3eefc75c693c4949825567588478bef0e53529bdec82944dda7"
    assert new["hash"] == "ff32e2e47431dae2d98fa70a78f85425f8d5929d5910c5ff482e74750c3fa1ef"
    assert old["definitions"] == new["definitions"], "Different runtime definitions: native port requires review"
    assert ast.dump(ast.parse(old["base"]), include_attributes=False) == ast.dump(ast.parse(new["base"]), include_attributes=False), "Different embedded executor semantics: cannot reuse G001 backend"
    assert new["policy"]["feature_schema"] == "semantic_route_switch_v1"
    support = [r for r in new["nash"]["opening_support"] if r["weight"] > 0]
    assert len(support) == 1 and support[0]["family"] == "G003"
    routes = {str(v["family"]): str(v["route_id"]) for v in new["metadata"]["opponent_routes"]}
    families = list(routes)
    targets = tuple(new["policy"].get("targets", ()))
    nodes = []
    for value in new["policy"]["nodes"]:
        selected = value["selected"]
        if not selected.get("enabled", True) or selected["opening"] != "G003":
            continue
        tree = selected["tree"]
        mapped = [str(v) if str(v) in routes else str(targets[int(v)]) for v in tree["classes"]]
        nodes.append(dict(checkpoint=int(selected["checkpoint"]), targets=[families.index(v) for v in mapped],
                          **{k: tree[k] for k in ("left", "right", "feature", "threshold", "value")}))
    nodes.sort(key=lambda n: n["checkpoint"])
    old_assets = json.loads(zlib.decompress((EXP / "native/g001_frozen.json.zlib").read_bytes()))
    assert old_assets["source_sha256"] == old["hash"]
    payload = dict(source_sha256=new["hash"], families=families, opening=families.index("G003"), nodes=nodes,
                   routes=[new["actions"][routes[f]] for f in families],
                   **{k: old_assets[k] for k in ("r5", "md", "moon", "moon_legacy")})
    assert all(len(r) == 719 for r in payload["routes"])
    blob = zlib.compress(json.dumps(payload, separators=(",", ":")).encode(), 9)
    dest = EXP / "native/g003_frozen.json.zlib"
    put(dest, blob)
    receipt = dict(source_sha256=new["hash"], asset_sha256=sha(blob), families=len(families),
                   opening="G003", checkpoints=[n["checkpoint"] for n in nodes], bytes=len(blob),
                   identical_runtime_definitions=True, identical_embedded_executor_ast=True,
                   identical_embedded_executor_bytes=old["base"] == new["base"],
                   embedded_executor_sha256=sha(new["base"]),
                   definition_count=len(new["definitions"]), source_executed=False,
                   native_backend="same searched-route executor; distinct complete G003 route library and trees",
                   parity_status="PENDING")
    put(EXP / "receipts/native_g003_static_export.json", json.dumps(receipt, indent=2).encode())
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
