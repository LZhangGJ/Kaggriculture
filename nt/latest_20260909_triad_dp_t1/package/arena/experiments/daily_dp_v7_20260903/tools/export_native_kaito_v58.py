"""Freeze V58's ten active priors/config without executing downloaded code."""
from pathlib import Path
import ast
import base64
import hashlib
import json
import zlib
from export_native_boatlee_v29 import put

EXP = Path(__file__).resolve().parents[1]
EXPECTED = "b041058ec187a8d0a01edc0eab8de068b53deca3e6c1973faf74ace6916ddcb9"
MODULES = EXP / "opponents/kaito_v58/inspection/static_bundle_v1/modules"


def main():
    source = EXP / "opponents/kaito_v58/output/main.py"
    assert hashlib.sha256(source.read_bytes()).hexdigest() == EXPECTED
    tree = ast.parse(source.read_text())
    values = {n.targets[0].id: n.value for n in tree.body if isinstance(n, ast.Assign)
              and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name)}

    def read(name):
        node = values[name]
        if isinstance(node, ast.Name):
            return read(node.id)
        if isinstance(node, ast.Call):
            if ast.unparse(node.func) == "frozenset":
                assert len(node.args) == 1 and not node.keywords
                return sorted(ast.literal_eval(node.args[0]))
            assert ast.unparse(node.func) == "json.loads", name
            assert "zlib.decompress" in ast.unparse(node) and "base64.b85decode" in ast.unparse(node)
            encoded = [n.value for n in ast.walk(node) if isinstance(n, ast.Constant)
                       and isinstance(n.value, str) and len(n.value) > 1000]
            assert len(encoded) == 1
            return json.loads(zlib.decompress(base64.b85decode(encoded[0])))
        return ast.literal_eval(node)

    routes = []
    for pair in values["_V58_POLICY_ROUTES"].elts:
        name, variable = pair.elts
        actions = read(variable.id)
        assert len(actions) == 719
        routes.append(dict(name=ast.literal_eval(name), variable=variable.id, actions=actions))
    assert len(routes) == 10
    config_node = values["_V54_CONFIG"]
    assert ast.unparse(config_node.func) == "_V51ResidualConfig"
    assert len(config_node.keywords) == 1 and config_node.keywords[0].arg is None
    overrides = ast.literal_eval(config_node.keywords[0].value)
    module = ast.parse((MODULES / "v50_base_source__v49.residual_controller.py").read_text())
    cls = next(n for n in module.body if isinstance(n, ast.ClassDef) and n.name == "ResidualConfig")
    config = {n.target.id: ast.literal_eval(n.value) for n in cls.body if isinstance(n, ast.AnnAssign)}
    config.update(overrides)
    assert not config["defer_enabled"] and not config["market_maker_enabled"]
    assert config["terminal_rule"] == "collision"
    assert config["preempt_items"] == config["front_items"]
    payload = dict(source_sha256=EXPECTED, routes=routes, config=config,
                   pet_second_shops=read("_V57_PET_SECOND_SHOPS"),
                   official_configuration=dict(turnsPerDay=24, shedCapacity=100,
                                               townShopSellInterval=4, townCenterSellInterval=24))
    # JSON needs a deterministic representation for the source's set literal.
    payload["pet_second_shops"] = sorted(payload["pet_second_shops"])
    blob = zlib.compress(json.dumps(payload, separators=(",", ":")).encode(), 9)
    put(EXP / "native/kaito_v58_frozen.json.zlib", blob)
    receipt = dict(status="STATIC_EXPORT_ONLY", source_sha256=EXPECTED,
                   asset_sha256=hashlib.sha256(blob).hexdigest(), bytes=len(blob),
                   routes=[dict(name=r["name"], variable=r["variable"], actions=len(r["actions"])) for r in routes],
                   config=config, source_executed=False, native_parity="PENDING")
    put(EXP / "receipts/native_kaito_v58_static_export.json", json.dumps(receipt, indent=2).encode())
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
