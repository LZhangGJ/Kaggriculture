"""Static data export only; never executes the downloaded notebook or agent."""
from pathlib import Path
import ast
import base64
import hashlib
import json
import zlib

EXP = Path(__file__).resolve().parents[1]
EXPECTED = "c4a6964cec3c1c99207c32bb1fd91e53c3ec01e6890da5734331cbeab1cc1267"


def put(path, data):
    if path.exists():
        assert path.read_bytes() == data, path
    else:
        path.write_bytes(data)


def main():
    source = EXP / "opponents/boatlee_v29/output/main.py"
    assert hashlib.sha256(source.read_bytes()).hexdigest() == EXPECTED
    tree = ast.parse(source.read_text())
    values = {n.targets[0].id: n.value for n in tree.body if isinstance(n, ast.Assign)
              and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name)}
    action_expr = ast.dump(values["_ACTIONS"], include_attributes=False)
    assert "b85decode" in action_expr and "decompress" in action_expr and "loads" in action_expr
    encoded = [n.value for n in ast.walk(values["_ACTIONS"]) if isinstance(n, ast.Constant)
               and isinstance(n.value, (bytes, str)) and len(n.value) > 1000]
    assert len(encoded) == 1
    actions = json.loads(zlib.decompress(base64.b85decode(encoded[0])))
    # This source has 720 entries, including a terminal unused entry. Preserve
    # the original indexing and clamp; a valid season calls entries 0..718.
    assert len(actions) == 720
    payload = {name: ast.literal_eval(values[name]) for name in
               ("_FR_ITEMS", "_WEED_REPLAY_STEPS", "_SHOP_PRODUCTS", "_AM_ITEMS", "_AM_BASE_PRICE", "_AM_CONFIG")}
    assert not payload["_FR_ITEMS"], "Unported enabled front-run branch: stop rather than substitute"
    payload.update(source_sha256=EXPECTED, actions=actions)
    blob = zlib.compress(json.dumps(payload, separators=(",", ":")).encode(), 9)
    put(EXP / "native/boatlee_v29_frozen.json.zlib", blob)
    receipt = dict(source_sha256=EXPECTED, asset_sha256=hashlib.sha256(blob).hexdigest(),
                   action_count=len(actions), season_action_calls=719, bytes=len(blob), source_executed=False,
                   disabled_fr_items=payload["_FR_ITEMS"], adaptive_market_config=payload["_AM_CONFIG"],
                   native_parity="PENDING")
    put(EXP / "receipts/native_boatlee_v29_static_export.json", json.dumps(receipt, indent=2).encode())
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
