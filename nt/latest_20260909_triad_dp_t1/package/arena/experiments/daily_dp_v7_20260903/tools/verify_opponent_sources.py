"""Verify frozen downloads, published hashes, archive entrypoints and defaults."""
from pathlib import Path
import ast
import hashlib
import json
import tarfile

EXP = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    pool = EXP / "opponents"
    inv = json.loads((pool / "source_inventory_v2.json").read_text(encoding="utf-8"))
    checks = []
    for entry in inv["opponents"]:
        for f in entry["files"]:
            path = EXP / f["path"].replace("\\", "/")
            assert path.stat().st_size == f["bytes"] and sha(path) == f["sha256"], path
            checks.append(str(path.relative_to(EXP)))
    registry = json.loads((pool / "registry.json").read_text(encoding="utf-8"))
    assert len(set(registry["required_opponents"])) == 7
    for key in registry["required_opponents"]:
        item = registry["opponents"][key]
        assert sha(EXP / item["source"]) == item["source_sha256"], key
    published = {
        "boatlee_v29": ("c4a6964cec3c1c99207c32bb1fd91e53c3ec01e6890da5734331cbeab1cc1267", "8dc512911c0173483211314f63cbf1d7e460cad33dfbc02f0f77d023f6d809fe"),
        "kaito_v58": ("b041058ec187a8d0a01edc0eab8de068b53deca3e6c1973faf74ace6916ddcb9", "90c679e02e78cb436128a6029d39c70a9f345912a1c9ff35956c5ee166b2061c"),
    }
    for key, (src, archive) in published.items():
        assert sha(pool / key / "output/main.py") == src
        assert sha(pool / key / "output/submission.tar.gz") == archive
    for key in ("boatlee_v29", "kaito_v58", "lynn_v5", "yhay81_six_day"):
        output = pool / key / "output"
        archive = output / ("sixday-publicstate-agent.tar.gz" if key == "yhay81_six_day" else "submission.tar.gz")
        with tarfile.open(archive) as tf:
            member = next(m for m in tf.getmembers() if m.name.lstrip("./") == "main.py")
            assert tf.extractfile(member).read() == (output / "main.py").read_bytes()
            if key == "lynn_v5":
                for member in tf.getmembers():
                    if member.isfile():
                        path = output / "generated_submission" / member.name
                        assert path.is_file() and path.read_bytes() == tf.extractfile(member).read(), member.name
            if key == "yhay81_six_day":
                member = next(m for m in tf.getmembers() if m.name.lstrip("./") == "agent.so")
                assert tf.extractfile(member).read() == (output / "agent.so").read_bytes()
    yhay_cell = ast.parse((pool / "yhay81_six_day/inspection/cell_002.txt").read_text())
    expected = next(ast.literal_eval(n.value) for n in yhay_cell.body if isinstance(n, ast.Assign)
                    and isinstance(n.targets[0], ast.Name) and n.targets[0].id == "expected")
    for filename, h in expected.items():
        assert sha(pool / "yhay81_six_day/output/sixday_r4_source" / filename) == h
    # Test only our own comparator; never execute opponent or Notebook source.
    tree = ast.parse((EXP / "tools/check_native.py").read_text())
    normalizer = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "normalized")
    ns = {}
    exec(compile(ast.Module(body=[normalizer], type_ignores=[]), "our_comparator_test", "exec"), ns)
    norm = ns["normalized"]
    for op in ("PICKUP", "PLACE"):
        assert norm({"farmer": [op, "WHEAT"]}) == norm({"farmer": [op, "WHEAT", 1]})
        assert norm({"farmer": [op, "WHEAT", 2]}) != norm({"farmer": [op, "WHEAT", 1]})
    before = json.loads((EXP / "receipts/pool_g001_g003_A50_v1/results.json").read_text())
    after = json.loads((EXP / "receipts/pool_g001_g003_A50_v2/results.json").read_text())
    def signatures(doc):
        return [(r['variant'],r['opponent'],r['seed'],r['seat'],r['cash'],r['opponent_cash'],r['opponent_switched']) for r in doc['rows']]
    assert signatures(before) == signatures(after)
    receipt = dict(status="PASS_SOURCE_ACQUISITION_NOT_FULL_POOL_ACCEPTANCE", checked_download_files=len(checks),
                   required_opponents=registry["required_opponents"], published_yhay_source_hashes=len(expected),
                   source_inventory_sha256=sha(pool / "source_inventory_v2.json"), registry_sha256=sha(pool / "registry.json"),
                   archive_entrypoint_parity=True, lynn_full_package_parity=True,
                   quantity_default_and_explicit_difference_tests=4,
                   repeated_panel_economic_results_unchanged=True,
                   opponent_code_executed=False,
                   native_pending=[k for k,v in registry['opponents'].items() if v['runtime']=='pending_native'])
    dest = EXP / "receipts/opponent_pool_source_acceptance_20260903.json"
    if dest.exists():
        raise FileExistsError(dest)
    dest.write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(receipt, ensure_ascii=False))


if __name__ == "__main__":
    main()
