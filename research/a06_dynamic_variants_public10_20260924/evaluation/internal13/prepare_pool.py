"""Extract only each archive's declared primary agent for paired evaluation."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path, PurePosixPath
import tarfile
import zipfile


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
INPUT = REPO / "gpt_review/gpt_code/a06_r12_update"
AGENTS = HERE / "agents"

# The first bundle declares R15C as its best delivered candidate; research
# subagents and historical baselines inside other bundles are not releases.
RELEASES = [
    ("r15c_rules", "48.75_A06_R12_RULE_IMPROVEMENT_20260924.zip", "A06_R12_RULE_IMPROVEMENT_20260924/A06_R15C_agent/"),
    ("r14_cashflow", "54.37_A06_R14_CASHFLOW_candidate_20260924 (1).zip", "A06_R14_CASHFLOW/"),
    ("r14_daily_staff", "56.25A06_R14_noML_RL_20260924.zip", "A06_R14_noML_RL_20260924/agent/"),
    ("r14_f5l3", "59_A06_R14_F5L3_Agent_20260924.zip", ""),
    ("r14_asset_tail", "60.62_A06_R14_AssetTail_20260924.zip", "A06_R14_AssetTail_20260924/agent/"),
    ("r14_frozen", "62.5_A06_R14_agent_source_20260924.zip", "A06_R14_20260924/agent/"),
    ("r14_terminal_switch", "63.75_A06_R14_RULE_TERMINAL_SWITCH_20260924.zip", "A06_R14_RULE_TERMINAL_SWITCH_20260924/agent/"),
    ("rule_future65", "65_A06R12_rule_only_104of160_20260924_full.zip", "A06R12_rule_only_104of160_20260924/agent/"),
    ("r13_nml", "69.38_A06_R13_NoML_NoRL_20260924_SUBMISSION.tar.gz", ""),
    ("rule_r18", "71.87_A06_R12_RULE_R18_20260924_agent.tar.gz", ""),
    ("r14_liquidity", "81.25_A06_R14_LIQUIDITY_20260924.zip", "A06_R14_LIQUIDITY_20260924/work/A06_R14_liquidity/"),
    ("r14_tl5", "85.0_A06R14_TL5_Agent_20260924.zip", "A06R14_TL5/"),
    ("r22_continuation", "A06_R22_NoML_Continuation_20260924.zip", "A06_R22_NoML_Continuation_20260924/agent/"),
]


def sha(data: bytes) -> str:
    return sha256(data).hexdigest()


def safe_relative(name: str, prefix: str) -> Path:
    if not name.startswith(prefix):
        raise ValueError(name)
    relative = PurePosixPath(name[len(prefix):])
    if relative.is_absolute() or not relative.parts or any(part in ("", ".", "..") for part in relative.parts):
        raise ValueError(f"unsafe archive member: {name}")
    return Path(*relative.parts)


def main() -> None:
    AGENTS.mkdir(parents=True, exist_ok=True)
    pool = []
    for ident, archive_name, prefix in RELEASES:
        archive_path = INPUT / archive_name
        destination = AGENTS / ident
        destination.mkdir(parents=True, exist_ok=True)
        files = {}
        if archive_path.suffix == ".zip":
            with zipfile.ZipFile(archive_path) as archive:
                members = ((info.filename, archive.read(info)) for info in archive.infolist()
                           if not info.is_dir() and info.filename.startswith(prefix))
                for name, data in members:
                    relative = safe_relative(name, prefix)
                    target = destination / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    if target.exists() and sha(target.read_bytes()) != sha(data):
                        raise ValueError(f"previous extraction differs: {target}")
                    target.write_bytes(data)
                    files[relative.as_posix()] = sha(data)
        else:
            with tarfile.open(archive_path, "r:gz") as archive:
                for member in archive.getmembers():
                    if not member.isfile() or not member.name.startswith(prefix):
                        continue
                    relative = safe_relative(member.name, prefix)
                    data = archive.extractfile(member).read()
                    target = destination / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    if target.exists() and sha(target.read_bytes()) != sha(data):
                        raise ValueError(f"previous extraction differs: {target}")
                    target.write_bytes(data)
                    files[relative.as_posix()] = sha(data)
        for required in ("main.py", "policy/a06.so", "policy/config.json"):
            if required not in files:
                raise ValueError(f"{ident} missing {required}")
        pool.append({
            "id": ident,
            "archive": archive_name,
            "archive_sha256": sha(archive_path.read_bytes()),
            "source_prefix": prefix,
            "entry": (destination / "main.py").relative_to(REPO).as_posix(),
            "working": destination.relative_to(REPO).as_posix(),
            "files": files,
        })
        print(f"{ident}: {len(files)} files, native {files['policy/a06.so'][:16]}", flush=True)
    output = HERE / "POOL.json"
    content = json.dumps(pool, ensure_ascii=False, indent=2) + "\n"
    if output.exists() and output.read_text(encoding="utf-8") != content:
        raise ValueError("frozen pool already exists with different hashes")
    output.write_text(content, encoding="utf-8")
    print(f"POOL: {len(pool)} release agents")


if __name__ == "__main__":
    main()
