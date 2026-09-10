"""Submit the explicitly authorized JointAFS R1 archive once and save receipts."""
from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import os
import sys
import time


HERE = Path(__file__).resolve().parent
FOLDER = HERE / "AFS"
COMPETITION = "kaggriculture"
MESSAGE = "NT P16 JointAFS R1 - joint animal feed successor planning"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def save(path: Path, value, overwrite: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w" if overwrite else "x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)


def api_client():
    token = Path.home() / ".kaggle/access_token"
    if not os.environ.get("KAGGLE_API_TOKEN") and token.is_file():
        os.environ["KAGGLE_API_TOKEN"] = token.read_text(encoding="utf-8").strip()
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    return api


def preflight(api=None):
    api = api or api_client()
    response = api.competitions_list(search=COMPETITION)
    competitions = getattr(response, "competitions", None) or response
    competition = next(x for x in competitions if str(x.ref).rstrip("/").endswith(COMPETITION))
    limit = int(getattr(competition, "max_daily_submissions", 0)) or None
    official = api.competition_get_submission_limits(COMPETITION)
    recent = list(api.competition_submissions(COMPETITION, page_size=100))
    today = datetime.now(timezone.utc).date()
    used = sum(x.date.date() == today for x in recent)
    result = {
        "date_utc": str(today),
        "limit": limit,
        "visible_used_today": used,
        "visible_remaining": max(0, limit - used) if limit else None,
        "official_team_limits": {
            "num_today": official.num_today,
            "num_total": official.num_total,
            "num_allowed_now": official.num_allowed_now,
            "limited_by_total": official.limited_by_total,
        },
        "duplicates": [x.ref for x in recent if x.description == MESSAGE],
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    save(HERE / "PREFLIGHT_QUOTA.json", result, overwrite=True)
    return result


def validate():
    build = read(FOLDER / "BUILD_RECEIPT.json")
    identity = read(FOLDER / "ACTION_IDENTITY.json")
    package = read(FOLDER / "PACKAGE_INTEGRITY.json")
    official = read(FOLDER / "OFFICIAL_FILE_ACCEPTANCE.json")
    assert build["status"] == "PASS_PORTABLE_GCC11"
    assert identity["status"] == package["status"] == official["status"] == "PASS"
    assert identity["games_per_binary"] == 22 and not identity["differences"]
    assert sha(FOLDER / "submission.tar.gz") == package["archive_sha256"] == official["archive_sha256"]
    assert sha(FOLDER / "agent.so") == build["binary_sha256"]
    assert len(official["games"]) == 2 and official["errors"] == 0
    assert all(
        game["status"] == "PASS" and game["steps"] == 719 and game["actions_over_1s"] == 0
        for game in official["games"]
    )
    return build, identity, package, official


def submit():
    build, identity, package, official = validate()
    if (FOLDER / "UPLOAD_ATTEMPT.json").exists():
        raise RuntimeError("Upload attempt already exists; inspect status instead of retrying")
    api = api_client()
    flight = preflight(api)
    if flight["duplicates"]:
        raise RuntimeError(f"Named AFS version already submitted: {flight['duplicates']}")
    if flight["visible_remaining"] is not None and flight["visible_remaining"] <= 0:
        raise RuntimeError("No visible remaining daily quota")
    if flight["official_team_limits"]["num_allowed_now"] <= 0:
        raise RuntimeError("Official team quota exhausted")
    save(
        FOLDER / "UPLOAD_ATTEMPT.json",
        {
            "status": "STARTING_SINGLE_UPLOAD",
            "message": MESSAGE,
            "archive_sha256": package["archive_sha256"],
            "time_utc": datetime.now(timezone.utc).isoformat(),
        },
    )
    response = api.competition_submit(
        str(FOLDER / "submission.tar.gz"), MESSAGE, COMPETITION, quiet=True
    )
    if not response.ref:
        raise RuntimeError(response.message)
    record = {
        "ref": response.ref,
        "message": MESSAGE,
        "competition": COMPETITION,
        "upload_message": response.message,
        "upload_calls": 1,
        "submitted_at_utc": datetime.now(timezone.utc).isoformat(),
        "archive_sha256": package["archive_sha256"],
        "archive_bytes": package["bytes"],
        "binary_sha256": build["binary_sha256"],
        "reference_binary_sha256": build["reference_binary_sha256"],
        "local_action_identity_games": identity["games_per_binary"],
        "local_official_file_games": len(official["games"]),
        "max_local_action_ms": official["max_action_ms"],
        "submission_url": f"https://www.kaggle.com/competitions/{COMPETITION}/submissions?submissionId={response.ref}",
    }
    save(FOLDER / "submission_record.json", record)
    print(json.dumps(record, ensure_ascii=False), flush=True)


def status(wait_seconds: int = 0):
    api = api_client()
    deadline = time.monotonic() + wait_seconds
    while True:
        if not (FOLDER / "submission_record.json").exists():
            raise RuntimeError("No submission_record.json")
        record = read(FOLDER / "submission_record.json")
        recent = list(api.competition_submissions(COMPETITION, page_size=100))
        row = next(x for x in recent if x.ref == record["ref"])
        state = str(row.status).split(".")[-1]
        result = {
            "ref": row.ref,
            "status": state,
            "public_score": row.public_score,
            "error_description": row.error_description,
            "description": row.description,
            "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        save(FOLDER / "latest_status.json", result, overwrite=True)
        with (FOLDER / "STATUS_HISTORY.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(result, ensure_ascii=False) + "\n")
        print(json.dumps(result, ensure_ascii=False), flush=True)
        if state.upper() in {"COMPLETE", "ERROR", "CANCELLED"} or time.monotonic() >= deadline:
            return
        time.sleep(20)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "status"
    if mode == "preflight":
        print(json.dumps(preflight(), ensure_ascii=False))
    elif mode == "submit":
        submit()
    elif mode == "status":
        wait = int(sys.argv[2]) if len(sys.argv) > 2 else 0
        status(wait)
    else:
        raise SystemExit("Use: release.py | submit_once.py preflight|submit|status [seconds]")
