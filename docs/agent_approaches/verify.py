"""Verify the published artifact bytes without credentials or native execution."""

from pathlib import Path
import hashlib
import json
import tarfile


ROOT = Path(__file__).resolve().parents[2]


def main():
    manifest = json.loads(Path(__file__).with_name("ARTIFACTS.json").read_text(encoding="utf-8"))
    unchanged = 0
    for item in manifest["files"]:
        data = (ROOT / item["path"]).read_bytes()
        assert len(data) == item["bytes"], item["path"]
        assert hashlib.sha256(data).hexdigest() == item["sha256"], item["path"]
        if item.get("matches_source_blob"):
            blob = b"blob " + str(len(data)).encode() + b"\0" + data
            assert hashlib.sha1(blob).hexdigest() == item["source_git_blob"], item["path"]
            unchanged += 1
    for item in manifest["archive_members"]:
        with tarfile.open(ROOT / item["archive"], "r:gz") as archive:
            member = archive.extractfile(item["member"])
            assert member is not None, item["member"]
            data = member.read()
        assert hashlib.sha256(data).hexdigest() == item["sha256"], item["member"]
    print(json.dumps({"status": "PASS", "checked_files": len(manifest["files"]),
                      "unchanged_source_blobs": unchanged,
                      "checked_archive_members": len(manifest["archive_members"]),
                      "scope": "Artifact bytes and recorded identities; no matches rerun."}, indent=2))


if __name__ == "__main__":
    main()
