"""Run the local Kaggriculture Replay diagnostic simulator."""

from __future__ import annotations

import argparse
import gzip
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import sys
import threading
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse
import webbrowser


PROJECT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = PROJECT_DIR / "src"
STATIC_DIR = PROJECT_DIR / "static"
VENDORED_OFFICIAL_VIEWER = (
    PROJECT_DIR
    / "vendor"
    / "official_1_32_7_visualizer"
    / "index.html"
)
WORKSPACE_OFFICIAL_VIEWER = (
    REPO_ROOT
    / "research"
    / "official_env_update_20260815"
    / "extracted_1_32_7"
    / "kaggle_environments"
    / "envs"
    / "kaggriculture"
    / "visualizer"
    / "default"
    / "dist"
    / "index.html"
)
DEFAULT_OFFICIAL_VIEWER = (
    VENDORED_OFFICIAL_VIEWER
    if VENDORED_OFFICIAL_VIEWER.is_file()
    else WORKSPACE_OFFICIAL_VIEWER
)

sys.path.insert(0, str(SRC_DIR))
from kaggriculture_replay_diagnostic_simulator_v1 import (  # noqa: E402
    ReplayDiagnostics,
    ReplayValidationError,
)


class ReplayStore:
    def __init__(self, official_viewer_path: Path) -> None:
        self._lock = threading.RLock()
        self._official_template = official_viewer_path.read_text(encoding="utf-8")
        self._diagnostics: ReplayDiagnostics | None = None
        self._viewer_html: bytes | None = None
        self._generation = 0

    def load_path(self, path: str) -> dict[str, Any]:
        diagnostics = ReplayDiagnostics.from_path(path)
        return self._activate(diagnostics)

    def load_json(self, replay: dict[str, Any], source_name: str) -> dict[str, Any]:
        diagnostics = ReplayDiagnostics(replay, source_name)
        return self._activate(diagnostics)

    def _activate(self, diagnostics: ReplayDiagnostics) -> dict[str, Any]:
        meta = diagnostics.meta()
        replay_json = json.dumps(
            diagnostics.replay,
            ensure_ascii=True,
            separators=(",", ":"),
        ).replace("</script", "<\\/script")
        agents_json = json.dumps(
            [
                {"index": index, "name": name}
                for index, name in enumerate(meta["players"])
            ],
            ensure_ascii=True,
            separators=(",", ":"),
        ).replace("</script", "<\\/script")
        bootstrap = (
            "<script>window.kaggle={environment:"
            + replay_json
            + ",step:0};window.kaggle.environment.info.Agents="
            + agents_json
            + ";</script>"
        )
        if "<head>" not in self._official_template:
            raise ReplayValidationError("官方 visualizer 模板缺少 <head>")
        viewer_html = self._official_template.replace(
            "<head>", "<head>" + bootstrap, 1
        ).encode("utf-8")
        with self._lock:
            self._diagnostics = diagnostics
            self._viewer_html = viewer_html
            self._generation += 1
            meta["generation"] = self._generation
            return meta

    def _require(self) -> ReplayDiagnostics:
        with self._lock:
            if self._diagnostics is None:
                raise ReplayValidationError("尚未载入 Replay")
            return self._diagnostics

    def meta(self) -> dict[str, Any]:
        diagnostics = self._require()
        meta = diagnostics.meta()
        with self._lock:
            meta["generation"] = self._generation
        return meta

    def frame(self, step: int) -> dict[str, Any]:
        return self._require().frame(step)

    def timeline(self) -> list[dict[str, Any]]:
        return self._require().timeline()

    def viewer_html(self) -> bytes:
        self._require()
        with self._lock:
            assert self._viewer_html is not None
            return self._viewer_html


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode(
        "utf-8"
    )


class SimulatorHandler(BaseHTTPRequestHandler):
    server_version = "KaggricultureDiagnosticSimulator/1.0"

    @property
    def store(self) -> ReplayStore:
        return self.server.store  # type: ignore[attr-defined]

    def log_message(self, fmt: str, *args: Any) -> None:
        sys.stderr.write("[simulator] " + fmt % args + "\n")

    def _send(
        self,
        body: bytes,
        content_type: str,
        status: HTTPStatus = HTTPStatus.OK,
        cache: str = "no-store",
    ) -> None:
        use_gzip = (
            len(body) >= 4096
            and "gzip" in self.headers.get("Accept-Encoding", "")
        )
        payload = gzip.compress(body, compresslevel=4) if use_gzip else body
        self.send_response(status.value)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", cache)
        self.send_header("X-Content-Type-Options", "nosniff")
        if use_gzip:
            self.send_header("Content-Encoding", "gzip")
        self.end_headers()
        self.wfile.write(payload)

    def _send_json(
        self, value: Any, status: HTTPStatus = HTTPStatus.OK
    ) -> None:
        self._send(_json_bytes(value), "application/json; charset=utf-8", status)

    def _error(self, exc: Exception, status: HTTPStatus = HTTPStatus.BAD_REQUEST) -> None:
        self._send_json({"ok": False, "error": str(exc)}, status)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/":
                return self._serve_static("index.html", "text/html; charset=utf-8")
            if parsed.path == "/app.js":
                return self._serve_static("app.js", "text/javascript; charset=utf-8")
            if parsed.path == "/styles.css":
                return self._serve_static("styles.css", "text/css; charset=utf-8")
            if parsed.path == "/api/meta":
                return self._send_json({"ok": True, "meta": self.store.meta()})
            if parsed.path == "/api/timeline":
                return self._send_json({"ok": True, "timeline": self.store.timeline()})
            if parsed.path == "/api/frame":
                query = parse_qs(parsed.query)
                step = int(query.get("step", ["0"])[0])
                return self._send_json({"ok": True, "frame": self.store.frame(step)})
            if parsed.path == "/official-viewer":
                return self._send(
                    self.store.viewer_html(),
                    "text/html; charset=utf-8",
                    cache="no-store",
                )
            self._error(FileNotFoundError(parsed.path), HTTPStatus.NOT_FOUND)
        except (ReplayValidationError, ValueError, OSError) as exc:
            self._error(exc)

    def _serve_static(self, name: str, content_type: str) -> None:
        path = STATIC_DIR / name
        self._send(path.read_bytes(), content_type)

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0:
                raise ReplayValidationError("请求内容为空")
            if length > 250 * 1024 * 1024:
                raise ReplayValidationError("Replay 超过 250MB 限制")
            body = self.rfile.read(length)
            if parsed.path == "/api/load":
                data = json.loads(body.decode("utf-8"))
                path = str(_as_mapping(data).get("path", "")).strip()
                if not path:
                    raise ReplayValidationError("请输入 Replay 文件路径")
                meta = self.store.load_path(path)
                return self._send_json({"ok": True, "meta": meta})
            if parsed.path == "/api/upload":
                data = json.loads(body.decode("utf-8"))
                query = parse_qs(parsed.query)
                name = unquote(query.get("name", ["browser-upload.json"])[0])
                meta = self.store.load_json(data, f"浏览器上传：{name}")
                return self._send_json({"ok": True, "meta": meta})
            self._error(FileNotFoundError(parsed.path), HTTPStatus.NOT_FOUND)
        except (ReplayValidationError, UnicodeDecodeError, json.JSONDecodeError, OSError) as exc:
            self._error(exc)


def _as_mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


class SimulatorServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address: tuple[str, int], store: ReplayStore) -> None:
        super().__init__(address, SimulatorHandler)
        self.store = store


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Kaggriculture 官方 Replay + 动作/报警诊断模拟器"
    )
    parser.add_argument("--replay", help="启动时载入的 Replay JSON")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--official-viewer",
        type=Path,
        default=DEFAULT_OFFICIAL_VIEWER,
        help="官方 1.32.7 visualizer index.html",
    )
    parser.add_argument("--open", action="store_true", help="启动后打开浏览器")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    viewer_path = args.official_viewer.resolve()
    if not viewer_path.is_file():
        print(f"官方 visualizer 不存在：{viewer_path}", file=sys.stderr)
        return 2
    store = ReplayStore(viewer_path)
    if args.replay:
        try:
            meta = store.load_path(args.replay)
            print(
                f"已载入 Episode {meta.get('episode_id')}，"
                f"{meta.get('players')}，{meta.get('steps')} frames"
            )
        except ReplayValidationError as exc:
            print(f"Replay 载入失败：{exc}", file=sys.stderr)
            return 2
    server = SimulatorServer((args.host, args.port), store)
    url = f"http://{args.host}:{args.port}/"
    print(f"Kaggriculture 诊断模拟器：{url}")
    print("按 Ctrl+C 停止。")
    if args.open:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
