"""Validate the repository's agent-facing module and experiment documents."""

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
ROOT_SECTIONS = ("仓库地图", "模块索引", "实验索引", "数据索引", "上传规则", "文档模板")
MODULE_SECTIONS = ("一句话说明", "状态", "代码地图", "输入与输出", "验证", "结果", "已知问题", "相关实验")
EXPERIMENT_SECTIONS = ("一句话结论", "状态", "问题", "代码位置", "数据位置", "方法", "结果", "复现", "局限", "下一步")


def validate_document(path: Path, sections: tuple[str, ...], errors: list[str]) -> None:
    if not path.is_file():
        errors.append(f"missing: {path.relative_to(ROOT)}")
        return
    text = path.read_text(encoding="utf-8")
    for section in sections:
        if f"## {section}" not in text:
            errors.append(f"{path.relative_to(ROOT)}: missing section '## {section}'")


def visible_directories(parent: Path) -> list[Path]:
    if not parent.is_dir():
        return []
    return sorted(path for path in parent.iterdir() if path.is_dir() and not path.name.startswith((".", "_")))


def main() -> int:
    errors: list[str] = []
    index_path = ROOT / "agent.md"
    validate_document(index_path, ROOT_SECTIONS, errors)
    index_text = index_path.read_text(encoding="utf-8") if index_path.is_file() else ""

    modules = visible_directories(ROOT / "agents")
    experiments = visible_directories(ROOT / "research" / "experiments")
    for directory, sections in [(path, MODULE_SECTIONS) for path in modules] + [
        (path, EXPERIMENT_SECTIONS) for path in experiments
    ]:
        document = directory / "agent.md"
        validate_document(document, sections, errors)
        relative = directory.relative_to(ROOT).as_posix()
        if relative not in index_text:
            errors.append(f"agent.md index does not list: {relative}/")

    if errors:
        print("agent documentation check failed:")
        for error in errors:
            print(f"- {error}")
        return 1

    print(f"agent documentation check passed: {len(modules)} module(s), {len(experiments)} experiment(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
