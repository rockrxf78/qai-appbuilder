#!/usr/bin/env python3
"""
Install the Ethos-U skill plugin into the AppBuilder skills directory.

One path:

1. Read the Ethos-U55 / Ethos-U85 target catalog.
2. Register ``psoc-e84-ethosu-skill`` under ``tools/qaiappbuilder/skills/``.
3. Mark the plugin NPU-optimised and persist the same target table the
   Vela compile path reads.

Re-running is idempotent. ``--skills-dir`` overrides the destination for tests.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from _ethosu_targets import (
    CANONICAL_SKILL,
    REQUIRED_NPUS,
    SKILL_ID,
    npu_families,
    registry_document,
    require_npu_coverage,
)


PLUGIN_FILES = ("SKILL.md", "npu.txt", "ethosu_targets.json", "CANONICAL.txt")


def qaiappbuilder_skills_dir() -> Path:
    """``tools/qaiappbuilder/skills`` next to ``tools/skills``."""
    tools_dir = Path(__file__).resolve().parents[3]
    return tools_dir / "qaiappbuilder" / "skills"


def canonical_skill_dir() -> Path:
    return Path(__file__).resolve().parents[1]


def _strip_frontmatter(text: str) -> str:
    if not text.startswith("---"):
        return text
    end = text.find("\n---", 3)
    if end < 0:
        return text
    rest = text[end + 4 :]
    if rest.startswith("\n"):
        rest = rest[1:]
    return rest


def render_plugin_skill() -> str:
    """Plugin SKILL.md: registration header plus the canonical workflow body."""
    require_npu_coverage()
    document = registry_document()
    rows = [
        "| Target | NPU | Accelerator | Vela config | Legacy core | FVP env |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for target in document["targets"]:
        assert isinstance(target, dict)
        legacy = "yes" if target["use_legacy_core"] else "no (Regor)"
        rows.append(
            "| `{key}` | `{npu}` | `{accelerator}` | `{vela}` | {legacy} | `{fvp}` |".format(
                key=target["key"],
                npu=target["npu"],
                accelerator=target["accelerator_config"],
                vela=target["vela_config"],
                legacy=legacy,
                fvp=target["fvp_env_var"],
            )
        )
    table = "\n".join(rows)
    families = ", ".join(f"`{name}`" for name in REQUIRED_NPUS)
    canonical_body = _strip_frontmatter(
        (canonical_skill_dir() / "SKILL.md").read_text(encoding="utf-8")
    ).rstrip()
    # Concatenate the canonical body so braces in SKILL.md are not formatted.
    header = (
        "---\n"
        "name: aipc-ethosu-toolkit\n"
        "description: ARM Ethos-U NPU workflow for Ethos-U55 and Ethos-U85. "
        "Installs this skill plugin, registers both NPUs, and compiles TFLite "
        "int8 with Vela for PSoC Edge E84 (U55-128) or Corstone-320 (U85-256).\n"
        "use_for: Deploying models onto an Arm Ethos-U55 or Ethos-U85 NPU, "
        "including PSoC E84, Vela compile, TFLite Micro, and Corstone FVP.\n"
        "tags: ethos-u, ethos-u55, ethos-u85, npu, vela, tflite-micro, psoc-e84\n"
        "---\n"
        "\n"
        "# AIPC Ethos-U Toolkit (Ethos-U55 and Ethos-U85)\n"
        "\n"
        "This file is the AppBuilder skill plugin. "
        "`scripts/install_ethosu_plugin.py` generates it from "
        f"`{CANONICAL_SKILL}` so install, registration, and NPU target "
        "selection stay on one path.\n"
        "\n"
        f"Registered NPU families: {families}.\n"
        "\n"
        "## Registered Ethos-U NPU targets\n"
        "\n"
        f"{table}\n"
        "\n"
        "Select a target with `--target <key>` on `run_pipeline.py`, "
        "`compile_with_vela.py`, `ethosu_preflight_check.py`, and "
        "`run_fvp_harness.py`. U55 targets default to Vela's legacy core. "
        "U85 targets use Regor and do not pass `--debug-force-legacy-core`.\n"
        "\n"
        "## Canonical workflow\n"
        "\n"
        + canonical_body
        + "\n"
    )
    return header


def render_npu_marker() -> str:
    require_npu_coverage()
    return "".join(f"{name}\n" for name in npu_families())


def render_targets_json() -> str:
    return json.dumps(registry_document(), indent=2, ensure_ascii=False) + "\n"


def plugin_file_contents() -> dict[str, str]:
    return {
        "SKILL.md": render_plugin_skill(),
        "npu.txt": render_npu_marker(),
        "ethosu_targets.json": render_targets_json(),
        "CANONICAL.txt": CANONICAL_SKILL + "\n",
    }


def install_plugin(skills_dir: Path | None = None) -> Path:
    """Write the skill plugin directory. Returns the skill directory path."""
    require_npu_coverage()
    root = skills_dir if skills_dir is not None else qaiappbuilder_skills_dir()
    dest = root / SKILL_ID
    dest.mkdir(parents=True, exist_ok=True)
    for name, contents in plugin_file_contents().items():
        (dest / name).write_text(contents, encoding="utf-8")
    return dest


def plugin_registration_status(skills_dir: Path | None = None) -> tuple[bool, str]:
    """Return whether the installed plugin advertises both Ethos-U families."""
    root = skills_dir if skills_dir is not None else qaiappbuilder_skills_dir()
    dest = root / SKILL_ID
    skill_md = dest / "SKILL.md"
    npu_marker = dest / "npu.txt"
    targets_path = dest / "ethosu_targets.json"
    if not skill_md.is_file() or not npu_marker.is_file() or not targets_path.is_file():
        return (
            False,
            f"Ethos-U skill plugin is not registered at {dest}",
        )
    try:
        payload = json.loads(targets_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return False, f"ethosu_targets.json is unreadable: {exc}"
    families = payload.get("npu_families")
    if not isinstance(families, list):
        return False, "ethosu_targets.json is missing npu_families"
    missing = [name for name in REQUIRED_NPUS if name not in families]
    if missing:
        return False, "registered plugin is missing NPU families: " + ", ".join(missing)
    marker = {
        line.strip()
        for line in npu_marker.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    if not set(REQUIRED_NPUS).issubset(marker):
        return False, "npu.txt does not list both ethos-u55 and ethos-u85"
    return True, f"registered at {dest} ({', '.join(REQUIRED_NPUS)})"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Register the Ethos-U55/U85 skill plugin with AppBuilder"
    )
    parser.add_argument(
        "--skills-dir",
        default="",
        help="Destination skills directory (default: tools/qaiappbuilder/skills)",
    )
    args = parser.parse_args()
    skills_dir = Path(args.skills_dir).resolve() if args.skills_dir else None
    try:
        dest = install_plugin(skills_dir)
    except (OSError, RuntimeError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1
    ok, detail = plugin_registration_status(dest.parent)
    print("=" * 72)
    print("Ethos-U skill plugin install")
    print("=" * 72)
    print(f"skill_id: {SKILL_ID}")
    print(f"path: {dest}")
    print(f"npu_families: {', '.join(REQUIRED_NPUS)}")
    print(f"registration: {'PASS' if ok else 'FAIL'} {detail}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
