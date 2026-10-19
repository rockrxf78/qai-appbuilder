# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Install → skill registration → Ethos-U55/U85 target support."""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SKILL_SCRIPTS = REPO_ROOT / "tools" / "skills" / "psoc-e84-ethosu-skill" / "scripts"
SKILL_CONFIGS = REPO_ROOT / "tools" / "skills" / "psoc-e84-ethosu-skill" / "configs"
QAI_SRC = REPO_ROOT / "tools" / "qaiappbuilder" / "src"
COMMITTED_PLUGIN = (
    REPO_ROOT / "tools" / "qaiappbuilder" / "skills" / "psoc-e84-ethosu-skill"
)

sys.path.insert(0, str(SKILL_SCRIPTS))
sys.path.insert(0, str(QAI_SRC))

targets = importlib.import_module("_ethosu_targets")
installer = importlib.import_module("install_ethosu_plugin")
compile_with_vela = importlib.import_module("compile_with_vela")
discovery = importlib.import_module("qai.platform.skills.discovery")


def test_catalog_covers_u55_and_u85() -> None:
    families = targets.npu_families()
    assert families == ["ethos-u55", "ethos-u85"]
    targets.require_npu_coverage()

    u55 = targets.get_target("psoc_e84")
    assert u55.npu == "ethos-u55"
    assert u55.accelerator_config == "ethos-u55-128"
    assert u55.use_legacy_core is True
    assert u55.vela_config == "vela_psoc_e84.ini"
    assert u55.fallback_system_config == "Ethos_U55_High_End_Embedded"

    u85 = targets.get_target("corstone320_u85_256_reference")
    assert u85.npu == "ethos-u85"
    assert u85.accelerator_config == "ethos-u85-256"
    assert u85.use_legacy_core is False
    assert u85.vela_config == "vela_ethos_u85.ini"
    assert u85.fallback_system_config == "Ethos_U85_SYS_DRAM_Mid"
    assert u85.fallback_memory_mode == "Dedicated_Sram_384KB"
    assert u85.fvp_env_var == "FVP_CORSTONE320_PATH"

    for target in targets.TARGETS.values():
        assert (SKILL_CONFIGS / target.vela_config).is_file()


def test_compile_flags_follow_the_selected_npu() -> None:
    u55 = targets.get_target("psoc_e84")
    u85 = targets.get_target("corstone320_u85_256_reference")

    assert compile_with_vela.resolve_legacy_core(u55, None) is True
    assert compile_with_vela.resolve_legacy_core(u85, None) is False
    assert compile_with_vela.resolve_legacy_core(u85, True) is True

    u55_ini = compile_with_vela.resolve_vela_config(u55, "")
    u85_ini = compile_with_vela.resolve_vela_config(u85, "")
    assert u55_ini.name == "vela_psoc_e84.ini"
    assert u85_ini.name == "vela_ethos_u85.ini"
    assert u55_ini.is_file() and u85_ini.is_file()

    assert compile_with_vela.resolve_fallback(u85, "", "") == (
        "Ethos_U85_SYS_DRAM_Mid",
        "Dedicated_Sram_384KB",
    )


def test_install_registers_npu_plugin(tmp_path: Path) -> None:
    dest = installer.install_plugin(tmp_path)
    ok, detail = installer.plugin_registration_status(tmp_path)
    assert ok, detail
    assert dest.name == "psoc-e84-ethosu-skill"

    payload = json.loads((dest / "ethosu_targets.json").read_text(encoding="utf-8"))
    assert payload["npu_families"] == ["ethos-u55", "ethos-u85"]
    accelerators = {item["accelerator_config"] for item in payload["targets"]}
    assert "ethos-u55-128" in accelerators
    assert "ethos-u85-256" in accelerators

    found = discovery.SkillDiscovery(tmp_path).scan()
    assert len(found) == 1
    skill = found[0]
    assert skill.skill_id == "psoc-e84-ethosu-skill"
    assert skill.npu_optimized is True
    assert "ethos-u85" in skill.tags
    assert discovery.resolve_skill_mode({"mode": "local"}, skill.npu_optimized) == "local"
    assert discovery.resolve_skill_mode({"mode": "both"}, False) == "cloud"


def test_committed_plugin_matches_installer(tmp_path: Path) -> None:
    fresh = installer.install_plugin(tmp_path)
    assert COMMITTED_PLUGIN.is_dir(), "run install_ethosu_plugin.py to register the skill"
    for name in installer.PLUGIN_FILES:
        assert (COMMITTED_PLUGIN / name).read_text(encoding="utf-8") == (
            fresh / name
        ).read_text(encoding="utf-8")
