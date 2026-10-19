#!/usr/bin/env python3
"""
Ethos-U target catalog shared by install, skill registration, and compile.

U55 and U85 are both first-class NPUs. Adding another SoC is a new row in
``TARGETS``; conversion scripts read this table instead of hard-coding one
accelerator.
"""

from __future__ import annotations

from dataclasses import dataclass


SKILL_ID = "psoc-e84-ethosu-skill"
CANONICAL_SKILL = "tools/skills/psoc-e84-ethosu-skill"
# Both families must stay registered. install_ethosu_plugin.py refuses to
# write the plugin if either one disappears from TARGETS.
REQUIRED_NPUS: tuple[str, ...] = ("ethos-u55", "ethos-u85")


@dataclass(frozen=True)
class EthosUTarget:
    """Deployment target metadata used by conversion, Vela, and reporting."""

    key: str
    display_name: str
    npu: str
    accelerator_config: str
    system_config: str
    memory_mode: str
    vela_config: str
    use_legacy_core: bool
    fallback_system_config: str
    fallback_memory_mode: str
    fvp_env_var: str
    fvp_binaries: tuple[str, ...]
    socmem_mb: int
    psram_mb: int
    flash_mb: int
    arena_budget_kb: int
    notes: str


# NOTE:
# - PSoC Edge E84 values are based on publicly available product summaries.
# - Corstone-320 / Ethos-U85 values follow Arm's reference Vela profile
#   (Ethos_U85_SYS_DRAM_Mid + Dedicated_Sram_384KB). They are not a board
#   memory contract.
# - Keep the "notes" field explicit so unknown/estimated values are visible
#   in generated reports and not treated as verified silicon constraints.
TARGETS: dict[str, EthosUTarget] = {
    "psoc_e84": EthosUTarget(
        key="psoc_e84",
        display_name="Infineon PSoC Edge E84 (Cortex-M55 + Ethos-U55-128)",
        npu="ethos-u55",
        accelerator_config="ethos-u55-128",
        system_config="psoc_e84_syscfg",
        memory_mode="psoc_e84_balanced",
        vela_config="vela_psoc_e84.ini",
        use_legacy_core=True,
        fallback_system_config="Ethos_U55_High_End_Embedded",
        fallback_memory_mode="Shared_Sram",
        fvp_env_var="FVP_CORSTONE300_PATH",
        fvp_binaries=(
            "FVP_Corstone_SSE-300_Ethos-U55",
            "FVP_Corstone_SSE-300_Ethos-U55.exe",
        ),
        socmem_mb=5,
        psram_mb=32,
        flash_mb=128,
        arena_budget_kb=4096,
        notes=(
            "Memory split (5MB SoCMEM / 32MB PSRAM / 128MB Flash) should be "
            "validated against the final board support package and linker map. "
            "Custom system config still needs Vela's legacy core on current releases."
        ),
    ),
    "corstone300_u55_128_reference": EthosUTarget(
        key="corstone300_u55_128_reference",
        display_name="Arm Corstone-300 reference (Cortex-M55 + Ethos-U55-128)",
        npu="ethos-u55",
        accelerator_config="ethos-u55-128",
        system_config="corstone300_ref",
        memory_mode="corstone300_ref_sram",
        vela_config="vela_psoc_e84.ini",
        use_legacy_core=True,
        fallback_system_config="Ethos_U55_High_End_Embedded",
        fallback_memory_mode="Shared_Sram",
        fvp_env_var="FVP_CORSTONE300_PATH",
        fvp_binaries=(
            "FVP_Corstone_SSE-300_Ethos-U55",
            "FVP_Corstone_SSE-300_Ethos-U55.exe",
        ),
        socmem_mb=8,
        psram_mb=0,
        flash_mb=64,
        arena_budget_kb=6144,
        notes=(
            "Reference profile for simulation and regression. "
            "Capacity numbers are placeholders and should not be treated as "
            "a board-specific memory contract."
        ),
    ),
    "corstone320_u85_256_reference": EthosUTarget(
        key="corstone320_u85_256_reference",
        display_name="Arm Corstone-320 reference (Cortex-M85 + Ethos-U85-256)",
        npu="ethos-u85",
        accelerator_config="ethos-u85-256",
        system_config="corstone320_u85_syscfg",
        memory_mode="corstone320_u85_dedicated_sram",
        vela_config="vela_ethos_u85.ini",
        use_legacy_core=False,
        fallback_system_config="Ethos_U85_SYS_DRAM_Mid",
        fallback_memory_mode="Dedicated_Sram_384KB",
        fvp_env_var="FVP_CORSTONE320_PATH",
        fvp_binaries=(
            "FVP_Corstone_SSE-320",
            "FVP_Corstone_SSE-320.exe",
        ),
        socmem_mb=2,
        psram_mb=0,
        flash_mb=64,
        arena_budget_kb=8192,
        notes=(
            "Arm reference profile, not a production SoC contract. "
            "Ethos-U85 is compiled with Vela Regor (legacy core is U55-only). "
            "Dedicated SRAM cache is 384KB; the tensor arena is scheduled in "
            "DRAM. SRAM/DRAM sizes here are placeholders for budgeting reports."
        ),
    ),
}

DEFAULT_TARGET_KEY = "psoc_e84"


def list_targets() -> list[str]:
    return sorted(TARGETS.keys())


def npu_families() -> list[str]:
    """Sorted NPU family ids present in the catalog (``ethos-u55``, ``ethos-u85``)."""
    return sorted({target.npu for target in TARGETS.values()})


def require_npu_coverage() -> None:
    """Raise if the catalog no longer covers every required Ethos-U family."""
    present = set(npu_families())
    missing = [name for name in REQUIRED_NPUS if name not in present]
    if missing:
        raise RuntimeError(
            "Ethos-U target catalog is missing NPU families: "
            + ", ".join(missing)
        )


def get_target(target_key: str) -> EthosUTarget:
    normalized = (target_key or DEFAULT_TARGET_KEY).strip().lower()
    if normalized not in TARGETS:
        known = ", ".join(list_targets())
        raise KeyError(f"Unknown target {target_key!r}. Known targets: {known}")
    return TARGETS[normalized]


def format_human_memory(target: EthosUTarget) -> str:
    return (
        f"SoCMEM={target.socmem_mb}MB, PSRAM={target.psram_mb}MB, "
        f"Flash={target.flash_mb}MB, arena_budget={target.arena_budget_kb}KB"
    )


def target_record(target: EthosUTarget) -> dict[str, object]:
    """JSON-ready view of one target. This is what plugin install persists."""
    return {
        "key": target.key,
        "display_name": target.display_name,
        "npu": target.npu,
        "accelerator_config": target.accelerator_config,
        "system_config": target.system_config,
        "memory_mode": target.memory_mode,
        "vela_config": target.vela_config,
        "use_legacy_core": target.use_legacy_core,
        "fallback_system_config": target.fallback_system_config,
        "fallback_memory_mode": target.fallback_memory_mode,
        "fvp_env_var": target.fvp_env_var,
        "fvp_binaries": list(target.fvp_binaries),
        "socmem_mb": target.socmem_mb,
        "psram_mb": target.psram_mb,
        "flash_mb": target.flash_mb,
        "arena_budget_kb": target.arena_budget_kb,
        "notes": target.notes,
    }


def registry_document() -> dict[str, object]:
    """Skill-registration payload: one document for every Ethos-U target."""
    require_npu_coverage()
    targets = [target_record(TARGETS[key]) for key in list_targets()]
    return {
        "skill_id": SKILL_ID,
        "canonical_skill": CANONICAL_SKILL,
        "npu_families": list(REQUIRED_NPUS),
        "targets": targets,
    }
