#!/usr/bin/env python3
"""
Ethos-U target catalog shared by all PSoC E84 skill scripts.

The table is intentionally small and explicit so that new Ethos-U SoCs can
be added without changing conversion/verification logic in every script.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EthosUTarget:
    """Deployment target metadata used by conversion, Vela, and reporting."""

    key: str
    display_name: str
    accelerator_config: str
    system_config: str
    memory_mode: str
    socmem_mb: int
    psram_mb: int
    flash_mb: int
    arena_budget_kb: int
    notes: str


# NOTE:
# - PSoC Edge E84 values are based on publicly available product summaries.
# - Keep the "notes" field explicit so unknown/estimated values are visible
#   in generated reports and not treated as verified silicon constraints.
TARGETS: dict[str, EthosUTarget] = {
    "psoc_e84": EthosUTarget(
        key="psoc_e84",
        display_name="Infineon PSoC Edge E84 (Cortex-M55 + Ethos-U55-128)",
        accelerator_config="ethos-u55-128",
        system_config="psoc_e84_syscfg",
        memory_mode="psoc_e84_balanced",
        socmem_mb=5,
        psram_mb=32,
        flash_mb=128,
        arena_budget_kb=4096,
        notes=(
            "Memory split (5MB SoCMEM / 32MB PSRAM / 128MB Flash) should be "
            "validated against the final board support package and linker map."
        ),
    ),
    "corstone300_u55_128_reference": EthosUTarget(
        key="corstone300_u55_128_reference",
        display_name="Arm Corstone-300 reference (Cortex-M55 + Ethos-U55-128)",
        accelerator_config="ethos-u55-128",
        system_config="corstone300_ref",
        memory_mode="corstone300_ref_sram",
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
}

DEFAULT_TARGET_KEY = "psoc_e84"


def list_targets() -> list[str]:
    return sorted(TARGETS.keys())


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

