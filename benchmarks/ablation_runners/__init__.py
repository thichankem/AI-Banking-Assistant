"""
Ablation study runners comparing Baseline vs Pipeline v1 vs Pipeline v2.
"""
from .run_ablation import run_ablation_benchmark, register_upgrade_pipeline

__all__ = ["run_ablation_benchmark", "register_upgrade_pipeline"]
