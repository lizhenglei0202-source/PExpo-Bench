"""Unified tool registry — all callable Python implementations.

The OpenAI function-calling schemas (TOOL_DEFS) live in
pexpo_bench/architectures/orchestrator.py; this module just exposes the
underlying Python callables keyed by tool name.
"""
from pexpo_bench.tools.tools import TOOL_REGISTRY as _BASE_TOOLS
from pexpo_bench.tools.health_tools import HEALTH_TOOLS
from pexpo_bench.tools.meta_tools import META_TOOLS

# Merge: base (python_sandbox, dose_calculator, etc.) + new health + meta
TOOL_REGISTRY = {**_BASE_TOOLS, **HEALTH_TOOLS, **META_TOOLS}

__all__ = ["TOOL_REGISTRY"]
