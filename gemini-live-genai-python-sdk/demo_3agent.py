#!/usr/bin/env python3
"""
3-Agent Orchestration Demo
===========================
Three Ollama agents work in sequence, each passing its output to the next.

  Agent 1 (analyst):  Inspects a Python file and writes a structured analysis.
  Agent 2 (coder):    Reads the analysis and writes a test file for it.
  Agent 3 (reviewer): Reads the source + tests, writes a quality review.

Each agent runs via run_ollama_agent() with native Ollama tool calling.
"""

import sys
import json
import time
from pathlib import Path

# Ensure we can import from the project
sys.path.insert(0, str(Path(__file__).parent))

from tools import run_ollama_agent


# ── Config ────────────────────────────────────────────────────────────────────

MODEL = "gemma4:e4b-64K"
OPTIONS = {"temperature": 0, "top_p": 0.85, "repeat_penalty": 1.15}
TOOLS = ["read_file", "write_file", "edit_file", "bash"]
TIMEOUT = 120

# The file to analyze — use task_manager.py since it's always present
TARGET_FILE = str(Path(__file__).parent / "task_manager.py")
ANALYSIS_FILE = "/tmp/agent_demo_analysis.md"
TEST_FILE = "/tmp/test_task_manager.py"
REVIEW_FILE = "/tmp/agent_demo_review.md"


def run_agent(name: str, system_prompt: str, user_prompt: str) -> str:
    """Run one agent and print its output."""
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    start = time.time()

    result = run_ollama_agent(
        model=MODEL,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        tool_names=TOOLS,
        options=OPTIONS,
        timeout=TIMEOUT,
    )

    elapsed = time.time() - start
    print(f"\n{result}")
    print(f"\n  ({elapsed:.1f}s)")
    return result


# ── Agent 1: Analyst ──────────────────────────────────────────────────────────

analysis = run_agent(
    name="Agent 1 — Analyst",
    system_prompt=(
        "You are a code analyst. You read source files and produce structured "
        "analysis documents. You MUST use the write_file tool to save your "
        "analysis to a file. Do not just describe what you would write — "
        "actually call write_file with the full content."
    ),
    user_prompt=(
        f"Step 1: Use read_file to read {TARGET_FILE}\n"
        f"Step 2: Use write_file to write a structured markdown analysis to "
        f"{ANALYSIS_FILE}\n"
        "Include: purpose, key classes/functions, public API, edge cases, "
        "and suggested improvements."
    ),
)


# ── Agent 2: Coder ───────────────────────────────────────────────────────────

tests = run_agent(
    name="Agent 2 — Coder",
    system_prompt=(
        "You are a test engineer. You read analysis documents and source code, "
        "then write comprehensive pytest test files. You MUST use the write_file "
        "tool to save the test file. Do not just describe tests — write them "
        "using write_file with complete, runnable test code."
    ),
    user_prompt=(
        f"Step 1: Use read_file to read the analysis at {ANALYSIS_FILE}\n"
        f"Step 2: Use read_file to read the source at {TARGET_FILE}\n"
        f"Step 3: Use write_file to write a pytest test file to {TEST_FILE}\n"
        "Cover the public API, edge cases, and error handling. "
        "Include a module docstring explaining coverage."
    ),
)


# ── Agent 3: Reviewer ────────────────────────────────────────────────────────

review = run_agent(
    name="Agent 3 — Reviewer",
    system_prompt=(
        "You are a senior code reviewer. You compare source code against its "
        "tests and produce a quality review. You MUST use the write_file tool "
        "to save your review. Do not just output text — call write_file."
    ),
    user_prompt=(
        f"Step 1: Use read_file to read the source at {TARGET_FILE}\n"
        f"Step 2: Use read_file to read the test file at {TEST_FILE}\n"
        f"Step 3: Use write_file to write a review to {REVIEW_FILE}\n"
        "Cover: test coverage gaps, correctness issues, missing edge cases, "
        "and a coverage score (0-100%). Reference specific function names."
    ),
)


# ── Summary ───────────────────────────────────────────────────────────────────

print(f"\n{'='*60}")
print("  Pipeline Complete")
print(f"{'='*60}")
print()

for label, path in [("Analysis", ANALYSIS_FILE), ("Tests", TEST_FILE), ("Review", REVIEW_FILE)]:
    p = Path(path)
    if p.exists():
        lines = p.read_text().splitlines()
        print(f"  {label}: {len(lines)} lines, {len(p.read_text())} chars")
        print(f"    -> {path}")
    else:
        print(f"  {label}: (not created)")
