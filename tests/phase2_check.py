#!/usr/bin/env python3
"""
Phase 2 preflight (hackathon: Agentic Evaluation).

Mirrors what organizers score in Phase 2:
  - Baseline / inference re-run (deterministic rule-based when HF_TOKEN unset)
  - Reward / aggregate score variance (not constant, in valid range)
  - Same paths as hosted eval: optional Docker image `modguard-rl` + RUN_MODE=eval

Usage (from repo root):
  uv run python tests/phase2_check.py
  PHASE2_SKIP_DOCKER=1 uv run python tests/phase2_check.py   # skip docker build/run
  PHASE2_QUICK=1 uv run python tests/phase2_check.py         # fewer inference episodes

Exit 0 only if every step passes.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _run(
    cmd: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    timeout: float | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=str(cwd or PROJECT_ROOT),
        env={**os.environ, **(env or {})},
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )


def _print_step(name: str) -> None:
    print(f"\n{'=' * 60}\n▶ {name}\n{'=' * 60}")


def _fail(name: str, proc: subprocess.CompletedProcess[str]) -> int:
    print(f"FAILED: {name} (exit {proc.returncode})")
    if proc.stdout:
        print("--- stdout ---\n", proc.stdout[-8000:])
    if proc.stderr:
        print("--- stderr ---\n", proc.stderr[-8000:])
    return 1


def _last_json_line(stderr: str) -> dict[str, object] | None:
    for line in reversed(stderr.strip().splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    return None


def step_judge() -> int:
    _print_step("tests/judge.py (rollouts + README + Docker + inference smoke)")
    proc = _run([sys.executable, str(PROJECT_ROOT / "tests" / "judge.py")], timeout=300)
    if proc.returncode != 0:
        return _fail("judge", proc)
    print(proc.stdout)
    if proc.stderr:
        print(proc.stderr)
    return 0


def step_stress() -> int:
    _print_step("tests/stress_check.py (reward diversity + edge coverage)")
    proc = _run([sys.executable, str(PROJECT_ROOT / "tests" / "stress_check.py")], timeout=300)
    if proc.returncode != 0:
        return _fail("stress_check", proc)
    print(proc.stdout)
    return 0


def step_openenv_validate() -> int:
    _print_step("openenv validate . (deployment readiness)")
    exe = shutil.which("openenv")
    if not exe:
        print("SKIP: openenv CLI not on PATH (install openenv-core / openenv package)")
        return 0
    proc = _run([exe, "validate", str(PROJECT_ROOT)], timeout=120)
    if proc.returncode != 0:
        return _fail("openenv validate", proc)
    print(proc.stdout or proc.stderr or "(no output)")
    return 0


def step_inference_phase2() -> int:
    _print_step("inference.py — full Phase-2-style baseline (RUN_MODE=eval, in-process)")
    quick = os.getenv("PHASE2_QUICK", "").strip().lower() in {"1", "true", "yes"}
    episodes = "2" if quick else os.getenv("NUM_EPISODES_PER_TASK", "8")
    proc = _run(
        [sys.executable, str(PROJECT_ROOT / "inference.py")],
        env={
            "RUN_MODE": "eval",
            "FORCE_INPROCESS": "1",
            "HF_TOKEN": "",
            "NUM_EPISODES_PER_TASK": episodes,
        },
        timeout=600,
    )
    if proc.returncode != 0:
        return _fail("inference", proc)
    summary = _last_json_line(proc.stderr)
    if summary is None:
        print(proc.stderr[-4000:])
        return _fail("inference (no JSON summary on stderr)", proc)
    overall = summary.get("overall") if isinstance(summary, dict) else None
    if not isinstance(overall, dict):
        return _fail("inference (invalid overall)", proc)
    agg = overall.get("aggregate")
    episodes_n = overall.get("episodes")
    print(f"  aggregate={agg}  episodes={episodes_n}")
    if agg is None:
        return _fail("inference (missing aggregate)", proc)
    try:
        agg_f = float(agg)
    except (TypeError, ValueError):
        return _fail("inference (aggregate not numeric)", proc)
    if not (0.0 <= agg_f <= 1.0):
        return _fail("inference (aggregate out of [0,1])", proc)
    # Phase 2: score variance — not a flat line across tasks
    tm = summary.get("task_metrics") if isinstance(summary, dict) else None
    if isinstance(tm, dict) and len(tm) >= 2:
        grades = []
        for _k, v in tm.items():
            if isinstance(v, dict) and "grader_score" in v:
                try:
                    grades.append(float(v["grader_score"]))
                except (TypeError, ValueError):
                    continue
        if len(grades) >= 2 and max(grades) - min(grades) < 1e-6:
            return _fail("inference (task grader scores are identical — variance check)", proc)
    print("  OK: inference completed with structured summary")
    return 0


def step_docker_eval() -> int:
    if os.getenv("PHASE2_SKIP_DOCKER", "").strip().lower() in {"1", "true", "yes"}:
        print("\nSKIP: PHASE2_SKIP_DOCKER=1")
        return 0
    if not shutil.which("docker"):
        print("\nSKIP: docker not on PATH")
        return 0
    _print_step("docker build -t modguard-rl . (hosted image name)")
    b = _run(
        ["docker", "build", "-t", "modguard-rl", "."],
        cwd=PROJECT_ROOT,
        timeout=600,
    )
    if b.returncode != 0:
        return _fail("docker build", b)
    print(b.stdout[-2000:] if b.stdout else "(build ok)")

    _print_step("docker run — same as HF entrypoint eval (RUN_MODE=eval)")
    episodes = "2" if os.getenv("PHASE2_QUICK", "").strip().lower() in {"1", "true", "yes"} else "8"
    r = _run(
        [
            "docker",
            "run",
            "--rm",
            "-e",
            "RUN_MODE=eval",
            "-e",
            "HF_TOKEN=",
            "-e",
            f"NUM_EPISODES_PER_TASK={episodes}",
            "modguard-rl",
        ],
        cwd=PROJECT_ROOT,
        timeout=600,
    )
    if r.returncode != 0:
        return _fail("docker run eval", r)
    summary = _last_json_line(r.stderr)
    if summary is None:
        print(r.stderr[-4000:])
        return _fail("docker eval (no JSON summary)", r)
    print("  docker eval OK; aggregate:", (summary.get("overall") or {}).get("aggregate"))
    return 0


def main() -> int:
    os.chdir(PROJECT_ROOT)
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    steps = [
        ("judge", step_judge),
        ("stress", step_stress),
        ("openenv_validate", step_openenv_validate),
        ("inference", step_inference_phase2),
        ("docker", step_docker_eval),
    ]
    for name, fn in steps:
        code = fn()
        if code != 0:
            print(f"\n✗ Phase2 preflight stopped at: {name}")
            return code

    print("\n" + "=" * 60)
    print("✓ Phase 2 preflight passed — safe to git push / HF Space deploy")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
