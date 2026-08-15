#!/usr/bin/env python3
"""Run IncidentLens/baseline experiments by orchestrating pipeline + emitter processes.

Synthetic-data machine:

    python evaluation/run_experiments.py --exp-type synth --venv venv

Real-data machine:

    python evaluation/run_experiments.py --exp-type real --venv venv

The script starts detection.full_pipeline first, waits until its socket is ready,
then starts the selected emitter.  When the emitter finishes a dataset/chunk,
the pipeline is terminated and the next experiment starts.

Progress/ETA:
  * Shows an overall experiment-level progress bar when tqdm is installed.
  * Prints elapsed time for each experiment and an ETA based on completed
    experiment durations.
  * Writes machine-readable progress to evaluation/experiment_logs/progress.json.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import signal
import socket
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional, Sequence

try:
    from tqdm.auto import tqdm  # type: ignore
except Exception:  # pragma: no cover
    tqdm = None  # type: ignore


DEFAULT_BASELINES = [
    "incidentlens",
    "direct_observation",
    "space_time_clustering",
    "text_only_clustering",
    "late_fusion_voting",
    "hotspot_scan",
    "generic_propagation",
    "generic_all",
]

BACKGROUND_AWARE_BASELINES = [
    "satscan_background",
    "hawkes_event_detector",
]

BACKGROUND_AWARE_BASELINE_SET = set(BACKGROUND_AWARE_BASELINES)
BACKGROUND_AWARE_SYNTHETIC_BASELINES = BACKGROUND_AWARE_BASELINES
BACKGROUND_AWARE_SYNTHETIC_BASELINE_SET = BACKGROUND_AWARE_BASELINE_SET
DEFAULT_REAL_BACKGROUND_BASELINES = BACKGROUND_AWARE_BASELINES

# Synthetic runs include the observation-stream background-aware baselines by
# default.  Real-data defaults intentionally stay closer to the older real suite:
# generic_propagation/generic_all are synthetic architectural ablations and can be
# re-enabled explicitly with --baselines and --real-exclude-baselines "".
DEFAULT_SYNTHETIC_BASELINES = DEFAULT_BASELINES + BACKGROUND_AWARE_SYNTHETIC_BASELINES
DEFAULT_REAL_BASELINES = [
    "incidentlens",
    "direct_observation",
    "space_time_clustering",
    "text_only_clustering",
    "late_fusion_voting",
    "hotspot_scan",
]

DEFAULT_REAL_DATES = [
    "20250106", "20250107", "20250108", "20250109", "20250110", "20250111",
    "20250112", "20250113", "20250114", "20250115", "20250116", "20250117",
    "20250118", "20250119", "20250120", "20250121", "20250122", "20250123",
    "20250124", "20250125", "20250126", "20250127", "20250128", "20250129",
    "20250130", "20250131", "20250201", "20250202", "20250328", "20250329",
    "20250404", "20250405", "20250430", "20250501", "20250605", "20250606",
    "20250607", "20250608", "20250609", "20250610", "20250611", "20250612",
    "20250630", "20250701", "20250702", "20250703", "20250704", "20250706",
    "20250707", "20250708", "20250710", "20250711", "20250712", "20250811",
    "20250812", "20260129", "20260130", "20260131", "20260212", "20260213",
    "20260214",
]

DEFAULT_REAL_NON_DATES = [
    "20250221", "20250222", "20250223", "20250224", "20250225", "20250226",
    "20250227", "20250228", "20250301", "20250302", "20250303", "20250304",
    "20250305", "20250306",
]


@dataclass(frozen=True)
class Experiment:
    name: str
    baseline: str
    incident_types: str
    results_folder: str
    emitter_module: str
    emitter_args: list[str] = field(default_factory=list)
    pipeline_args: list[str] = field(default_factory=list)
    dates: tuple[str, ...] = ()


@dataclass
class ExperimentRunResult:
    name: str
    baseline: str
    results_folder: str
    dates: tuple[str, ...]
    status: str
    elapsed_seconds: float
    started_at: str
    finished_at: str
    error: Optional[str] = None


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def format_duration(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h {minutes}m {secs}s"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def _tee_stream(
    stream,
    prefix: str,
    log_path: Path,
    *,
    quiet: bool = False,
    readiness_event: Optional[threading.Event] = None,
    readiness_markers: Sequence[str] = (),
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8", errors="replace") as log_file:
        for line in iter(stream.readline, ""):
            text = line.rstrip("\n")
            if readiness_event is not None and any(marker in text for marker in readiness_markers):
                readiness_event.set()
            if not quiet:
                print(f"[{prefix}] {text}", flush=True)
            log_file.write(text + "\n")
            log_file.flush()


def _start_process(
    cmd: list[str],
    *,
    prefix: str,
    log_dir: Path,
    env: dict[str, str],
    quiet_child_logs: bool,
    readiness_event: Optional[threading.Event] = None,
    readiness_markers: Sequence[str] = (),
) -> subprocess.Popen:
    print("\n$ " + " ".join(cmd), flush=True)
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        env=env,
        preexec_fn=os.setsid if os.name != "nt" else None,
    )
    assert proc.stdout is not None
    assert proc.stderr is not None
    threading.Thread(
        target=_tee_stream,
        args=(proc.stdout, f"{prefix}:out", log_dir / f"{prefix}.stdout.log"),
        kwargs={
            "quiet": quiet_child_logs,
            "readiness_event": readiness_event,
            "readiness_markers": readiness_markers,
        },
        daemon=True,
    ).start()
    threading.Thread(
        target=_tee_stream,
        args=(proc.stderr, f"{prefix}:err", log_dir / f"{prefix}.stderr.log"),
        kwargs={
            "quiet": quiet_child_logs,
            "readiness_event": readiness_event,
            "readiness_markers": readiness_markers,
        },
        daemon=True,
    ).start()
    return proc


def _wait_for_pipeline_ready(
    proc: subprocess.Popen,
    readiness_event: threading.Event,
    *,
    timeout_s: float,
    host: str,
    port: int,
) -> None:
    """Wait until full_pipeline logs that it is listening.

    Do not probe the TCP port with socket.create_connection here.  This pipeline
    treats an accepted connection as an actual parser/emitter session; a readiness
    probe can therefore consume the only session, make the pipeline process the
    empty probe as 0 reports, and leave the real emitter with ConnectionRefused.
    """
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if readiness_event.wait(timeout=0.2):
            return
        rc = proc.poll()
        if rc is not None:
            raise RuntimeError(f"Pipeline exited before listening on {host}:{port}; exit code={rc}")
    raise TimeoutError(
        f"Timed out waiting for full_pipeline to log readiness on {host}:{port}. "
        "Check the pipeline stderr log for startup errors."
    )


def _terminate_process(proc: subprocess.Popen, *, name: str, grace_s: float = 10.0) -> None:
    if proc.poll() is not None:
        return
    print(f"Stopping {name} ...", flush=True)
    try:
        if os.name != "nt":
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        else:
            proc.terminate()
        proc.wait(timeout=grace_s)
    except subprocess.TimeoutExpired:
        print(f"{name} did not stop within {grace_s}s; killing.", flush=True)
        if os.name != "nt":
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        else:
            proc.kill()
        proc.wait(timeout=grace_s)
    except ProcessLookupError:
        pass


def _tail_file(path: Path, *, max_lines: int = 80, max_chars: int = 12000) -> str:
    """Return a compact tail of a log file for failure diagnostics."""
    try:
        if not path.exists():
            return ""
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()[-int(max_lines):]
        text = "\n".join(lines)
        if len(text) > int(max_chars):
            text = text[-int(max_chars):]
        return text
    except Exception as exc:
        return f"<could not read {path}: {exc}>"


def resolve_python_executable(*, explicit_python: str, venv: Optional[str]) -> tuple[str, dict[str, str]]:
    env_updates: dict[str, str] = {}
    if not venv:
        return explicit_python, env_updates

    venv_path = Path(venv).expanduser().resolve()
    if os.name == "nt":
        python_path = venv_path / "Scripts" / "python.exe"
        bin_dir = venv_path / "Scripts"
    else:
        python_path = venv_path / "bin" / "python"
        bin_dir = venv_path / "bin"

    if not python_path.exists():
        raise FileNotFoundError(f"Could not find venv Python at {python_path}")

    env_updates["VIRTUAL_ENV"] = str(venv_path)
    env_updates["PATH"] = str(bin_dir) + os.pathsep + os.environ.get("PATH", "")
    return str(python_path), env_updates


def _pipeline_cmd(exp: Experiment, python_exe: str, *, skip_completed: bool) -> list[str]:
    cmd = [
        python_exe,
        "-u",
        "-m",
        "detection.full_pipeline",
        "--incident-types",
        exp.incident_types,
        "--baseline",
        exp.baseline,
        "--results-folder",
        exp.results_folder,
    ]
    cmd.append("--skip-completed" if skip_completed else "--no-skip-completed")
    cmd.extend(exp.pipeline_args)
    return cmd


def _emitter_cmd(exp: Experiment, python_exe: str) -> list[str]:
    return [python_exe, "-u", "-m", exp.emitter_module, *exp.emitter_args]


def _read_dates_file(path: str | Path) -> list[str]:
    out: list[str] = []
    with Path(path).open("r", encoding="utf-8") as infile:
        for line in infile:
            text = line.strip()
            if not text or text.startswith("#"):
                continue
            for part in text.replace(",", " ").split():
                if part:
                    out.append(part)
    return out


def _normalize_dates(values: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        digits = "".join(ch for ch in str(value) if ch.isdigit())
        if len(digits) != 8:
            raise ValueError(f"Expected YYYYMMDD date, got {value!r}")
        if digits not in seen:
            seen.add(digits)
            out.append(digits)
    return sorted(out)


def contiguous_date_chunks(dates: Sequence[str]) -> list[list[str]]:
    """Split sorted YYYYMMDD dates into maximal day-contiguous chunks."""
    normalized = _normalize_dates(dates)
    if not normalized:
        return []
    chunks: list[list[str]] = [[normalized[0]]]
    prev = datetime.strptime(normalized[0], "%Y%m%d").date()
    for date_str in normalized[1:]:
        cur = datetime.strptime(date_str, "%Y%m%d").date()
        if cur == prev + timedelta(days=1):
            chunks[-1].append(date_str)
        else:
            chunks.append([date_str])
        prev = cur
    return chunks


def _valid_nonempty_json_file(path: Path) -> bool:
    """Return True only for present, non-empty, parseable JSON result files."""
    if not path.exists() or not path.is_file():
        return False
    try:
        if path.stat().st_size <= 0:
            return False
        with path.open("r", encoding="utf-8") as infile:
            json.load(infile)
        return True
    except Exception:
        return False


def _completion_marker_failed(path: Path) -> bool:
    """Return True only when an optional completion marker explicitly says failure.

    The real source of truth for resume/skip is the three replay artifacts
    written by full_pipeline: low_level_results.json, high_level_results.json,
    and timing.json.  Some older successful runs do not have a reliable
    experiment_complete.json marker, so the marker must be optional.  However,
    if a parseable marker explicitly says failed/incomplete/running, do not skip.
    """
    if not path.exists() or not path.is_file():
        return False
    try:
        if path.stat().st_size <= 0:
            return False
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        # Ignore malformed optional markers.  Required result artifacts decide.
        return False

    if isinstance(data, dict):
        status = str(data.get("status") or data.get("state") or "").strip().lower()
        if status in {"failed", "failure", "error", "incomplete", "running", "started"}:
            return True
        for key in ("completed", "complete", "success", "ok"):
            if key in data and not bool(data.get(key)):
                return True
    return False


def result_date_complete(baseline: str, results_folder: str, date_str: str) -> bool:
    """Return True when a real-data date has completed replay result artifacts.

    Results are stored at:
        evaluation/results/<baseline>/<results_folder>/<YYYYMMDD>/

    A date is complete when the three replay artifacts are present, non-empty,
    and parseable JSON.  experiment_complete.json is optional because older
    successful runs may not have it; it is used only to reject explicit failure
    markers.  A date folder by itself is never sufficient.
    """
    result_dir = Path("evaluation/results") / baseline / results_folder / date_str
    required = ["low_level_results.json", "high_level_results.json", "timing.json"]
    if not all(_valid_nonempty_json_file(result_dir / name) for name in required):
        return False
    return not _completion_marker_failed(result_dir / "experiment_complete.json")


def incomplete_real_dates(exp: Experiment) -> list[str]:
    if not exp.dates:
        return []
    return [date_str for date_str in exp.dates if not result_date_complete(exp.baseline, exp.results_folder, date_str)]


def real_chunk_complete(exp: Experiment) -> bool:
    if not exp.dates:
        return False
    return not incomplete_real_dates(exp)


def _emitter_args_with_dates(emitter_args: Sequence[str], dates: Sequence[str]) -> list[str]:
    """Replace the --dates segment in an existing real-emitter command."""
    args = list(emitter_args)
    if "--dates" not in args:
        return args
    start = args.index("--dates")
    end = start + 1
    while end < len(args) and not str(args[end]).startswith("--"):
        end += 1
    return args[: start + 1] + list(dates) + args[end:]


def write_progress_file(
    *,
    progress_path: Path,
    experiments: Sequence[Experiment],
    completed: Sequence[ExperimentRunResult],
    current: Optional[Experiment],
    run_started_at: float,
) -> None:
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    total = len(experiments)
    done = len(completed)
    elapsed = time.time() - run_started_at
    completed_durations = [item.elapsed_seconds for item in completed if item.status == "completed" and item.elapsed_seconds > 0]
    avg_completed = sum(completed_durations) / len(completed_durations) if completed_durations else None
    remaining = total - done
    eta_seconds = avg_completed * remaining if avg_completed is not None else None

    payload = {
        "updated_at": utc_now_iso(),
        "total_experiments": total,
        "completed_experiments": done,
        "remaining_experiments": remaining,
        "current_experiment": current.name if current else None,
        "elapsed_seconds": round(elapsed, 3),
        "elapsed_human": format_duration(elapsed),
        "average_completed_experiment_seconds": round(avg_completed, 3) if avg_completed is not None else None,
        "eta_seconds": round(eta_seconds, 3) if eta_seconds is not None else None,
        "eta_human": format_duration(eta_seconds) if eta_seconds is not None else None,
        "completed": [item.__dict__ for item in completed],
        "planned": [
            {
                "name": exp.name,
                "baseline": exp.baseline,
                "results_folder": exp.results_folder,
                "dates": list(exp.dates),
            }
            for exp in experiments
        ],
    }
    tmp = progress_path.with_suffix(progress_path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(progress_path)


def run_experiment(
    exp: Experiment,
    *,
    python_exe: str,
    socket_host: str,
    socket_port: int,
    startup_timeout_s: float,
    log_root: Path,
    dry_run: bool,
    child_env_updates: Optional[dict[str, str]],
    skip_completed: bool,
    quiet_child_logs: bool,
) -> ExperimentRunResult:
    safe_name = exp.name.replace("/", "_").replace(" ", "_")
    log_dir = log_root / safe_name
    log_dir.mkdir(parents=True, exist_ok=True)

    # Preserve existing logs/results.  Add a separator so diagnostics for this
    # attempt remain readable without deleting previous evidence.
    run_header = f"\n===== run_experiments attempt started {utc_now_iso()} for {exp.name} =====\n"
    for log_name in (
        "pipeline.stdout.log",
        "pipeline.stderr.log",
        "emitter.stdout.log",
        "emitter.stderr.log",
    ):
        try:
            with (log_dir / log_name).open("a", encoding="utf-8", errors="replace") as log_file:
                log_file.write(run_header)
        except Exception:
            pass

    env = os.environ.copy()
    env.update(child_env_updates or {})
    env["PYTHONUNBUFFERED"] = "1"

    started_at = utc_now_iso()
    start_s = time.time()

    run_exp = exp
    if skip_completed and exp.dates:
        incomplete_dates = incomplete_real_dates(exp)
        if not incomplete_dates:
            elapsed = time.time() - start_s
            print(
                f"\n=== Skipping {exp.name}; replay result files already exist for all {len(exp.dates)} date(s) ===",
                flush=True,
            )
            return ExperimentRunResult(
                name=exp.name,
                baseline=exp.baseline,
                results_folder=exp.results_folder,
                dates=exp.dates,
                status="skipped",
                elapsed_seconds=elapsed,
                started_at=started_at,
                finished_at=utc_now_iso(),
            )
        if len(incomplete_dates) < len(exp.dates):
            print(
                f"\n=== Resuming {exp.name}; {len(exp.dates) - len(incomplete_dates)}/{len(exp.dates)} date(s) already complete, "
                f"running only incomplete date(s): {', '.join(incomplete_dates)} ===",
                flush=True,
            )
            run_exp = replace(
                exp,
                dates=tuple(incomplete_dates),
                emitter_args=_emitter_args_with_dates(exp.emitter_args, incomplete_dates),
            )

    # The runner now owns the strict skip/resume decision for real date chunks.
    # Once it decides to run, pass --no-skip-completed to full_pipeline for real
    # chunks so a laxer folder-exists check inside the pipeline cannot skip an
    # incomplete replay.  For non-real/no-date experiments, preserve the user's
    # original --skip-completed value.
    pipeline_skip_completed = False if (skip_completed and exp.dates) else skip_completed
    pipe_cmd = _pipeline_cmd(run_exp, python_exe, skip_completed=pipeline_skip_completed)
    emit_cmd = _emitter_cmd(run_exp, python_exe)

    if dry_run:
        print(f"\n=== DRY RUN: {run_exp.name} ===")
        print("pipeline:", " ".join(pipe_cmd))
        print("emitter: ", " ".join(emit_cmd))
        elapsed = time.time() - start_s
        return ExperimentRunResult(
            name=exp.name,
            baseline=exp.baseline,
            results_folder=exp.results_folder,
            dates=exp.dates,
            status="dry_run",
            elapsed_seconds=elapsed,
            started_at=started_at,
            finished_at=utc_now_iso(),
        )

    print(f"\n=== Running {exp.name} ===", flush=True)
    pipeline_ready = threading.Event()
    pipeline = _start_process(
        pipe_cmd,
        prefix="pipeline",
        log_dir=log_dir,
        env=env,
        quiet_child_logs=quiet_child_logs,
        readiness_event=pipeline_ready,
        readiness_markers=("Listening for parser reports",),
    )
    emitter: Optional[subprocess.Popen] = None

    try:
        _wait_for_pipeline_ready(
            pipeline,
            pipeline_ready,
            timeout_s=startup_timeout_s,
            host=socket_host,
            port=socket_port,
        )
        print(f"Pipeline reports it is listening at {socket_host}:{socket_port}", flush=True)

        emitter = _start_process(
            emit_cmd,
            prefix="emitter",
            log_dir=log_dir,
            env=env,
            quiet_child_logs=quiet_child_logs,
        )
        emitter_rc = emitter.wait()
        if emitter_rc != 0:
            raise RuntimeError(f"Emitter failed for {exp.name} with exit code {emitter_rc}")

        time.sleep(2.0)
        pipeline_rc = pipeline.poll()
        if pipeline_rc not in (None, 0):
            raise RuntimeError(f"Pipeline exited early for {exp.name} with exit code {pipeline_rc}")

        elapsed = time.time() - start_s
        print(f"=== Finished {run_exp.name} in {format_duration(elapsed)} ===", flush=True)
        return ExperimentRunResult(
            name=exp.name,
            baseline=exp.baseline,
            results_folder=exp.results_folder,
            dates=exp.dates,
            status="completed",
            elapsed_seconds=elapsed,
            started_at=started_at,
            finished_at=utc_now_iso(),
        )
    except Exception as exc:
        elapsed = time.time() - start_s

        # Give tee threads a moment to flush final lines, then include both
        # process return codes and all four log tails.
        time.sleep(0.25)
        pipeline_rc = pipeline.poll()
        emitter_rc = emitter.poll() if emitter is not None else None

        error_parts = [
            f"{type(exc).__name__}: {exc}",
            f"experiment={exp.name}",
            f"baseline={exp.baseline}",
            f"results_folder={exp.results_folder}",
            f"planned_dates={','.join(exp.dates) if exp.dates else '<none>'}",
            f"executed_dates={','.join(run_exp.dates) if run_exp.dates else '<none>'}",
            f"elapsed={format_duration(elapsed)}",
            f"pipeline_returncode={pipeline_rc}",
            f"emitter_returncode={emitter_rc if emitter is not None else '<not-started>'}",
            "pipeline_cmd=" + " ".join(pipe_cmd),
            "emitter_cmd=" + " ".join(emit_cmd),
        ]

        for label, rel_path in (
            ("pipeline.stdout.log", "pipeline.stdout.log"),
            ("pipeline.stderr.log", "pipeline.stderr.log"),
            ("emitter.stdout.log", "emitter.stdout.log"),
            ("emitter.stderr.log", "emitter.stderr.log"),
        ):
            tail = _tail_file(log_dir / rel_path, max_lines=120, max_chars=20000)
            if tail:
                error_parts.append(f"--- {label} tail ---\n{tail}")

        return ExperimentRunResult(
            name=exp.name,
            baseline=exp.baseline,
            results_folder=exp.results_folder,
            dates=exp.dates,
            status="failed",
            elapsed_seconds=elapsed,
            started_at=started_at,
            finished_at=utc_now_iso(),
            error="\n".join(error_parts),
        )
    finally:
        if emitter is not None:
            _terminate_process(emitter, name=f"emitter for {exp.name}", grace_s=3.0)
        _terminate_process(pipeline, name=f"pipeline for {exp.name}")


def _synthetic_emitter_args(
    *,
    batch_root: str | Sequence[str],
    recursive_discovery: bool,
    sync_multilevel_incidents: bool,
    multi_level_sort_mode: str,
    sensor_density: float = 1.0,
    missing_observations: float = 0.0,
    corrupt_observations: float = 0.0,
    perturbation_seed: str = "0",
    modality_filter: str = "all",
    duplicate_reports: int = 1,
    write_reports_to_incident_folders: Optional[bool] = None,
    reports_output_filename: Optional[str] = None,
    emit_to_socket: Optional[bool] = None,
    wait_for_socket_ack: Optional[bool] = None,
    print_reports: Optional[bool] = None,
    display_ground_truth: Optional[bool] = None,
    display_sensor_locations: Optional[bool] = None,
    skip_if_ground_truth_outside_grid: Optional[bool] = None,
) -> list[str]:
    roots = [str(x) for x in batch_root] if isinstance(batch_root, (list, tuple)) else [str(batch_root)]
    args = ["--batch-root", *roots]
    args.append("--recursive-discovery" if recursive_discovery else "--no-recursive-discovery")
    args.append("--sync-multilevel-incidents" if sync_multilevel_incidents else "--no-sync-multilevel-incidents")
    args += ["--multi-level-sort-mode", multi_level_sort_mode]
    args += ["--sensor-density", str(sensor_density)]
    args += ["--missing-observations", str(missing_observations)]
    args += ["--corrupt-observations", str(corrupt_observations)]
    args += ["--perturbation-seed", str(perturbation_seed)]
    args += ["--modality-filter", str(modality_filter)]
    args += ["--duplicate-reports", str(max(1, int(duplicate_reports or 1)))]
    if write_reports_to_incident_folders is not None:
        args.append("--write-reports-to-incident-folders" if write_reports_to_incident_folders else "--no-write-reports-to-incident-folders")
    if reports_output_filename:
        args += ["--reports-output-filename", str(reports_output_filename)]
    if emit_to_socket is not None:
        args.append("--emit-to-socket" if emit_to_socket else "--no-emit-to-socket")
    if wait_for_socket_ack is not None:
        args.append("--wait-for-socket-ack" if wait_for_socket_ack else "--no-wait-for-socket-ack")
    if print_reports is not None:
        args.append("--print-reports" if print_reports else "--no-print-reports")
    if display_ground_truth is not None:
        args.append("--display-ground-truth" if display_ground_truth else "--no-display-ground-truth")
    if display_sensor_locations is not None:
        args.append("--display-sensor-locations" if display_sensor_locations else "--no-display-sensor-locations")
    if skip_if_ground_truth_outside_grid is not None:
        args.append("--skip-if-ground-truth-outside-grid" if skip_if_ground_truth_outside_grid else "--no-skip-if-ground-truth-outside-grid")
    return args

def _real_emitter_args(args: argparse.Namespace, *, dates: Sequence[str]) -> list[str]:
    cmd = [
        "--dates",
        *list(dates),
        "--socket-host",
        args.socket_host,
        "--socket-port",
        str(args.socket_port),
        "--emit-to-socket",
        "--no-write-ordered-reports-jsonl",
        "--wait-for-socket-ack",
        "--replay-interval-seconds",
        str(args.real_replay_interval_seconds),
    ]
    if args.real_temp_root:
        cmd.extend(["--temp-root", str(args.real_temp_root)])
    if args.real_raw_root:
        cmd.extend(["--raw-root", str(args.real_raw_root)])
    cmd.append("--auto-prepare-temp-data" if args.real_auto_prepare_temp_data else "--no-auto-prepare-temp-data")
    cmd.append("--clear-temp-before-prepare" if args.real_clear_temp_before_prepare else "--no-clear-temp-before-prepare")
    cmd.append("--skip-existing-temp-data" if args.real_skip_existing_temp_data else "--no-skip-existing-temp-data")
    cmd.append("--emit-cached-positive-only" if args.real_emit_cached_positive_only else "--no-emit-cached-positive-only")
    if args.real_observation_cache_root:
        cmd += ["--observation-cache-root", str(args.real_observation_cache_root)]
    cmd += ["--cached-positive-missing-policy", str(args.real_cached_positive_missing_policy)]
    if args.real_data_sources:
        cmd += ["--data-sources", *args.real_data_sources]
    return cmd


def _pipeline_lazy_negative_args(args: argparse.Namespace, *, enable: bool) -> list[str]:
    """Return full_pipeline CLI args controlling lazy cached negative retrieval."""
    cmd = ["--lazy-negative-retrieval" if enable else "--no-lazy-negative-retrieval"]
    cmd += ["--lazy-negative-max-per-hour", str(args.real_lazy_negative_max_per_hour)]
    if args.real_observation_cache_root:
        cmd += ["--lazy-negative-cache-root", str(args.real_observation_cache_root)]
    return cmd





def discover_synthetic_incident_folders(batch_root: str | Path, *, observations_filename: str = "observations.txt") -> list[Path]:
    root = Path(batch_root)
    if (root / observations_filename).exists():
        return [root]
    if not root.exists():
        raise FileNotFoundError(f"Synthetic batch root does not exist: {root}")
    folders = [p.parent for p in root.glob(f"*/{observations_filename}")]
    return sorted(folders, key=lambda p: p.name)


def select_synthetic_incident_subset(
    *,
    batch_root: str | Path,
    k: int,
    seed: int,
    selection_file: str | Path,
) -> list[str]:
    """Select or reuse a deterministic K-incident subset for ablation/scalability runs."""
    selection_path = Path(selection_file)
    if selection_path.exists():
        roots = []
        with selection_path.open("r", encoding="utf-8") as infile:
            for line in infile:
                text = line.strip()
                if text and not text.startswith("#"):
                    roots.append(text)
        if roots:
            return roots

    folders = discover_synthetic_incident_folders(batch_root)
    if not folders:
        raise RuntimeError(f"No synthetic incident folders found under {batch_root}")
    rng = random.Random(int(seed))
    if len(folders) > int(k):
        folders = sorted(rng.sample(folders, int(k)), key=lambda p: p.name)
    selection_path.parent.mkdir(parents=True, exist_ok=True)
    with selection_path.open("w", encoding="utf-8") as outfile:
        outfile.write(f"# Deterministic synthetic incident subset: batch_root={batch_root}, k={k}, seed={seed}\n")
        for folder in folders:
            outfile.write(str(folder) + "\n")
    return [str(p) for p in folders]


def _synth_pipeline_common_args(args: argparse.Namespace) -> list[str]:
    return (["--anomaly-preprocessing"] if args.synth_anomaly_preprocessing else ["--no-anomaly-preprocessing"]) + ["--no-lazy-negative-retrieval"]


def _resolve_optional_file(path_value: Optional[str | Path]) -> Optional[Path]:
    """Resolve an optional file path relative to cwd, this script, or repo root."""
    if path_value is None:
        return None
    raw = Path(path_value).expanduser()
    candidates = [raw]
    if not raw.is_absolute():
        script_dir = Path(__file__).resolve().parent
        candidates.extend([
            Path.cwd() / raw,
            script_dir / raw,
            script_dir.parent / raw,
        ])
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate.resolve()
    return None


def _background_baseline_config_args(
    args: argparse.Namespace,
    baseline: str,
    *,
    config_attr: str,
    source_constraints_attr: str,
    generated_dir_attr: str,
    label: str,
) -> list[str]:
    """Return --baseline-config args for Hawkes/SaTScan-style baselines.

    If a full per-baseline config is supplied, use it directly. Otherwise, when
    a source-constraints JSON is available, write the small wrapper config shape
    expected by the baseline constructors: {"source_constraints": "..."}.
    """
    if baseline not in BACKGROUND_AWARE_BASELINE_SET:
        return []

    requested_config = getattr(args, config_attr)
    explicit_config = _resolve_optional_file(requested_config)
    if explicit_config is not None:
        return ["--baseline-config", str(explicit_config)]
    if requested_config:
        raise FileNotFoundError(f"{label} background baseline config not found: {requested_config}")

    source_constraints_value = getattr(args, source_constraints_attr)
    source_constraints = _resolve_optional_file(source_constraints_value)
    if source_constraints is None:
        return []

    config_dir = Path(getattr(args, generated_dir_attr))
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / f"{label}_{baseline}.json"
    payload = {"source_constraints": str(source_constraints)}
    config_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return ["--baseline-config", str(config_path)]


def _synth_background_baseline_config_args(args: argparse.Namespace, baseline: str) -> list[str]:
    return _background_baseline_config_args(
        args,
        baseline,
        config_attr="synthetic_background_baseline_config",
        source_constraints_attr="synthetic_source_constraints",
        generated_dir_attr="synthetic_generated_baseline_config_dir",
        label="synthetic",
    )


def _real_background_baseline_config_args(args: argparse.Namespace, baseline: str) -> list[str]:
    return _background_baseline_config_args(
        args,
        baseline,
        config_attr="real_background_baseline_config",
        source_constraints_attr="real_background_source_constraints",
        generated_dir_attr="real_generated_baseline_config_dir",
        label="real",
    )


def _synth_baseline_pipeline_args(args: argparse.Namespace, baseline: str) -> list[str]:
    return _synth_pipeline_common_args(args) + _synth_background_baseline_config_args(args, baseline)


def _real_pipeline_common_args(args: argparse.Namespace) -> list[str]:
    return (
        ["--anomaly-preprocessing"]
        if args.real_anomaly_preprocessing
        else ["--no-anomaly-preprocessing"]
    ) + _pipeline_lazy_negative_args(args, enable=bool(args.real_lazy_negative_retrieval))


def _real_background_pipeline_common_args(args: argparse.Namespace) -> list[str]:
    """Return pipeline flags for real-data Hawkes/SaTScan baselines.

    These baselines consume observation-model outputs directly. They do not use
    IncidentLens hypothesis propagation or lazy negative retrieval, and for the
    cached-positive real-data replay they should not pay the anomaly-preprocessing
    startup/runtime cost unless explicitly requested.
    """
    return (
        ["--anomaly-preprocessing"]
        if args.real_background_anomaly_preprocessing
        else ["--no-anomaly-preprocessing"]
    ) + _pipeline_lazy_negative_args(args, enable=bool(args.real_background_lazy_negative_retrieval))


def _real_baseline_pipeline_args(args: argparse.Namespace, baseline: str) -> list[str]:
    if baseline in BACKGROUND_AWARE_BASELINE_SET:
        return _real_background_pipeline_common_args(args) + _real_background_baseline_config_args(args, baseline)
    return _real_pipeline_common_args(args) + _real_background_baseline_config_args(args, baseline)


def _real_default_baselines(args: argparse.Namespace) -> list[str]:
    baselines = list(args.baselines or DEFAULT_REAL_BASELINES)
    if args.include_real_background:
        for baseline in DEFAULT_REAL_BACKGROUND_BASELINES:
            if baseline not in baselines:
                baselines.append(baseline)

    real_excluded = set(args.real_exclude_baselines or [])
    if real_excluded:
        baselines = [b for b in baselines if b not in real_excluded]
    return baselines


def build_synth_experiments(args: argparse.Namespace) -> list[Experiment]:
    baselines = list(args.baselines or DEFAULT_SYNTHETIC_BASELINES)
    if args.only_baseline:
        baselines = [b for b in baselines if b == args.only_baseline]

    experiments: list[Experiment] = []

    if args.include_composition and (args.only_baseline in (None, "incidentlens")):
        experiments.append(
            Experiment(
                name="incidentlens_synth_comp",
                baseline="incidentlens",
                incident_types=args.composition_incident_types,
                results_folder=args.composition_results_folder,
                emitter_module=args.synthetic_emitter_module,
                emitter_args=_synthetic_emitter_args(
                    batch_root=args.composition_batch_root,
                    recursive_discovery=args.composition_recursive_discovery,
                    sync_multilevel_incidents=False,
                    multi_level_sort_mode=args.multi_level_sort_mode,
                ),
                pipeline_args=_synth_pipeline_common_args(args),
            )
        )

    if args.include_low_level:
        for baseline in baselines:
            experiments.append(
                Experiment(
                    name=f"{baseline}_synth_low",
                    baseline=baseline,
                    incident_types=args.low_level_incident_types,
                    results_folder=args.low_level_results_folder,
                    emitter_module=args.synthetic_emitter_module,
                    emitter_args=_synthetic_emitter_args(
                        batch_root=args.low_level_batch_root,
                        recursive_discovery=False,
                        sync_multilevel_incidents=False,
                        multi_level_sort_mode=args.multi_level_sort_mode,
                    ),
                    pipeline_args=_synth_baseline_pipeline_args(args, baseline),
                )
            )

    return experiments



def _coverage_smoke_cases(args: argparse.Namespace) -> list[tuple[str, float, float]]:
    """Return (case_name, sensor_density, missing_observation_fraction)."""
    cases = [
        ("none", 0.0, 0.0),
        ("low", float(args.coverage_smoke_low_sensor_density), float(args.coverage_smoke_low_missing_observations)),
        ("medium", float(args.coverage_smoke_medium_sensor_density), float(args.coverage_smoke_medium_missing_observations)),
        ("high", float(args.coverage_smoke_high_sensor_density), float(args.coverage_smoke_high_missing_observations)),
    ]
    if args.only_coverage_smoke_case:
        wanted = str(args.only_coverage_smoke_case)
        cases = [case for case in cases if case[0] == wanted]
    return cases


def build_coverage_smoke_experiments(args: argparse.Namespace) -> list[Experiment]:
    """Build a small synthetic coverage sanity suite.

    The same K incident folders are replayed four times with increasing emitted
    source/report availability.  Each replay also writes the exact normalized
    REPORT stream used for that condition into the simulator incident folder
    using a condition-specific filename.  evaluate_results.py --mode
    coverage_smoke then uses those files as the non-circular coverage inventory
    for the matching experiment.
    """
    selected_roots = select_synthetic_incident_subset(
        batch_root=args.coverage_smoke_batch_root,
        k=args.coverage_smoke_k,
        seed=args.coverage_smoke_seed,
        selection_file=args.coverage_smoke_selection_file,
    )
    baselines = list(args.coverage_smoke_baselines or ["incidentlens"])
    if args.only_baseline:
        baselines = [b for b in baselines if b == args.only_baseline]

    experiments: list[Experiment] = []
    for case_name, density, missing in _coverage_smoke_cases(args):
        results_folder = f"{args.coverage_smoke_results_prefix}_{case_name}"
        reports_filename = args.coverage_smoke_reports_filename_template.format(
            experiment=results_folder,
            case=case_name,
            prefix=args.coverage_smoke_results_prefix,
        )
        for baseline in baselines:
            experiments.append(
                Experiment(
                    name=f"{baseline}_{results_folder}",
                    baseline=baseline,
                    incident_types=args.coverage_smoke_incident_types,
                    results_folder=results_folder,
                    emitter_module=args.synthetic_emitter_module,
                    emitter_args=_synthetic_emitter_args(
                        batch_root=selected_roots,
                        recursive_discovery=False,
                        sync_multilevel_incidents=False,
                        multi_level_sort_mode=args.multi_level_sort_mode,
                        sensor_density=density,
                        missing_observations=missing,
                        corrupt_observations=0.0,
                        perturbation_seed=f"coverage_smoke_{args.coverage_smoke_seed}_{case_name}",
                        modality_filter="all",
                        write_reports_to_incident_folders=True,
                        reports_output_filename=reports_filename,
                        emit_to_socket=True,
                        wait_for_socket_ack=True,
                        print_reports=False,
                        display_ground_truth=False,
                        display_sensor_locations=False,
                        skip_if_ground_truth_outside_grid=False,
                    ),
                    pipeline_args=_synth_baseline_pipeline_args(args, baseline),
                )
            )
    return experiments


def build_real_experiments(args: argparse.Namespace) -> list[Experiment]:
    baselines = _real_default_baselines(args)
    if args.only_baseline:
        baselines = [b for b in baselines if b == args.only_baseline]

    real_dates = list(args.real_dates or DEFAULT_REAL_DATES)
    if args.real_dates_file:
        real_dates = _read_dates_file(args.real_dates_file)
    real_non_dates = list(args.real_non_dates or DEFAULT_REAL_NON_DATES)
    if args.real_non_dates_file:
        real_non_dates = _read_dates_file(args.real_non_dates_file)

    experiments: list[Experiment] = []

    if args.include_real_all:
        for chunk in contiguous_date_chunks(real_dates):
            start, end = chunk[0], chunk[-1]
            chunk_name = start if start == end else f"{start}_{end}"
            for baseline in baselines:
                experiments.append(
                    Experiment(
                        name=f"{baseline}_real_all_{chunk_name}",
                        baseline=baseline,
                        incident_types=args.real_incident_types,
                        results_folder=args.real_results_folder,
                        emitter_module=args.real_emitter_module,
                        emitter_args=_real_emitter_args(args, dates=chunk),
                        dates=tuple(chunk),
                        pipeline_args=_real_baseline_pipeline_args(args, baseline),
                    )
                )

    if args.include_real_non and (args.only_baseline in (None, "incidentlens")):
        for chunk in contiguous_date_chunks(real_non_dates):
            start, end = chunk[0], chunk[-1]
            chunk_name = start if start == end else f"{start}_{end}"
            experiments.append(
                Experiment(
                    name=f"incidentlens_real_non_{chunk_name}",
                    baseline="incidentlens",
                    incident_types=args.real_incident_types,
                    results_folder=args.real_non_results_folder,
                    emitter_module=args.real_emitter_module,
                    emitter_args=_real_emitter_args(args, dates=chunk),
                    dates=tuple(chunk),
                    pipeline_args=_real_baseline_pipeline_args(args, "incidentlens"),
                )
            )

    return experiments



def build_real_background_experiments(args: argparse.Namespace) -> list[Experiment]:
    """Build real-data Hawkes and SaTScan background-aware baseline experiments.

    This is separated from the main real suite so you can run the new baselines
    without re-running every existing real-data baseline. It uses the same real
    emitter, date chunking, incident-type list, and cached-positive replay knobs
    as the main real suite.
    """
    baselines = list(args.real_background_baselines or DEFAULT_REAL_BACKGROUND_BASELINES)
    if args.only_baseline:
        baselines = [b for b in baselines if b == args.only_baseline]

    real_dates = list(args.real_dates or DEFAULT_REAL_DATES)
    if args.real_dates_file:
        real_dates = _read_dates_file(args.real_dates_file)

    incident_types = args.real_background_incident_types or args.real_incident_types
    results_folder = args.real_background_results_folder or args.real_results_folder

    experiments: list[Experiment] = []
    if args.include_real_all:
        for chunk in contiguous_date_chunks(real_dates):
            start, end = chunk[0], chunk[-1]
            chunk_name = start if start == end else f"{start}_{end}"
            for baseline in baselines:
                experiments.append(
                    Experiment(
                        name=f"{baseline}_real_background_{chunk_name}",
                        baseline=baseline,
                        incident_types=incident_types,
                        results_folder=results_folder,
                        emitter_module=args.real_emitter_module,
                        emitter_args=_real_emitter_args(args, dates=chunk),
                        dates=tuple(chunk),
                        pipeline_args=_real_baseline_pipeline_args(args, baseline),
                    )
                )
    return experiments


def build_modality_ablation_experiments(args: argparse.Namespace) -> list[Experiment]:
    """Build IncidentLens modality-ablation experiments over a fixed incident subset.

    The two variants reuse the same deterministic K incident folders and keep the
    detector architecture fixed to full IncidentLens.  Only the emitted reports
    change:
      * sensor_only: camera/traffic/air-quality/weather observations
      * operational_text_only: CitizenApp/X/official-alert observations, excluding
        label-construction news/article text
    """
    if args.only_baseline not in (None, "incidentlens"):
        return []

    selected_roots = select_synthetic_incident_subset(
        batch_root=args.modality_ablation_batch_root,
        k=args.modality_ablation_k,
        seed=args.modality_ablation_seed,
        selection_file=args.modality_ablation_selection_file,
    )
    base_emitter_args = {
        "batch_root": selected_roots,
        "recursive_discovery": False,
        "sync_multilevel_incidents": False,
        "multi_level_sort_mode": args.multi_level_sort_mode,
        "sensor_density": 1.0,
        "missing_observations": 0.0,
        "corrupt_observations": 0.0,
        "perturbation_seed": str(args.modality_ablation_seed),
    }
    common_pipeline = _synth_pipeline_common_args(args)
    experiments: list[Experiment] = []

    variants = [
        ("sensor_only", "sensor_only"),
        ("operational_text_only", "operational_text_only"),
    ]
    for variant_name, modality_filter in variants:
        if args.only_modality_variant and variant_name != args.only_modality_variant:
            continue
        experiments.append(
            Experiment(
                name=f"modality_ablation_{variant_name}",
                baseline="incidentlens",
                incident_types=args.modality_ablation_incident_types,
                results_folder=f"{args.modality_ablation_results_prefix}_{variant_name}",
                emitter_module=args.synthetic_emitter_module,
                emitter_args=_synthetic_emitter_args(
                    **base_emitter_args,
                    modality_filter=modality_filter,
                ),
                pipeline_args=common_pipeline + [
                    "--architectural-ablation",
                    "full",
                    "--max-active-particles",
                    str(args.scalability_default_active_particle_cap),
                ],
            )
        )

    return experiments


def build_ablation_and_scalability_experiments(args: argparse.Namespace) -> list[Experiment]:
    """Build synthetic architectural-ablation and scalability/stress experiments.

    All variants use the same deterministic K incident folders so accuracy/runtime
    differences are comparable across variants.
    """
    selected_roots = select_synthetic_incident_subset(
        batch_root=args.ablation_scalability_batch_root,
        k=args.ablation_scalability_k,
        seed=args.ablation_scalability_seed,
        selection_file=args.ablation_scalability_selection_file,
    )
    base_emitter_args = {
        "batch_root": selected_roots,
        "recursive_discovery": False,
        "sync_multilevel_incidents": False,
        "multi_level_sort_mode": args.multi_level_sort_mode,
        "perturbation_seed": str(args.ablation_scalability_seed),
    }
    common_pipeline = _synth_pipeline_common_args(args)
    experiments: list[Experiment] = []

    ablations = [
        ("full_incidentlens", "incidentlens", "full"),
        ("generic_propagation", "generic_propagation", "generic_propagation"),
        ("no_reverse_proposal", "incidentlens", "no_reverse_proposal"),
        ("no_source_priors", "incidentlens", "no_source_priors"),
        ("no_hypothesis_diversity", "incidentlens", "no_hypothesis_diversity"),
        ("generic_clustering_stability", "incidentlens", "generic_clustering_stability"),
    ]
    for variant_name, baseline, ablation in ablations:
        if args.only_ablation_variant and variant_name != args.only_ablation_variant:
            continue
        experiments.append(
            Experiment(
                name=f"ablation_{variant_name}",
                baseline=baseline,
                incident_types=args.ablation_scalability_incident_types,
                results_folder=f"{args.ablation_scalability_results_prefix}_ablation_{variant_name}",
                emitter_module=args.synthetic_emitter_module,
                emitter_args=_synthetic_emitter_args(
                    **base_emitter_args,
                    sensor_density=1.0,
                    missing_observations=0.0,
                    corrupt_observations=0.0,
                ),
                pipeline_args=common_pipeline + ["--architectural-ablation", ablation, "--max-active-particles", str(args.scalability_default_active_particle_cap)],
            )
        )

    # Runtime/scalability: vary one dimension at a time from the default.
    default_cap = int(args.scalability_default_active_particle_cap)
    default_density = 1.0
    default_missing = 0.0
    default_corrupt = 0.0
    scalability_cases: list[tuple[str, int, float, float, float]] = []
    seen_case_keys: set[tuple[int, float, float, float]] = set()

    def add_case(name: str, cap: int, density: float, missing: float, corrupt: float) -> None:
        key = (int(cap), round(float(density), 4), round(float(missing), 4), round(float(corrupt), 4))
        if key in seen_case_keys:
            return
        seen_case_keys.add(key)
        scalability_cases.append((name, int(cap), float(density), float(missing), float(corrupt)))

    add_case("default", default_cap, default_density, default_missing, default_corrupt)
    for cap in [100, 500, 1000]:
        add_case(f"active_particles_{cap}", cap, default_density, default_missing, default_corrupt)
    for density in [1.0, 0.5, 0.25]:
        add_case(f"sensor_density_{int(density * 100)}", default_cap, density, default_missing, default_corrupt)
    for missing in [0.0, 0.25, 0.5]:
        add_case(f"missing_{int(missing * 100)}", default_cap, default_density, missing, default_corrupt)
    for corrupt in [0.0, 0.25, 0.5]:
        add_case(f"corrupt_{int(corrupt * 100)}", default_cap, default_density, default_missing, corrupt)

    for case_name, cap, density, missing, corrupt in scalability_cases:
        if args.only_scalability_case and case_name != args.only_scalability_case:
            continue
        experiments.append(
            Experiment(
                name=f"scalability_{case_name}",
                baseline="incidentlens",
                incident_types=args.ablation_scalability_incident_types,
                results_folder=f"{args.ablation_scalability_results_prefix}_scalability_{case_name}",
                emitter_module=args.synthetic_emitter_module,
                emitter_args=_synthetic_emitter_args(
                    **base_emitter_args,
                    sensor_density=density,
                    missing_observations=missing,
                    corrupt_observations=corrupt,
                ),
                pipeline_args=common_pipeline + ["--architectural-ablation", "full", "--max-active-particles", str(cap)],
            )
        )

    return experiments


def build_ablation_last_experiments(args: argparse.Namespace) -> list[Experiment]:
    """Run only the late/fixed generic ablation variants on the same K incidents.

    This is useful when the full ablation/scalability suite has already run and
    only the corrected generic variants need to be filled in.  Both variants are
    intentionally launched as baseline="incidentlens" with an architectural
    ablation flag so the results are written under:

        evaluation/results/incidentlens/ablation_scalability_ablation_generic_all/
        evaluation/results/incidentlens/ablation_scalability_ablation_generic_propagation/

    rather than under evaluation/results/generic_all/ or
    evaluation/results/generic_propagation/.
    """
    selected_roots = select_synthetic_incident_subset(
        batch_root=args.ablation_scalability_batch_root,
        k=args.ablation_scalability_k,
        seed=args.ablation_scalability_seed,
        selection_file=args.ablation_scalability_selection_file,
    )
    base_emitter_args = {
        "batch_root": selected_roots,
        "recursive_discovery": False,
        "sync_multilevel_incidents": False,
        "multi_level_sort_mode": args.multi_level_sort_mode,
        "perturbation_seed": str(args.ablation_scalability_seed),
    }
    common_pipeline = _synth_pipeline_common_args(args)

    experiments: list[Experiment] = []
    variants = [
        # Correct generic propagation ablation: task labels/source priors/
        # clustering remain incident-specific; only propagation is generic.
        ("generic_propagation", "generic_propagation"),
        # Full generic architectural baseline: generic propagation + generic/
        # uniform source priors + generic clustering/stability, while final
        # prediction still selects a label from the incident list.
        ("generic_all", "generic_all"),
    ]

    for variant_name, architectural_ablation in variants:
        if args.only_ablation_variant and variant_name != args.only_ablation_variant:
            continue
        experiments.append(
            Experiment(
                name=f"ablation_{variant_name}",
                baseline="incidentlens",
                incident_types=args.ablation_scalability_incident_types,
                results_folder=f"{args.ablation_scalability_results_prefix}_ablation_{variant_name}",
                emitter_module=args.synthetic_emitter_module,
                emitter_args=_synthetic_emitter_args(
                    **base_emitter_args,
                    sensor_density=1.0,
                    missing_observations=0.0,
                    corrupt_observations=0.0,
                ),
                pipeline_args=common_pipeline + [
                    "--architectural-ablation",
                    architectural_ablation,
                    "--max-active-particles",
                    str(args.scalability_default_active_particle_cap),
                ],
            )
        )

    return experiments


def build_timing_experiments(args: argparse.Namespace) -> list[Experiment]:
    """Run one-incident timing/scalability cases with full anomaly + observation models.

    These runs are intended to estimate end-to-end timing/throughput/memory under
    the same one-factor scalability perturbations, but without using cached
    observation outputs.  Each case runs exactly one synthetic incident folder
    (default: simulator/generated/batch_incident_runs/wildfire1).
    """
    timing_root = str(args.timing_incident_folder)
    base_emitter_args = {
        "batch_root": timing_root,
        "recursive_discovery": False,
        "sync_multilevel_incidents": False,
        "multi_level_sort_mode": args.multi_level_sort_mode,
        "perturbation_seed": str(args.timing_seed),
    }

    # Force both stages to run:
    #   --anomaly-preprocessing enables the anomaly detector.
    #   --no-anomaly-skip-low-relevance prevents the anomaly gate from skipping
    #     the heavy observation model on low-relevance reports.
    #   --no-observation-cache disables cached observation-model outputs.
    # The anomaly cache flags are included for compatibility with anomaly-only
    # or future anomaly-cache paths.
    common_pipeline = [
        "--anomaly-preprocessing",
        "--no-anomaly-skip-low-relevance",
        "--no-observation-cache",
        "--no-observation-cache-read",
        "--no-observation-cache-write",
        "--no-anomaly-cache-read",
        "--no-anomaly-cache-write",
        "--no-lazy-negative-retrieval",
        "--particle-log-mode",
        "off",
    ]

    default_cap = int(args.scalability_default_active_particle_cap)
    default_density = 1.0
    default_missing = 0.0
    default_corrupt = 0.0
    scalability_cases: list[tuple[str, int, float, float, float]] = []
    seen_case_keys: set[tuple[int, float, float, float]] = set()

    def add_case(name: str, cap: int, density: float, missing: float, corrupt: float) -> None:
        key = (int(cap), round(float(density), 4), round(float(missing), 4), round(float(corrupt), 4))
        if key in seen_case_keys:
            return
        seen_case_keys.add(key)
        scalability_cases.append((name, int(cap), float(density), float(missing), float(corrupt)))

    add_case("default", default_cap, default_density, default_missing, default_corrupt)
    for cap in [100, 500, 1000]:
        add_case(f"active_particles_{cap}", cap, default_density, default_missing, default_corrupt)
    for density in [1.0, 0.5, 0.25]:
        add_case(f"sensor_density_{int(density * 100)}", default_cap, density, default_missing, default_corrupt)
    for missing in [0.0, 0.25, 0.5]:
        add_case(f"missing_{int(missing * 100)}", default_cap, default_density, missing, default_corrupt)
    for corrupt in [0.0, 0.25, 0.5]:
        add_case(f"corrupt_{int(corrupt * 100)}", default_cap, default_density, default_missing, corrupt)

    experiments: list[Experiment] = []
    for case_name, cap, density, missing, corrupt in scalability_cases:
        if args.only_timing_case and case_name != args.only_timing_case:
            continue
        experiments.append(
            Experiment(
                name=f"timing_{case_name}",
                baseline="incidentlens",
                incident_types=args.timing_incident_types,
                results_folder=f"{args.timing_results_prefix}_scalability_{case_name}",
                emitter_module=args.synthetic_emitter_module,
                emitter_args=_synthetic_emitter_args(
                    **base_emitter_args,
                    sensor_density=density,
                    missing_observations=missing,
                    corrupt_observations=corrupt,
                ),
                pipeline_args=common_pipeline + [
                    "--architectural-ablation",
                    "full",
                    "--max-active-particles",
                    str(cap),
                ],
            )
        )

    return experiments



def _actual_timing_incident_roots(args: argparse.Namespace) -> list[str]:
    """Resolve the fixed incident folders used for actual_timing runs."""
    roots: list[str] = []
    batch_root = Path(args.actual_timing_batch_root)
    for value in args.actual_timing_incident_names:
        candidate = Path(str(value))
        if candidate.is_absolute() or len(candidate.parts) > 1:
            root = candidate
        else:
            root = batch_root / str(value)
        roots.append(str(root))
    return roots


def build_actual_timing_experiments(args: argparse.Namespace) -> list[Experiment]:
    """Controlled throughput-vs-report-count experiment with full OM/anomaly.

    This runs a fixed incident set (default wildfire1 only) under
    report duplication factors 1, 2, 3, and 5.  The emitter duplicates every
    post-filter report with unique report IDs, while the pipeline forces anomaly
    preprocessing and observation-model inference and disables caches, so wall
    clock timing reflects the full online processing path.
    """
    if args.only_baseline not in (None, "incidentlens"):
        return []

    selected_roots = _actual_timing_incident_roots(args)
    common_pipeline = [
        "--anomaly-preprocessing",
        "--no-anomaly-skip-low-relevance",
        "--no-observation-cache",
        "--no-observation-cache-read",
        "--no-observation-cache-write",
        "--no-anomaly-cache-read",
        "--no-anomaly-cache-write",
        "--no-lazy-negative-retrieval",
        "--particle-log-mode",
        "off",
        "--architectural-ablation",
        "full",
        "--max-active-particles",
        str(args.actual_timing_active_particle_cap),
    ]

    experiments: list[Experiment] = []
    for duplicate_factor in args.actual_timing_duplication_levels:
        duplicate_factor = int(duplicate_factor)
        if duplicate_factor < 1:
            raise ValueError("actual timing duplication levels must be >= 1")
        if args.only_actual_timing_duplication is not None and duplicate_factor != int(args.only_actual_timing_duplication):
            continue
        experiments.append(
            Experiment(
                name=f"actual_timing_dup{duplicate_factor}",
                baseline="incidentlens",
                incident_types=args.actual_timing_incident_types,
                results_folder=f"{args.actual_timing_results_prefix}_dup{duplicate_factor}",
                emitter_module=args.synthetic_emitter_module,
                emitter_args=_synthetic_emitter_args(
                    batch_root=selected_roots,
                    recursive_discovery=False,
                    sync_multilevel_incidents=False,
                    multi_level_sort_mode=args.multi_level_sort_mode,
                    sensor_density=1.0,
                    missing_observations=0.0,
                    corrupt_observations=0.0,
                    perturbation_seed=str(args.actual_timing_seed),
                    modality_filter="all",
                    duplicate_reports=duplicate_factor,
                ),
                pipeline_args=common_pipeline,
            )
        )
    return experiments


def build_experiments(args: argparse.Namespace) -> list[Experiment]:
    experiments: list[Experiment] = []
    if args.exp_type in {"synth", "all"}:
        experiments.extend(build_synth_experiments(args))
    if args.exp_type in {"real", "all"}:
        experiments.extend(build_real_experiments(args))
    if args.exp_type == "real_background":
        experiments.extend(build_real_background_experiments(args))
    if args.exp_type == "ablation_and_scalability":
        experiments.extend(build_ablation_and_scalability_experiments(args))
    if args.exp_type == "ablation_last":
        experiments.extend(build_ablation_last_experiments(args))
    if args.exp_type == "timing":
        experiments.extend(build_timing_experiments(args))
    if args.exp_type == "actual_timing":
        experiments.extend(build_actual_timing_experiments(args))
    if args.exp_type == "modality_ablation":
        experiments.extend(build_modality_ablation_experiments(args))
    if args.exp_type == "coverage_smoke":
        experiments.extend(build_coverage_smoke_experiments(args))
    return experiments


def print_progress_summary(completed: Sequence[ExperimentRunResult], total: int, run_started_at: float) -> None:
    done = len(completed)
    elapsed = time.time() - run_started_at
    completed_durations = [item.elapsed_seconds for item in completed if item.status == "completed" and item.elapsed_seconds > 0]
    avg_completed = sum(completed_durations) / len(completed_durations) if completed_durations else None
    remaining = total - done
    eta = avg_completed * remaining if avg_completed is not None else None
    eta_text = format_duration(eta) if eta is not None else "unknown"
    avg_text = format_duration(avg_completed) if avg_completed is not None else "unknown"
    print(
        f"Progress: {done}/{total} experiments complete; "
        f"elapsed={format_duration(elapsed)}; "
        f"avg_completed={avg_text}; "
        f"ETA={eta_text}",
        flush=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--exp-type",
        "--exp_type",
        choices=["synth", "real", "all", "real_background", "ablation_and_scalability", "ablation_last", "modality_ablation", "timing", "actual_timing", "coverage_smoke"],
        default="synth",
    )
    parser.add_argument("--python", default=sys.executable, help="Python executable to use. Defaults to the Python running this script.")
    parser.add_argument("--venv", default=None, help="Optional virtual environment directory, e.g. venv or .venv.")
    parser.add_argument("--socket-host", default="127.0.0.1")
    parser.add_argument("--socket-port", type=int, default=8765)
    parser.add_argument("--startup-timeout-s", type=float, default=120.0)
    parser.add_argument("--log-root", default="evaluation/experiment_logs")
    parser.add_argument("--progress-json", default="evaluation/experiment_logs/progress.json")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-completed", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--quiet-child-logs",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Still writes pipeline/emitter logs to files, but does not mirror every line to the terminal.",
    )
    parser.add_argument(
        "--no-progress-bar",
        action="store_true",
        help="Disable the outer tqdm progress bar even if tqdm is installed.",
    )

    parser.add_argument(
        "--baselines",
        nargs="*",
        default=None,
        help=(
            "Optional explicit baseline list. Defaults: synthetic runs use "
            "DEFAULT_SYNTHETIC_BASELINES, real runs use DEFAULT_REAL_BASELINES."
        ),
    )
    parser.add_argument("--only-baseline", default=None)
    parser.add_argument("--multi-level-sort-mode", choices=["time_then_step", "step_then_time"], default="time_then_step")

    # Synthetic experiments.
    parser.add_argument("--include-composition", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-low-level", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--synthetic-emitter-module", default="evaluation.synthetic_emitter")
    parser.add_argument("--composition-batch-root", default="simulator/generated/batch_small_area")
    parser.add_argument("--composition-incident-types", default="evaluation/incident_list_synth_area.txt")
    parser.add_argument("--composition-results-folder", default="synth_comp")
    parser.add_argument("--composition-recursive-discovery", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--low-level-batch-root", default="simulator/generated/batch_incident_runs")
    parser.add_argument("--low-level-incident-types", default="evaluation/incident_list_synth_batch.txt")
    parser.add_argument("--low-level-results-folder", default="synth_low")
    parser.add_argument("--synth-anomaly-preprocessing", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--synthetic-source-constraints",
        default="generic_incident_source_constraints.json",
        help=(
            "Optional source-constraints JSON used to auto-generate --baseline-config "
            "for satscan_background and hawkes_event_detector synthetic runs. "
            "If the file is not found, those baselines run with their constructor defaults."
        ),
    )
    parser.add_argument(
        "--synthetic-background-baseline-config",
        default=None,
        help=(
            "Optional explicit JSON config passed to both satscan_background and "
            "hawkes_event_detector during synthetic runs. Overrides "
            "--synthetic-source-constraints."
        ),
    )
    parser.add_argument(
        "--synthetic-generated-baseline-config-dir",
        default="evaluation/experiment_logs/generated_baseline_configs",
        help="Directory for generated one-baseline config wrappers for synthetic background baselines.",
    )

    # Synthetic coverage/correlation smoke test.
    parser.add_argument("--coverage-smoke-batch-root", default="simulator/generated/batch_incident_runs")
    parser.add_argument("--coverage-smoke-incident-types", default="evaluation/incident_list_synth_batch.txt")
    parser.add_argument("--coverage-smoke-results-prefix", default="coverage_smoke")
    parser.add_argument("--coverage-smoke-k", type=int, default=8)
    parser.add_argument("--coverage-smoke-seed", type=int, default=23)
    parser.add_argument(
        "--coverage-smoke-selection-file",
        default="evaluation/experiment_selections/coverage_smoke_k8_seed23.txt",
        help="File storing the fixed K synthetic incidents used by every coverage-smoke condition.",
    )
    parser.add_argument(
        "--coverage-smoke-baselines",
        nargs="*",
        default=["incidentlens"],
        help="Baselines to replay for the coverage-smoke suite. Default: incidentlens only.",
    )
    parser.add_argument(
        "--only-coverage-smoke-case",
        choices=["none", "low", "medium", "high"],
        default=None,
        help="Optionally run only one coverage-smoke condition.",
    )
    parser.add_argument(
        "--coverage-smoke-reports-filename-template",
        default="{experiment}_reports.jsonl",
        help="Condition-specific normalized REPORT JSONL filename written into each selected synthetic incident folder.",
    )
    parser.add_argument("--coverage-smoke-low-sensor-density", type=float, default=0.25)
    parser.add_argument("--coverage-smoke-low-missing-observations", type=float, default=0.50)
    parser.add_argument("--coverage-smoke-medium-sensor-density", type=float, default=0.50)
    parser.add_argument("--coverage-smoke-medium-missing-observations", type=float, default=0.25)
    parser.add_argument("--coverage-smoke-high-sensor-density", type=float, default=1.0)
    parser.add_argument("--coverage-smoke-high-missing-observations", type=float, default=0.0)

    # Synthetic architectural ablation + scalability/stress experiments.
    parser.add_argument("--ablation-scalability-batch-root", default="simulator/generated/batch_incident_runs")
    parser.add_argument("--ablation-scalability-incident-types", default="evaluation/incident_list_synth_batch.txt")
    parser.add_argument("--ablation-scalability-results-prefix", default="ablation_scalability")
    parser.add_argument("--ablation-scalability-k", type=int, default=40)
    parser.add_argument("--ablation-scalability-seed", type=int, default=13)
    parser.add_argument(
        "--ablation-scalability-selection-file",
        default="evaluation/experiment_selections/ablation_scalability_k40_seed13.txt",
        help="File storing the fixed K synthetic incidents used by every ablation/scalability variant.",
    )

    # Synthetic modality ablations for IncidentLens.
    parser.add_argument("--modality-ablation-batch-root", default="simulator/generated/batch_incident_runs")
    parser.add_argument("--modality-ablation-incident-types", default="evaluation/incident_list_synth_batch.txt")
    parser.add_argument("--modality-ablation-results-prefix", default="modality_ablation")
    parser.add_argument("--modality-ablation-k", type=int, default=40)
    parser.add_argument("--modality-ablation-seed", type=int, default=13)
    parser.add_argument(
        "--modality-ablation-selection-file",
        default="evaluation/experiment_selections/ablation_scalability_k40_seed13.txt",
        help="File storing the fixed 40 synthetic incidents used by each modality-ablation variant; defaults to the shared ablation/scalability selection.",
    )
    parser.add_argument(
        "--only-modality-variant",
        choices=["sensor_only", "operational_text_only"],
        default=None,
        help="Optionally run only one modality-ablation variant.",
    )

    parser.add_argument("--scalability-default-active-particle-cap", type=int, default=100)
    parser.add_argument("--only-ablation-variant", default=None)
    parser.add_argument("--only-scalability-case", default=None)

    # One-incident timing experiments. These reuse the scalability perturbation
    # cases but force anomaly + observation model execution and disable caches.
    parser.add_argument("--timing-incident-folder", default="simulator/generated/batch_incident_runs/wildfire1")
    parser.add_argument("--timing-incident-types", default="evaluation/incident_list_synth_batch.txt")
    parser.add_argument("--timing-results-prefix", default="timing")
    parser.add_argument("--timing-seed", type=int, default=13)
    parser.add_argument("--only-timing-case", default=None)

    # Controlled throughput-vs-report-count timing experiment.
    parser.add_argument("--actual-timing-batch-root", default="simulator/generated/batch_incident_runs")
    parser.add_argument("--actual-timing-incident-names", nargs="*", default=["wildfire1"])
    parser.add_argument("--actual-timing-incident-types", default="evaluation/incident_list_synth_batch.txt")
    parser.add_argument("--actual-timing-results-prefix", default="actual_timing")
    parser.add_argument("--actual-timing-duplication-levels", nargs="*", type=int, default=[10])
    parser.add_argument("--actual-timing-active-particle-cap", type=int, default=100)
    parser.add_argument("--actual-timing-seed", type=int, default=13)
    parser.add_argument("--only-actual-timing-duplication", type=int, default=None)

    # Real experiments.
    parser.add_argument("--include-real-all", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-real-non", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--real-emitter-module", default="evaluation.real_emitter")
    parser.add_argument("--real-incident-types", default="evaluation/filtered_incidents.txt")
    parser.add_argument("--real-results-folder", default="real_all_incidents")
    parser.add_argument("--real-non-results-folder", default="real_non_incidents")
    parser.add_argument("--real-temp-root", default=None, help="Override configured extracted real-data root.")
    parser.add_argument("--real-raw-root", default=None, help="Override configured source/date TAR archive root.")
    parser.add_argument("--real-dates", nargs="*", default=None)
    parser.add_argument("--real-dates-file", default=None)
    parser.add_argument("--real-non-dates", nargs="*", default=None)
    parser.add_argument("--real-non-dates-file", default=None)
    parser.add_argument("--real-data-sources", nargs="*", default=None)
    parser.add_argument("--real-anomaly-preprocessing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--real-emit-cached-positive-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--real-observation-cache-root", default=None, help="Override configured real observation-cache root.")
    parser.add_argument("--real-cached-positive-missing-policy", choices=["keep_all", "drop_all"], default="keep_all")
    parser.add_argument(
        "--real-lazy-negative-retrieval",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Enable lazy retrieval of cached clean-negative observations for real "
            "IncidentLens/generic-propagation runs. This preserves propagation "
            "negative evidence while positive-only replay skips non-anomalous stream traffic."
        ),
    )
    parser.add_argument(
        "--real-lazy-negative-max-per-hour",
        type=int,
        default=3,
        help="Maximum lazy-retrieved negative observations per hour used for retrospective scoring.",
    )
    parser.add_argument("--real-auto-prepare-temp-data", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--real-clear-temp-before-prepare",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Allow real_emitter/data_process to clear evaluation/temp before extraction. Default: false.",
    )
    parser.add_argument(
        "--real-skip-existing-temp-data",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip real source/date temp folders that already contain extracted data. Default: true.",
    )
    parser.add_argument("--real-replay-interval-seconds", type=float, default=0.0)
    parser.add_argument(
        "--real-exclude-baselines",
        nargs="*",
        default=[],
        help=(
            "Baselines excluded only from main real experiments. Defaults to no extra "
            "exclusions because DEFAULT_REAL_BASELINES already omits synthetic-only "
            "generic_propagation/generic_all."
        ),
    )
    parser.add_argument(
        "--include-real-background",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "When --exp-type real/all is used, also include the real-data "
            "satscan_background and hawkes_event_detector baselines in the main real suite. "
            "Use --exp-type real_background to run only those new real-data baselines."
        ),
    )
    parser.add_argument(
        "--real-background-baselines",
        nargs="*",
        default=DEFAULT_REAL_BACKGROUND_BASELINES,
        help="Background-aware baselines used by --exp-type real_background.",
    )
    parser.add_argument(
        "--real-background-incident-types",
        default=None,
        help="Optional incident-type file for real_background; defaults to --real-incident-types.",
    )
    parser.add_argument(
        "--real-background-results-folder",
        default=None,
        help="Optional results folder for real_background; defaults to --real-results-folder.",
    )
    parser.add_argument(
        "--real-background-anomaly-preprocessing",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Enable anomaly preprocessing for real-data Hawkes/SaTScan background baselines. "
            "Default: false, because cached-positive real-data replay already avoids the heavy negative stream."
        ),
    )
    parser.add_argument(
        "--real-background-lazy-negative-retrieval",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Enable lazy negative retrieval for real-data Hawkes/SaTScan background baselines. "
            "Default: false, because these baselines do not use IncidentLens retrospective propagation scoring."
        ),
    )
    parser.add_argument(
        "--real-background-source-constraints",
        default="generic_incident_source_constraints.json",
        help=(
            "Optional source-constraints JSON used to auto-generate --baseline-config "
            "for satscan_background and hawkes_event_detector real-data runs. If the "
            "file is not found, those baselines run with constructor defaults."
        ),
    )
    parser.add_argument(
        "--real-background-baseline-config",
        default=None,
        help=(
            "Optional explicit JSON config passed to both satscan_background and "
            "hawkes_event_detector during real-data runs. Overrides "
            "--real-background-source-constraints."
        ),
    )
    parser.add_argument(
        "--real-generated-baseline-config-dir",
        default="evaluation/experiment_logs/generated_baseline_configs",
        help="Directory for generated one-baseline config wrappers for real background baselines.",
    )

    args = parser.parse_args()
    python_exe, child_env_updates = resolve_python_executable(explicit_python=args.python, venv=args.venv)

    experiments = build_experiments(args)
    if not experiments:
        print("No experiments selected.", file=sys.stderr)
        return 1

    print(f"Using Python executable: {python_exe}")
    if args.venv:
        print(f"Using virtual environment: {Path(args.venv).expanduser().resolve()}")
    print(f"Experiment type: {args.exp_type}")
    print("Planned experiments:")
    for idx, exp in enumerate(experiments, start=1):
        date_part = f", dates={exp.dates[0]}..{exp.dates[-1]} ({len(exp.dates)}d)" if exp.dates else ""
        print(f"  {idx:03d}/{len(experiments):03d} - {exp.name}: baseline={exp.baseline}, results_folder={exp.results_folder}{date_part}, pipeline_args={' '.join(exp.pipeline_args)}, emitter_args={' '.join(exp.emitter_args[:8])}{' ...' if len(exp.emitter_args) > 8 else ''}")

    completed: list[ExperimentRunResult] = []
    run_started_at = time.time()
    progress_path = Path(args.progress_json)
    write_progress_file(
        progress_path=progress_path,
        experiments=experiments,
        completed=completed,
        current=None,
        run_started_at=run_started_at,
    )
    print(f"Progress JSON: {progress_path}")

    iterator = enumerate(experiments, start=1)
    progress_bar = None
    if tqdm is not None and not args.no_progress_bar:
        progress_bar = tqdm(total=len(experiments), unit="exp", desc=f"{args.exp_type} experiments")
        iterator = enumerate(experiments, start=1)
    elif not args.no_progress_bar:
        print("tqdm is not installed; using plain progress logs. Install with `pip install tqdm` for a progress bar.")

    try:
        for idx, exp in iterator:
            if progress_bar is not None:
                progress_bar.set_description(f"{idx}/{len(experiments)} {exp.name[:40]}")
                progress_bar.set_postfix_str("running")
            write_progress_file(
                progress_path=progress_path,
                experiments=experiments,
                completed=completed,
                current=exp,
                run_started_at=run_started_at,
            )

            result = run_experiment(
                exp,
                python_exe=python_exe,
                socket_host=args.socket_host,
                socket_port=args.socket_port,
                startup_timeout_s=args.startup_timeout_s,
                log_root=Path(args.log_root),
                dry_run=args.dry_run,
                child_env_updates=child_env_updates,
                skip_completed=args.skip_completed,
                quiet_child_logs=args.quiet_child_logs,
            )
            completed.append(result)

            if result.status == "failed":
                write_progress_file(
                    progress_path=progress_path,
                    experiments=experiments,
                    completed=completed,
                    current=None,
                    run_started_at=run_started_at,
                )
                print(f"\nExperiment failed: {result.name}", file=sys.stderr)
                print(result.error or "<no error details>", file=sys.stderr)
                return 1

            if progress_bar is not None:
                progress_bar.update(1)
                progress_bar.set_postfix_str(f"last={result.status}, {format_duration(result.elapsed_seconds)}")
            print_progress_summary(completed, len(experiments), run_started_at)
            write_progress_file(
                progress_path=progress_path,
                experiments=experiments,
                completed=completed,
                current=None,
                run_started_at=run_started_at,
            )
    finally:
        if progress_bar is not None:
            progress_bar.close()

    total_elapsed = time.time() - run_started_at
    print(f"\nAll selected experiments completed in {format_duration(total_elapsed)}.", flush=True)
    print(f"Final progress JSON: {progress_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
