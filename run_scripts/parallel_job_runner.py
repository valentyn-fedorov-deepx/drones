import argparse
import os
import shlex
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple, Optional


def parse_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Launch multiple runs of a script/module in parallel from a params file.")
    p.add_argument("--params-file", type=Path, required=True,
                   help="Path to a .txt file. Each non-empty, non-comment line is one job's CLI flags.")
    # Choose either a filesystem script OR a Python module
    m = p.add_mutually_exclusive_group(required=True)
    m.add_argument("--script-path", type=Path, help="Path to a single-run script file (e.g., make_video.py).")
    m.add_argument("--module", type=str, help="Dotted module path (e.g., run_scripts.gen_video) to execute with -m.")
    p.add_argument("--max-procs", type=int, default=os.cpu_count() or 2,
                   help="Maximum number of parallel processes (default: CPU count).")
    p.add_argument("--logs-dir", type=Path, default=Path("logs"),
                   help="Directory for per-job logs.")
    p.add_argument("--pythonpath", action="append", default=[],
                   help="Path to prepend to PYTHONPATH (repeatable). Useful so the module is importable.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the commands that would run, but don't run them.")
    return p.parse_args()


def read_jobs(params_file: Path) -> List[Tuple[int, List[str]]]:
    jobs: List[Tuple[int, List[str]]] = []
    with params_file.open("r", encoding="utf-8") as f:
        for i, raw in enumerate(f, 1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            args_list = shlex.split(line)
            if args_list:
                jobs.append((i, args_list))
    return jobs


def job_label(idx: int, arglist: List[str]) -> str:
    try:
        i = arglist.index("--data-path")
        return Path(arglist[i + 1]).stem or f"job{idx}"
    except (ValueError, IndexError):
        return f"job{idx}"


def build_env(extra_pythonpaths: List[str]) -> dict:
    env = os.environ.copy()
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("OPENCV_LOG_LEVEL", "SILENT")
    if extra_pythonpaths:
        existing = env.get("PYTHONPATH")
        joined = os.pathsep.join(extra_pythonpaths + ([existing] if existing else []))
        env["PYTHONPATH"] = joined
    return env


def run_job(
    idx: int,
    arglist: List[str],
    *,
    module: Optional[str],
    script_path: Optional[Path],
    logs_dir: Path,
    env: dict,
) -> Tuple[str, int, Path]:
    label = job_label(idx, arglist)
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / f"{idx:03d}_{label}.log"

    if module:
        cmd = [sys.executable, "-m", module] + arglist
    else:
        cmd = [sys.executable, str(script_path)] + arglist  # type: ignore[arg-type]

    with log_path.open("w", buffering=1, encoding="utf-8") as lf:
        lf.write("COMMAND:\n  " + " ".join(shlex.quote(c) for c in cmd) + "\n\n")
        lf.flush()
        rc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env).returncode

    return label, rc, log_path


def main():
    args = parse_cli()
    if args.script_path and not args.script_path.exists():
        print(f"ERROR: --script-path not found: {args.script_path}", file=sys.stderr)
        sys.exit(2)

    jobs = read_jobs(args.params_file)
    if not jobs:
        print(f"No runnable lines found in {args.params_file} (are they commented out?).", file=sys.stderr)
        sys.exit(1)

    env = build_env(args.pythonpath)

    print(f"Discovered {len(jobs)} job(s) from {args.params_file}")
    print(f"Mode: {'module ' + args.module if args.module else 'script ' + str(args.script_path)}")
    if args.pythonpath:
        print("PYTHONPATH prepend:", os.pathsep.join(args.pythonpath))
    print(f"Max parallel processes: {args.max_procs}")
    print(f"Logs: {args.logs_dir.resolve()}\n")

    if args.dry_run:
        for idx, arglist in jobs:
            label = job_label(idx, arglist)
            if args.module:
                cmd = [sys.executable, "-m", args.module] + arglist
            else:
                cmd = [sys.executable, str(args.script_path)] + arglist
            print(f"[DRY-RUN] {idx:03d} {label}: " + " ".join(shlex.quote(c) for c in cmd))
        return

    try:
        with ProcessPoolExecutor(max_workers=args.max_procs) as ex:
            futures = [
                ex.submit(
                    run_job, idx, arglist,
                    module=args.module, script_path=args.script_path,
                    logs_dir=args.logs_dir, env=env
                )
                for idx, arglist in jobs
            ]
            ok = fail = 0
            for fut in as_completed(futures):
                label, rc, log_path = fut.result()
                if rc == 0:
                    ok += 1
                    print(f"[OK]   {label}  (log: {log_path})")
                else:
                    fail += 1
                    print(f"[FAIL] {label}  (exit {rc})  log: {log_path}")
            print(f"\nDone. {ok} succeeded, {fail} failed.")
            if fail:
                sys.exit(1)
    except KeyboardInterrupt:
        print("\nInterrupted. Some child processes may still be running; check the logs.", file=sys.stderr)
        sys.exit(130)


if __name__ == "__main__":
    main()
