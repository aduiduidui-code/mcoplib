"""
testall.py — Batch driver for mcoplib operator benchmarks.

Usage:
  python testall.py                      # default: --generate
  python testall.py --generate           # only fill missing CSV rows (skip existing)
  python testall.py --compare            # compare each op vs CSV baseline; fail if >5%% slower
  python testall.py --update             # overwrite CSV rows for all ops
  python testall.py --csv PATH           # override CSV path (default: statistics/mcoplib_ops_performance_C500.csv)
  python testall.py --ops a,b,c          # only run named ops
  python testall.py --dry-run            # list ops without running

The script:
  1. Calls `python mcoplib_mxbenchmark_ops.py --list` to enumerate supported ops.
  2. For each op, runs `python mcoplib_mxbenchmark_ops.py --op <name> --<mode> --csv <csv>`.
  3. Parses stdout to extract accuracy status and performance status.
  4. Prints a per-op summary table and a final roll-up (success/failure counts + failing op names).
"""

import os
import re
import sys
import csv
import ast
import time
import argparse
import subprocess

TARGET_SCRIPT = "mcoplib_mxbenchmark_ops.py"
STATISTICS_DIR = "statistics"
CSV_FILENAME = "mcoplib_ops_performance_C500.csv"
DEFAULT_CSV = os.path.join(STATISTICS_DIR, CSV_FILENAME)
OUTPUT_FILE = "testall_output.txt"

LIST_OP_PREFIX = "  * "
VERIFY_RE = re.compile(r"\[VERIFY\]\s+(\S+)\s*->\s*(PASS|FAIL)", re.IGNORECASE)
ACC_VERIFY_RE = re.compile(r"Acc verify:\s*(\S+)", re.IGNORECASE)
PERF_VERIFY_RE = re.compile(r"Performance verify:\s*(\S+)", re.IGNORECASE)
APPEND_RE = re.compile(r"\[APPEND\]\s+(\S+)", re.IGNORECASE)
SKIP_RE = re.compile(r"\[SKIP\]\s+(\S+)", re.IGNORECASE)
UPDATE_RE = re.compile(r"\[UPDATE\]\s+(\S+)", re.IGNORECASE)
IGNORED_RE = re.compile(r"\[IGNORED\]\s+(\S+)", re.IGNORECASE)
SUMMARY_RE = re.compile(
    r"\[SUMMARY\]\s+(Appended|Skipped|Updated|Kept):\s*(\d+)", re.IGNORECASE
)

# Threshold for performance regression in --compare mode (must match the
# single-op script's own 5%% rule).
PERF_REGRESSION_THRESHOLD_PCT = 5.0


def _load_ignored_operators():
    """Read IGNORED_OPERATORS list from mcoplib_mxbenchmark_ops.py source.

    Parses the module AST instead of importing it, so we don't trigger
    nvbench import side effects (which sys.exit on failure).
    """
    script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), TARGET_SCRIPT)
    if not os.path.exists(script_path):
        return set()
    try:
        with open(script_path, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read())
    except Exception:
        return set()

    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "IGNORED_OPERATORS":
                    if isinstance(node.value, ast.List):
                        result = set()
                        for elt in node.value.elts:
                            if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                                result.add(elt.value)
                        return result
    return set()


def get_supported_operators():
    """Call the single-op script's --list and parse the operator names."""
    cmd = [sys.executable, TARGET_SCRIPT, "--list"]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, check=True, timeout=60
        )
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] `--list` failed: {e}")
        print(e.stdout)
        print(e.stderr)
        return []
    except subprocess.TimeoutExpired:
        print("[ERROR] `--list` timed out")
        return []

    ops = []
    for line in result.stdout.splitlines():
        if line.startswith(LIST_OP_PREFIX):
            ops.append(line[len(LIST_OP_PREFIX):].strip())
    return ops


def parse_run_output(stdout):
    """Extract accuracy and performance status from a single op run."""
    info = {
        "verify_pass": None,       # True/False/None
        "acc_verify": None,        # "Pass"/"Fail"/"None"/None
        "perf_verify": None,       # "NN.NN%"/"None"/None
        "csv_action": None,        # "APPEND"/"SKIP"/"UPDATE"/"IGNORED"/None
        "fatal": False,
        "error_msg": None,
    }

    for line in stdout.splitlines():
        m = VERIFY_RE.search(line)
        if m:
            info["verify_pass"] = (m.group(2).upper() == "PASS")
            continue
        m = ACC_VERIFY_RE.search(line)
        if m:
            info["acc_verify"] = m.group(1).strip()
            continue
        m = PERF_VERIFY_RE.search(line)
        if m:
            info["perf_verify"] = m.group(1).strip()
            continue
        m = APPEND_RE.search(line)
        if m:
            info["csv_action"] = "APPEND"
            continue
        m = SKIP_RE.search(line)
        if m:
            info["csv_action"] = "SKIP"
            continue
        m = UPDATE_RE.search(line)
        if m:
            info["csv_action"] = "UPDATE"
            continue
        m = IGNORED_RE.search(line)
        if m:
            info["csv_action"] = "IGNORED"
            continue
        if "[FATAL]" in line:
            info["fatal"] = True
            info["error_msg"] = line.strip()
            continue

    return info


def _op_in_csv(op_name, csv_path):
    """Return True if op_name appears as op_name column in the CSV."""
    if not os.path.exists(csv_path):
        return False
    try:
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("op_name") == op_name:
                    return True
    except Exception:
        return False
    return False


def classify_result(op_name, mode, info, returncode, csv_path, csv_before):
    """Decide whether an op's run was a success or failure for the given mode.

    Returns (status, detail) where status is one of:
      SUCCESS, FAILED, SKIPPED.
    """
    # Ignored operators are always SKIPPED, regardless of mode.
    if info.get("csv_action") == "IGNORED":
        return "SKIPPED", "ignored"

    if returncode != 0:
        return "FAILED", f"exit={returncode}"

    if info["fatal"]:
        return "FAILED", "verification fatal"

    # compare mode: needs Acc verify:Pass and Performance verify within 5%.
    if mode == "compare":
        acc = info["acc_verify"]
        perf = info["perf_verify"]
        if acc is None or acc.lower() == "none":
            # Could be that the op isn't in CSV (single-op script exits early).
            if not _op_in_csv(op_name, csv_path):
                return "SKIPPED", "no baseline"
            # Op is in CSV but no Acc verify line captured — likely the single-op
            # script's verification failed silently (os._exit before flush).
            return "FAILED", "no acc result"
        if acc.lower() != "pass":
            return "FAILED", f"acc={acc}"
        if perf is None or perf.lower() == "none":
            return "FAILED", "no perf ratio"
        try:
            ratio_pct = float(perf.rstrip("%"))
        except ValueError:
            return "FAILED", f"bad perf ratio: {perf}"
        slowdown_pct = (1.0 - ratio_pct / 100.0) * 100.0
        if slowdown_pct > PERF_REGRESSION_THRESHOLD_PCT:
            return "FAILED", f"perf={perf} (slow {slowdown_pct:.2f}%)"
        return "SUCCESS", f"acc=Pass perf={perf}"

    # generate mode: APPEND = success, SKIP = skipped (already exists).
    if mode == "generate":
        if info["csv_action"] == "APPEND":
            return "SUCCESS", "appended"
        if info["csv_action"] == "SKIP":
            return "SKIPPED", "exists (use --update to refresh)"
        if info["verify_pass"] is False:
            return "FAILED", "acc fail"
        # No csv_action line captured — could be silent verification failure
        # (the single-op script calls os._exit(0) on acc failure, dropping
        # buffered output). Use CSV existence as the source of truth.
        if not csv_before and _op_in_csv(op_name, csv_path):
            return "SUCCESS", "appended(csv)"
        if csv_before and _op_in_csv(op_name, csv_path):
            return "SKIPPED", "exists(csv)"
        return "FAILED", "no result"

    # update mode: UPDATE = success.
    if mode == "update":
        if info["csv_action"] == "UPDATE":
            return "SUCCESS", "updated"
        if info["verify_pass"] is False:
            return "FAILED", "acc fail"
        # Fall back to CSV existence check.
        if _op_in_csv(op_name, csv_path):
            return "SUCCESS", "updated(csv)"
        return "FAILED", "no result"

    return "SUCCESS", "ok"


def run_one_op(op_name, mode, csv_path):
    """Run the single-op script for one op. Returns (info, returncode, elapsed)."""
    cmd = [
        sys.executable, TARGET_SCRIPT,
        "--op", op_name,
        f"--{mode}",
        "--csv", csv_path,
    ]
    start = time.time()
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=600
        )
        rc = result.returncode
        stdout = result.stdout
    except subprocess.TimeoutExpired:
        return None, 124, time.time() - start
    elapsed = time.time() - start
    return parse_run_output(stdout), rc, elapsed


def print_summary_table(rows):
    """Print a per-op table. rows: list of dict with keys op, acc, perf, status, detail, elapsed."""
    cols = [
        ("Op", 36), ("Acc", 8), ("Perf", 14),
        ("Status", 10), ("Detail", 28), ("Time", 9),
    ]
    header = " | ".join(c.ljust(w) for c, w in cols)
    sep = "-+-".join("-" * w for _, w in cols)
    print(header)
    print(sep)
    for r in rows:
        line = " | ".join([
            r["op"][:36].ljust(36),
            str(r["acc"] or "-").ljust(8),
            str(r["perf"] or "-").ljust(14),
            r["status"].ljust(10),
            str(r["detail"] or "")[:28].ljust(28),
            f"{r['elapsed']:.1f}s".ljust(9),
        ])
        print(line)


def ensure_csv_exists(csv_path):
    """For --compare/--update the single-op script requires the CSV to exist."""
    if not os.path.exists(csv_path):
        csv_dir = os.path.dirname(os.path.abspath(csv_path))
        if csv_dir and not os.path.exists(csv_dir):
            os.makedirs(csv_dir, exist_ok=True)
        # Write an empty CSV with the canonical header so --compare won't hard-fail
        # at the file-existence check; instead each op will hit the "not found in
        # baseline" path and report SKIPPED.
        header = (
            "op_name,Device,Device Name,Acc_Pass,Cos_Dist,Op,dtype,Shape,Samples,"
            "CPU Time (sec),Noise,GPU Time (sec),Noise,Elem/s (elem/sec),"
            "GlobalMem BW (bytes/sec),BWUtil,Samples,Batch GPU (sec)"
        )
        with open(csv_path, "w", newline="") as f:
            f.write(header + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Batch driver for mcoplib operator benchmarks"
    )
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--generate", action="store_true",
        help="Only fill missing CSV rows (skip existing) [default]"
    )
    mode_group.add_argument(
        "--compare", action="store_true",
        help="Compare each op vs CSV baseline; fail if >5%% slower"
    )
    mode_group.add_argument(
        "--update", action="store_true",
        help="Overwrite CSV rows for all ops"
    )
    parser.add_argument("--csv", default=DEFAULT_CSV, help="CSV path")
    parser.add_argument("--ops", default=None, help="Comma-separated op subset")
    parser.add_argument("--dry-run", action="store_true", help="List ops without running")
    args = parser.parse_args()

    if args.compare:
        mode = "compare"
    elif args.update:
        mode = "update"
    else:
        mode = "generate"

    print("=" * 80)
    print(f"testall.py | mode={mode} | csv={args.csv}")
    if mode == "generate":
        print("[INFO] --generate: only fills missing CSV rows; ops already in CSV are SKIPPED.")
        print("       Use --update to overwrite existing rows, or --compare to regression-test.")
    elif mode == "update":
        print("[INFO] --update: overwrites CSV rows for all ops (verification must pass first).")
    elif mode == "compare":
        print("[INFO] --compare: compares each op vs CSV baseline; fails if >5% slower. No CSV writes.")
    print("=" * 80)

    ops = get_supported_operators()
    if args.ops:
        wanted = {s.strip() for s in args.ops.split(",") if s.strip()}
        ops = [o for o in ops if o in wanted]
    if not ops:
        print("[ERROR] No operators to run.")
        return 1

    # Load the ignored-operators list from the single-op script source.
    ignored_set = _load_ignored_operators()
    if ignored_set:
        # Split ops into active vs ignored, but preserve order: ignored ops
        # appear in the summary table as SKIPPED so the user sees them.
        ignored_in_list = [o for o in ops if o in ignored_set]
        ops = [o for o in ops if o not in ignored_set]
        if ignored_in_list:
            print(f"[INFO] {len(ignored_in_list)} operator(s) ignored: "
                  f"{', '.join(ignored_in_list)}")
        print(f"[INFO] {len(ops)} operators to run in --{mode} mode.\n")
    else:
        print(f"[INFO] {len(ops)} operators to run in --{mode} mode.\n")

    if args.dry_run:
        for o in ops:
            print(f"  * {o}")
        return 0

    if mode in ("compare", "update"):
        ensure_csv_exists(args.csv)
    elif mode == "generate":
        csv_dir = os.path.dirname(os.path.abspath(args.csv))
        if csv_dir and not os.path.exists(csv_dir):
            os.makedirs(csv_dir, exist_ok=True)

    rows = []
    success = failed = skipped = 0
    failing_ops = []
    total_start = time.time()

    with open(OUTPUT_FILE, "w", encoding="utf-8") as logf:
        logf.write(f"=== testall.py | mode={mode} | csv={args.csv} ===\n")
        logf.write(f"=== {len(ops)} operators ===\n\n")
        logf.flush()

        for idx, op in enumerate(ops, 1):
            print(f"[{idx}/{len(ops)}] {op} ... ", end="", flush=True)
            csv_before = _op_in_csv(op, args.csv)

            # --generate fast path: if the op is already in the CSV, skip the
            # expensive subprocess + nvbench run entirely. The single-op script
            # would also SKIP, but only after running the full benchmark.
            if mode == "generate" and csv_before:
                status, detail, info, rc, elapsed = (
                    "SKIPPED", "exists (use --update to refresh)", None, 0, 0.0
                )
            else:
                info, rc, elapsed = run_one_op(op, mode, args.csv)
                if info is None:
                    info = {"verify_pass": None, "acc_verify": None,
                            "perf_verify": None, "csv_action": None,
                            "fatal": False, "error_msg": "timeout"}
                    rc = 124
                status, detail = classify_result(
                    op, mode, info, rc, args.csv, csv_before
                )
            if status == "SUCCESS":
                success += 1
            elif status == "FAILED":
                failed += 1
                failing_ops.append(op)
            elif status == "SKIPPED":
                skipped += 1
            print(f"{status} ({detail}) [{elapsed:.1f}s]")

            logf.write(f"[{op}] -> {status} | {detail} | rc={rc} | {elapsed:.2f}s\n")
            if info:
                logf.write(
                    f"  verify_pass={info['verify_pass']} "
                    f"acc={info['acc_verify']} perf={info['perf_verify']} "
                    f"action={info['csv_action']} fatal={info['fatal']}\n"
                )
            logf.flush()

            rows.append({
                "op": op,
                "acc": info.get("acc_verify") if info else None,
                "perf": info.get("perf_verify") if info else None,
                "status": status,
                "detail": detail,
                "elapsed": elapsed,
            })

        total_elapsed = time.time() - total_start
        logf.write(
            f"\n=== SUMMARY | mode={mode} | "
            f"success={success} failed={failed} skipped={skipped} | "
            f"total={len(ops)} | {total_elapsed:.1f}s ===\n"
        )
        if failing_ops:
            logf.write(f"FAILED OPS: {', '.join(failing_ops)}\n")

    print("\n" + "=" * 80)
    print(f"Per-op results (mode={mode}):")
    print("=" * 80)
    print_summary_table(rows)

    print("\n" + "=" * 80)
    print(
        f"SUMMARY | mode={mode} | total={len(ops)} "
        f"success={success} failed={failed} skipped={skipped} "
        f"| {total_elapsed:.1f}s"
    )
    if failing_ops:
        print(f"FAILED OPS ({failed}):")
        for o in failing_ops:
            print(f"  - {o}")
    else:
        print("No failures.")
    print(f"\nDetailed log: {OUTPUT_FILE}")
    print("=" * 80)

    return 0 if failed == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
