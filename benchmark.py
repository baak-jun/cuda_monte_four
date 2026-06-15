import argparse
import json
import statistics
import subprocess
import time
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
BIN_DIR = PROJECT_DIR / "build" / "Release"

MC_CUDA_BIN = BIN_DIR / "monte_carlo_cuda.exe"
MC_CPU_BIN = BIN_DIR / "monte_carlo_cpu.exe"
CPU_SOLVER_BIN = BIN_DIR / "perfect_solver_cpu.exe"
HYBRID_BIN = BIN_DIR / "perfect_frontier_cuda.exe"


def parse_args():
    parser = argparse.ArgumentParser(description="Connect Four CPU/CUDA benchmark suite")
    parser.add_argument("--quick", action="store_true", help="run a small smoke benchmark")
    parser.add_argument("--output", type=Path, default=PROJECT_DIR / "benchmark_results.json")
    parser.add_argument("--mc-simulations", type=int, default=10_000_000)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument(
        "--frontier-only",
        action="store_true",
        help="run only the hybrid frontier-depth experiment",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=120,
        help="timeout for each hybrid frontier run",
    )
    parser.add_argument(
        "--single-run",
        action="store_true",
        help="skip warm-up and measure each frontier depth once",
    )
    parser.add_argument(
        "--frontier-depths",
        type=int,
        nargs="+",
        help="frontier depths to test (default: 2 through 7)",
    )
    return parser.parse_args()


def parse_metrics(output):
    metrics = {}
    for raw_line in output.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if "=" in line:
            key, value = line.split("=", 1)
            metrics[key.strip()] = value.strip()
        elif " " in line:
            key, value = line.split(None, 1)
            metrics[key] = value.strip()
    return metrics


def run_once(command, timeout):
    start = time.perf_counter()
    try:
        completed = subprocess.run(
            [str(part) for part in command],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return {
            "ok": False,
            "timeout": True,
            "wall_sec": timeout,
            "stdout": "",
            "stderr": "TIMEOUT",
            "metrics": {},
        }

    wall_sec = time.perf_counter() - start
    return {
        "ok": completed.returncode == 0,
        "timeout": False,
        "returncode": completed.returncode,
        "wall_sec": wall_sec,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "metrics": parse_metrics(completed.stdout),
    }


def summarize(values):
    if not values:
        return None
    return {
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
        "samples": len(values),
    }


def run_repeated(command, repeats, timeout, warmups=1):
    for _ in range(warmups):
        warmup = run_once(command, timeout)
        if not warmup["ok"]:
            return {"ok": False, "phase": "warmup", "error": warmup}

    runs = []
    for _ in range(repeats):
        result = run_once(command, timeout)
        if not result["ok"]:
            return {"ok": False, "phase": "measurement", "error": result}
        runs.append(result)

    kernel_times = []
    for run in runs:
        if "kernel_ms" in run["metrics"]:
            kernel_times.append(float(run["metrics"]["kernel_ms"]))

    return {
        "ok": True,
        "end_to_end_sec": summarize([run["wall_sec"] for run in runs]),
        "kernel_ms": summarize(kernel_times),
        "metrics": runs[-1]["metrics"],
    }


def integer_counts(metrics):
    keys = ("black_wins", "white_wins", "draws")
    if not all(key in metrics for key in keys):
        return None
    return {key: int(metrics[key]) for key in keys}


def require_binaries():
    missing = [
        path
        for path in (MC_CUDA_BIN, MC_CPU_BIN, CPU_SOLVER_BIN, HYBRID_BIN)
        if not path.exists()
    ]
    if missing:
        names = ", ".join(str(path) for path in missing)
        raise SystemExit(f"Missing executables: {names}. Build Release first.")


def benchmark_monte_carlo(results, simulations, seed, quick):
    repeats = 1 if quick else 5
    cpu_repeats = 1 if quick else 3
    thread_counts = [64, 128, 256] if quick else [64, 128, 256, 512, 1024]

    print("\n[1/3] Monte Carlo CPU baseline and CUDA block-size tuning")
    cpu_command = [MC_CPU_BIN, "--simulations", simulations, "--seed", seed]
    cpu_result = run_repeated(cpu_command, cpu_repeats, timeout=180, warmups=1)
    results["mc_cpu_baseline"] = cpu_result
    cpu_counts = integer_counts(cpu_result.get("metrics", {})) if cpu_result["ok"] else None

    if cpu_result["ok"]:
        median = cpu_result["end_to_end_sec"]["median"]
        print(f"  CPU baseline: {median:.4f}s median")
    else:
        print("  CPU baseline failed")

    for threads in thread_counts:
        command = [
            MC_CUDA_BIN,
            "--simulations",
            simulations,
            "--threads",
            threads,
            "--seed",
            seed,
        ]
        measured = run_repeated(command, repeats, timeout=120, warmups=1)
        entry = {"block_threads": threads, **measured}
        if measured["ok"]:
            gpu_counts = integer_counts(measured["metrics"])
            entry["counts"] = gpu_counts
            entry["matches_cpu_counts"] = cpu_counts is not None and gpu_counts == cpu_counts
            wall = measured["end_to_end_sec"]["median"]
            kernel = measured["kernel_ms"]["median"] if measured["kernel_ms"] else float("nan")
            print(
                f"  block={threads:4d}: wall={wall:.4f}s, kernel={kernel:.3f}ms, "
                f"CPU match={entry['matches_cpu_counts']}"
            )
        else:
            print(f"  block={threads:4d}: failed")
        results["mc_block_size_tuning"].append(entry)


def benchmark_cpu_vs_hybrid(results, quick):
    repeats = 1 if quick else 3
    depths = [10, 12] if quick else list(range(10, 17))

    print("\n[2/3] CPU solver vs hybrid CUDA solver")
    for depth in depths:
        cpu_command = [CPU_SOLVER_BIN, "--max-depth", depth, "--reserve", 2_000_000]
        hybrid_command = [
            HYBRID_BIN,
            "--max-depth",
            depth,
            "--frontier-depth",
            5,
            "--threads",
            256,
        ]
        cpu = run_repeated(cpu_command, repeats, timeout=120, warmups=1)
        hybrid = run_repeated(hybrid_command, repeats, timeout=120, warmups=1)
        matches = False
        if cpu["ok"] and hybrid["ok"]:
            matches = (
                cpu["metrics"].get("score") == hybrid["metrics"].get("score")
                and cpu["metrics"].get("best_move") == hybrid["metrics"].get("best_move")
            )

        results["cpu_vs_hybrid"].append(
            {
                "depth": depth,
                "cpu": cpu,
                "hybrid": hybrid,
                "score_and_best_move_match": matches,
            }
        )
        print(f"  depth={depth:2d}: result match={matches}")


def benchmark_frontier_depth(
    results,
    quick,
    timeout_seconds,
    single_run,
    selected_depths,
):
    repeats = 1 if quick or single_run else 3
    warmups = 0 if single_run else 1
    max_depth = 12 if quick else 16
    frontier_depths = selected_depths or ([2, 4] if quick else [2, 3, 4, 5, 6, 7])

    print("\n[3/3] Hybrid frontier-depth tuning")
    for frontier_depth in frontier_depths:
        command = [
            HYBRID_BIN,
            "--max-depth",
            max_depth,
            "--frontier-depth",
            frontier_depth,
            "--threads",
            256,
        ]
        measured = run_repeated(
            command,
            repeats,
            timeout=timeout_seconds,
            warmups=warmups,
        )
        results["frontier_depth_tuning"].append(
            {"frontier_depth": frontier_depth, "max_depth": max_depth, **measured}
        )
        if measured["ok"]:
            wall = measured["end_to_end_sec"]["median"]
            kernel = measured["kernel_ms"]["median"] if measured["kernel_ms"] else float("nan")
            print(f"  frontier={frontier_depth}: wall={wall:.4f}s, kernel={kernel:.3f}ms")
        else:
            print(f"  frontier={frontier_depth}: failed")


def main():
    args = parse_args()
    require_binaries()
    simulations = 100_000 if args.quick else args.mc_simulations
    results = {
        "metadata": {
            "benchmark_kind": "block-size tuning, not thread scaling",
            "mc_simulations": simulations,
            "seed": args.seed,
            "quick": args.quick,
            "timing": {
                "end_to_end_sec": "subprocess wall time including CUDA initialization",
                "kernel_ms": "CUDA event time reported by the executable",
            },
        },
        "mc_cpu_baseline": {},
        "mc_block_size_tuning": [],
        "cpu_vs_hybrid": [],
        "frontier_depth_tuning": [],
    }

    print("=== Connect Four Benchmark ===")
    if not args.frontier_only:
        benchmark_monte_carlo(results, simulations, args.seed, args.quick)
        benchmark_cpu_vs_hybrid(results, args.quick)
    benchmark_frontier_depth(
        results,
        args.quick,
        args.timeout_seconds,
        args.single_run,
        args.frontier_depths,
    )

    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nRaw results saved to {args.output}")


if __name__ == "__main__":
    main()
