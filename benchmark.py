import subprocess
import time
import json
import os
import sys

# 디렉터리 경로 설정
project_dir = r"C:\Users\Nabi\Desktop\26-1\Parallel Programming\project"
bin_dir = os.path.join(project_dir, "build", "Release")

mc_bin = os.path.join(bin_dir, "monte_carlo_cuda.exe")
cpu_bin = os.path.join(bin_dir, "perfect_solver_cpu.exe")
hybrid_bin = os.path.join(bin_dir, "perfect_frontier_cuda.exe")

# 실행 파일 존재 여부 확인
for path in [mc_bin, cpu_bin, hybrid_bin]:
    if not os.path.exists(path):
        print(f"Error: Executable not found at {path}")
        print("Please build the project first.")
        sys.exit(1)

def run_benchmark(cmd, timeout=30):
    start = time.perf_counter()
    try:
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=timeout)
        duration = time.perf_counter() - start
        return duration, res.stdout, res.stderr, False
    except subprocess.TimeoutExpired:
        return timeout, "", "TIMEOUT", True

print("=== Connect Four Solver Benchmark Start ===")

results = {
    "mc_thread_scaling": [],
    "cpu_vs_hybrid": [],
    "frontier_depth_tuning": []
}

# 1. CUDA Monte Carlo Thread Scaling
# Simulations: 10,000,000
mc_simulations = 10000000
thread_counts = [64, 128, 256, 512, 1024]
print("\n[1/3] Running CUDA Monte Carlo Thread Scaling...")
for threads in thread_counts:
    cmd = [mc_bin, "--simulations", str(mc_simulations), "--threads", str(threads)]
    print(f"  Threads: {threads}...", end="", flush=True)
    
    # 3번 반복해서 평균 내기
    durations = []
    success = True
    for _ in range(3):
        dur, stdout, stderr, is_timeout = run_benchmark(cmd, timeout=60)
        if is_timeout or "error" in stderr.lower():
            success = False
            break
        durations.append(dur)
        
    if success:
        avg_dur = sum(durations) / len(durations)
        sims_per_sec = mc_simulations / avg_dur
        results["mc_thread_scaling"].append({
            "threads": threads,
            "duration_sec": avg_dur,
            "sims_per_sec": sims_per_sec
        })
        print(f" Done. Avg Time: {avg_dur:.4f}s ({sims_per_sec/1e6:.2f} Msims/sec)")
    else:
        print(" Failed / Timeout.")

# 2. CPU Negamax Alpha-Beta vs Hybrid CUDA Negamax
# Depths: 10 to 16
depths = list(range(10, 17))
print("\n[2/3] Comparing CPU Solver and Hybrid CUDA Solver...")
for depth in depths:
    print(f"  Depth: {depth}...")
    
    # CPU Solver
    cpu_cmd = [cpu_bin, "--max-depth", str(depth), "--reserve", "2000000"]
    print("    Running CPU...", end="", flush=True)
    cpu_dur, cpu_out, cpu_err, cpu_timeout = run_benchmark(cpu_cmd, timeout=30)
    
    cpu_nodes = 0
    if not cpu_timeout and cpu_out:
        for line in cpu_out.splitlines():
            if line.startswith("nodes "):
                cpu_nodes = int(line.split()[1])
        print(f" Done. Time: {cpu_dur:.4f}s, Nodes: {cpu_nodes}")
    else:
        print(" Timeout/Failed")
        cpu_dur = None
        
    # Hybrid GPU Solver
    hybrid_cmd = [hybrid_bin, "--max-depth", str(depth), "--frontier-depth", "5", "--threads", "256"]
    print("    Running Hybrid GPU...", end="", flush=True)
    hyb_dur, hyb_out, hyb_err, hyb_timeout = run_benchmark(hybrid_cmd, timeout=30)
    
    hyb_nodes = 0
    gpu_leaves = 0
    if not hyb_timeout and hyb_out:
        for line in hyb_out.splitlines():
            if line.startswith("tree_nodes "):
                hyb_nodes = int(line.split()[1])
            elif line.startswith("gpu_leaves "):
                gpu_leaves = int(line.split()[1])
        print(f" Done. Time: {hyb_dur:.4f}s, Nodes: {hyb_nodes}, GPU Leaves: {gpu_leaves}")
    else:
        print(" Timeout/Failed")
        hyb_dur = None
        
    results["cpu_vs_hybrid"].append({
        "depth": depth,
        "cpu": {
            "duration_sec": cpu_dur,
            "nodes": cpu_nodes,
            "timeout": cpu_timeout
        },
        "hybrid": {
            "duration_sec": hyb_dur,
            "nodes": hyb_nodes,
            "gpu_leaves": gpu_leaves,
            "timeout": hyb_timeout
        }
    })

# 3. Frontier Depth Tuning (Hybrid CUDA Solver)
# Fixed Max Depth: 16, vary Frontier Depth from 2 to 7
tuning_max_depth = 16
frontier_depths = [2, 3, 4, 5, 6, 7]
print("\n[3/3] Tuning Frontier Depth for Hybrid Solver (Max Depth 16)...")
for f_depth in frontier_depths:
    cmd = [hybrid_bin, "--max-depth", str(tuning_max_depth), "--frontier-depth", str(f_depth), "--threads", "256"]
    print(f"  Frontier Depth: {f_depth}...", end="", flush=True)
    dur, out, err, is_timeout = run_benchmark(cmd, timeout=30)
    
    tree_nodes = 0
    gpu_leaves = 0
    if not is_timeout and out:
        for line in out.splitlines():
            if line.startswith("tree_nodes "):
                tree_nodes = int(line.split()[1])
            elif line.startswith("gpu_leaves "):
                gpu_leaves = int(line.split()[1])
        results["frontier_depth_tuning"].append({
            "frontier_depth": f_depth,
            "duration_sec": dur,
            "tree_nodes": tree_nodes,
            "gpu_leaves": gpu_leaves
        })
        print(f" Done. Time: {dur:.4f}s, Tree Nodes: {tree_nodes}, GPU Leaves: {gpu_leaves}")
    else:
        print(" Timeout/Failed")

# Save raw data to JSON
json_path = os.path.join(project_dir, "benchmark_results.json")
with open(json_path, "w") as f:
    json.dump(results, f, indent=4)
print(f"\nRaw results saved to {json_path}")

# Display Markdown Summary Tables
print("\n=== BENCHMARK SUMMARY ===")

print("\n### 1. CUDA Monte Carlo Thread Scaling")
print("| Threads | Avg Time (s) | Throughput (Msims/sec) | Speedup |")
print("|---------|--------------|------------------------|---------|")
base_time = results["mc_thread_scaling"][0]["duration_sec"] if results["mc_thread_scaling"] else 1.0
for entry in results["mc_thread_scaling"]:
    speedup = base_time / entry["duration_sec"]
    print(f"| {entry['threads']} | {entry['duration_sec']:.4f}s | {entry['sims_per_sec']/1e6:.2f} | {speedup:.2f}x |")

print("\n### 2. CPU vs Hybrid GPU Solver")
print("| Depth | CPU Time (s) | CPU Nodes | Hybrid Time (s) | Hybrid Tree Nodes | Speedup |")
print("|-------|--------------|-----------|-----------------|-------------------|---------|")
for entry in results["cpu_vs_hybrid"]:
    cpu_t = f"{entry['cpu']['duration_sec']:.4f}s" if entry['cpu']['duration_sec'] is not None else "TIMEOUT"
    cpu_n = entry['cpu']['nodes']
    hyb_t = f"{entry['hybrid']['duration_sec']:.4f}s" if entry['hybrid']['duration_sec'] is not None else "TIMEOUT"
    hyb_n = entry['hybrid']['nodes']
    
    if entry['cpu']['duration_sec'] and entry['hybrid']['duration_sec']:
        sp = f"{entry['cpu']['duration_sec'] / entry['hybrid']['duration_sec']:.2f}x"
    else:
        sp = "N/A"
    print(f"| {entry['depth']} | {cpu_t} | {cpu_n} | {hyb_t} | {hyb_n} | {sp} |")

print("\n### 3. Frontier Depth Tuning (Max Depth 16)")
print("| Frontier Depth | GPU Leaves | CPU Tree Nodes | Total Time (s) |")
print("|----------------|------------|----------------|----------------|")
for entry in results["frontier_depth_tuning"]:
    print(f"| {entry['frontier_depth']} | {entry['gpu_leaves']} | {entry['tree_nodes']} | {entry['duration_sec']:.4f}s |")
