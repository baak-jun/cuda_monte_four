from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import subprocess
import time
import os
import re
from collections import defaultdict
import json
from datetime import datetime

app = FastAPI(title="AI Board Game Server")

# Simple in-memory rate limiter: max 20 requests per minute per IP
ip_request_history = defaultdict(list)

def get_client_ip(request: Request) -> str:
    cf_ip = request.headers.get("cf-connecting-ip")
    if cf_ip:
        return cf_ip
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    real_ip = request.headers.get("x-real-ip")
    if real_ip:
        return real_ip
    return request.client.host

def check_rate_limit(client_ip: str) -> bool:
    now = time.time()
    # Keep only requests within the last 60 seconds
    ip_request_history[client_ip] = [t for t in ip_request_history[client_ip] if now - t < 60]
    if len(ip_request_history[client_ip]) >= 20:
        return False
    ip_request_history[client_ip].append(now)
    return True

# Allow CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

PROJECT_DIR = r"C:\Users\Nabi\Desktop\26-1\Parallel Programming\project"
BIN_DIR = os.path.join(PROJECT_DIR, "build", "Release")

MC_BIN = os.path.join(BIN_DIR, "play_monte_carlo_cuda.exe")
SOLVER_BIN = os.path.join(BIN_DIR, "perfect_solver_cpu.exe")

GOMOKU_MC_BIN = os.path.join(BIN_DIR, "gomoku_monte_carlo_cuda.exe")
GOMOKU_SOLVER_BIN = os.path.join(BIN_DIR, "gomoku_solver_cpu.exe")

class Connect4Request(BaseModel):
    moves: str
    mode: str  # "cuda_mc" or "cpu_solver"
    simulations: int = 1000000
    threads: int = 256

class GomokuRequest(BaseModel):
    moves: str
    mode: str  # "cuda_mc" or "cpu_solver"
    simulations: int = 10000
    threads: int = 256

@app.get("/")
async def get_index():
    index_path = os.path.join(PROJECT_DIR, "web", "index.html")
    if not os.path.exists(index_path):
        raise HTTPException(status_code=404, detail="index.html not found")
    return FileResponse(index_path)

@app.post("/api/connect4")
async def play_connect4(req: Connect4Request, request: Request):
    # 1. DoS Rate Limit check
    client_ip = get_client_ip(request)
    if not check_rate_limit(client_ip):
        raise HTTPException(status_code=429, detail="Too many requests. Please wait a minute.")

    # Verify binaries exist
    bin_path = MC_BIN if req.mode == "cuda_mc" else SOLVER_BIN
    if not os.path.exists(bin_path):
        return {"error": f"Executable not found at {bin_path}. Please build the project in Release mode."}

    # 2. Prevent DoS by capping parameters
    simulations = min(max(1000, req.simulations), 1000000)
    threads = min(max(32, req.threads), 512)

    # Setup command
    if req.mode == "cuda_mc":
        cmd = [
            bin_path,
            "--moves", req.moves,
            "--once",
            "--simulations-per-move", str(simulations),
            "--threads", str(threads),
            "--seed", "12345"
        ]
    else: # cpu_solver
        # Set max search depth dynamically to prevent hang on early moves
        m_len = len(req.moves)
        max_depth = min(42, m_len + 13)
        cmd = [
            bin_path,
            "--moves", req.moves,
            "--max-depth", str(max_depth)
        ]

    start_time = time.perf_counter()
    try:
        # Run subprocess
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=60)
        duration = time.perf_counter() - start_time
        
        if res.returncode != 0:
            return {"error": f"Process exited with code {res.returncode}. Error: {res.stderr}"}
            
        stdout = res.stdout
        print(f"Stdout:\n{stdout}")
        
        # Parse best move
        best_move = -1
        match_move = re.search(r"best_move\s+(\d+)", stdout)
        if match_move:
            best_move = int(match_move.group(1))
            
        # Parse outcome or result
        outcome = "unknown"
        if req.mode == "cuda_mc":
            # Search for draws, wins
            if "draw" in stdout.lower():
                outcome = "draw"
            elif "computer wins" in stdout.lower():
                outcome = "white wins"
            elif "you win" in stdout.lower():
                outcome = "black wins"
        else: # cpu_solver
            match_res = re.search(r"result\s+(\w+)", stdout)
            if match_res:
                outcome = match_res.group(1) # win, loss, draw, etc.

        # Extract extra info
        extra_info = ""
        if req.mode == "cuda_mc":
            # Extract win/draw rates table
            lines = stdout.splitlines()
            rates_start = -1
            for idx, line in enumerate(lines):
                if "computer win rate" in line:
                    rates_start = idx
                    break
            if rates_start != -1:
                extra_info = "\n".join(lines[rates_start:rates_start+4])
        else: # cpu_solver
            # Extract summary, nodes, cache hits, cutoffs, and depth note
            lines = stdout.splitlines()
            info_parts = []
            for line in lines:
                if any(x in line for x in ["result", "score", "nodes", "cache_hits", "cutoffs", "note"]):
                    info_parts.append(line)
            extra_info = "\n".join(info_parts)

        return {
            "best_move": best_move,
            "outcome": outcome,
            "duration_sec": duration,
            "extra_info": extra_info
        }

    except subprocess.TimeoutExpired:
        return {"error": "AI solver timed out (limit: 60s)."}
    except Exception as e:
        return {"error": f"Server error: {str(e)}"}

@app.post("/api/gomoku")
async def play_gomoku(req: GomokuRequest, request: Request):
    # 1. DoS Rate Limit check
    client_ip = get_client_ip(request)
    if not check_rate_limit(client_ip):
        raise HTTPException(status_code=429, detail="Too many requests. Please wait a minute.")

    # Verify binaries exist
    bin_path = GOMOKU_MC_BIN if req.mode == "cuda_mc" else GOMOKU_SOLVER_BIN
    if not os.path.exists(bin_path):
        return {"error": f"Executable not found at {bin_path}. Please build the project."}

    # 2. Prevent DoS by capping parameters
    simulations = min(max(100, req.simulations), 20000)
    threads = min(max(32, req.threads), 512)

    # Setup command
    if req.mode == "cuda_mc":
        cmd = [
            bin_path,
            "--moves", req.moves,
            "--simulations-per-move", str(simulations),
            "--threads", str(threads),
            "--seed", "12345"
        ]
    else: # cpu_solver
        cmd = [
            bin_path,
            "--moves", req.moves,
            "--max-depth", "4"
        ]

    start_time = time.perf_counter()
    try:
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=60)
        duration = time.perf_counter() - start_time
        
        if res.returncode != 0:
            return {"error": f"Process exited with code {res.returncode}. Error: {res.stderr}"}
            
        stdout = res.stdout
        print(f"Gomoku Stdout:\n{stdout}")
        
        # Parse best move (e.g., best_move 8_9)
        best_move = "-1_-1"
        match_move = re.search(r"best_move\s+(\d+_\d+)", stdout)
        if match_move:
            best_move = match_move.group(1)
            
        outcome = "unknown"
        match_res = re.search(r"result\s+(\w+)", stdout)
        if match_res:
            outcome = match_res.group(1)
            if outcome == "black_win":
                outcome = "black wins"
            elif outcome == "white_win":
                outcome = "white wins"

        extra_info = ""
        lines = stdout.splitlines()
        if req.mode == "cuda_mc":
            rates_start = -1
            for idx, line in enumerate(lines):
                if "win rates for candidates" in line:
                    rates_start = idx
                    break
            if rates_start != -1:
                extra_info = "\n".join(lines[rates_start:rates_start+8])
        else: # cpu_solver
            info_parts = []
            for line in lines:
                if any(x in line for x in ["result", "score", "nodes", "duration"]):
                    info_parts.append(line)
            extra_info = "\n".join(info_parts)

        return {
            "best_move": best_move,
            "outcome": outcome,
            "duration_sec": duration,
            "extra_info": extra_info
        }

    except subprocess.TimeoutExpired:
        return {"error": "Gomoku AI solver timed out (limit: 60s)."}
    except Exception as e:
        return {"error": f"Server error: {str(e)}"}

class LogRequest(BaseModel):
    game: str
    mode: str
    outcome: str
    moves: str

@app.post("/api/log")
async def log_game(req: LogRequest, request: Request):
    client_ip = get_client_ip(request)
    if not check_rate_limit(client_ip):
        raise HTTPException(status_code=429, detail="Too many requests.")
        
    timestamp = datetime.now().isoformat()
    log_entry = {
        "timestamp": timestamp,
        "client_ip": client_ip,
        "game": req.game,
        "mode": req.mode,
        "outcome": req.outcome,
        "moves": req.moves
    }
    
    log_file_path = os.path.join(PROJECT_DIR, "game_logs.jsonl")
    try:
        with open(log_file_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
        return {"status": "success"}
    except Exception as e:
        return {"error": f"Failed to write log: {str(e)}"}
