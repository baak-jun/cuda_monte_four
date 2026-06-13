from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import subprocess
import time
import os
import re

app = FastAPI(title="AI Board Game Server")

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

class Connect4Request(BaseModel):
    moves: str
    mode: str  # "cuda_mc" or "cpu_solver"
    simulations: int = 1000000
    threads: int = 256

@app.get("/")
async def get_index():
    index_path = os.path.join(PROJECT_DIR, "web", "index.html")
    if not os.path.exists(index_path):
        raise HTTPException(status_code=404, detail="index.html not found")
    return FileResponse(index_path)

@app.post("/api/connect4")
async def play_connect4(req: Connect4Request):
    # Verify binaries exist
    bin_path = MC_BIN if req.mode == "cuda_mc" else SOLVER_BIN
    if not os.path.exists(bin_path):
        return {"error": f"Executable not found at {bin_path}. Please build the project in Release mode."}

    # Setup command
    if req.mode == "cuda_mc":
        cmd = [
            bin_path,
            "--moves", req.moves,
            "--once",
            "--simulations-per-move", str(req.simulations),
            "--threads", str(req.threads),
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
