# CUDA/OpenMP 게임 AI 병렬화 프로젝트

CUDA와 OpenMP를 이용해 커넥트 4(Connect 4)와 오목(Gomoku) AI의 계산을 병렬화한 프로젝트입니다.

- CUDA Flat Monte Carlo rollout
- OpenMP 멀티코어 CPU Monte Carlo
- CPU Negamax/Alpha-Beta 탐색
- React + TypeScript 웹 화면
- FastAPI 백엔드
- CPU/GPU 성능 및 정확성 벤치마크

> 현재 CUDA Monte Carlo 구현은 트리 기반 MCTS가 아니라, 독립적인 무작위 rollout을 대량 실행해 후보 수를 평가하는 **Flat Monte Carlo** 방식입니다.

## 주요 구현

### 비트보드 승리 판정

커넥트 4 보드를 흑·백 각각 하나의 `uint64_t` 비트보드로 표현합니다. 세로, 가로, 대각선 방향의 비트 시프트와 AND 연산으로 연속된 돌을 검사하여 반복적인 좌표 순회를 줄였습니다.

### Xorshift32 난수 생성

각 CUDA 스레드는 독립적인 32비트 난수 상태를 가집니다. `Xorshift32`는 XOR와 시프트 연산만 사용하므로 rollout 내부에서 가볍게 실행할 수 있습니다.

### CUDA Shared Memory 집계

스레드별 승·패·무승부 결과를 블록의 공유 메모리에 먼저 집계한 뒤 블록 단위로 전역 카운터에 반영합니다. 보완 실험에서는 전역 atomic 직접 사용보다 중앙값 기준 약 3.4% 개선되었습니다.

### OpenMP CPU 병렬화

CPU Monte Carlo 구현은 OpenMP로 rollout 반복문을 병렬화합니다. `--threads` 옵션으로 사용할 CPU 스레드 수를 지정할 수 있습니다.

### CPU-GPU 하이브리드 탐색

CPU가 frontier를 생성하고 GPU가 하위 노드를 평가하는 실험적 탐색기도 포함합니다. 현재 구현은 분기 발산, 중복 계산, 전치 테이블 부재로 인해 CPU 단독 탐색보다 느릴 수 있으며 개선 연구 대상으로 남겨 두었습니다.

## 프로젝트 구조

| 경로 | 설명 |
|---|---|
| `src/` | C++ 및 CUDA 소스 코드 |
| `include/` | 커넥트 4·오목 공용 헤더 |
| `tests/` | 보드 규칙 단위 테스트 |
| `frontend/` | React + TypeScript 프론트엔드 |
| `web/` | FastAPI가 제공하는 빌드된 웹 정적 파일 |
| `app.py` | FastAPI 서버와 게임 AI API |
| `benchmark.py` | CPU/GPU 성능 및 정확성 측정 |
| `benchmark_results.json` | 최신 벤치마크 결과 |
| `start_services.bat` | Windows 웹 서버 간편 실행 파일 |

## 실행 환경

웹 포털 전체 기능을 실행하려면 다음 프로그램이 필요합니다.

- Windows 10/11
- Python 3.10 이상
- CMake 3.24 이상
- Visual Studio 2022 또는 Build Tools의 `Desktop development with C++`
- NVIDIA GPU 및 CUDA Toolkit 11.8 이상
- Node.js 20 이상: 프론트엔드를 다시 빌드할 때만 필요

CUDA가 없는 환경에서는 `-DBUILD_CUDA=OFF`로 CPU 프로그램만 빌드할 수 있습니다. 다만 웹의 GPU Monte Carlo 모드는 사용할 수 없습니다.

## 처음부터 웹 실행하기

### 1. 저장소 복제

```powershell
git clone https://github.com/baak-jun/cuda_monte_four.git
cd cuda_monte_four
```

PR 브랜치를 직접 확인할 때는 저장소를 복제한 뒤 해당 브랜치로 전환합니다.

### 2. Python 패키지 설치

PowerShell에서 프로젝트 루트를 기준으로 실행합니다.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

PowerShell 실행 정책 때문에 가상환경 활성화가 막히면 다음 명령을 현재 창에서 한 번 실행합니다.

```powershell
Set-ExecutionPolicy -Scope Process Bypass
```

### 3. C++ 및 CUDA 프로그램 빌드

```powershell
cmake -S . -B build -DBUILD_CUDA=ON
cmake --build build --config Release
```

빌드가 끝나면 실행 파일이 `build\Release\`에 생성됩니다.

### 4. 프론트엔드 빌드

저장소의 `web/` 폴더에 이미 빌드 결과가 포함되어 있으므로 일반 실행 시에는 생략할 수 있습니다. 프론트엔드 코드를 수정했을 때만 실행합니다.

```powershell
cd frontend
npm ci
npm run build
cd ..
```

### 5. 웹 서버 실행

가장 간단한 방법은 프로젝트 루트의 `start_services.bat`을 더블 클릭하는 것입니다. 서버가 실행되면 기본 브라우저가 자동으로 열립니다.

직접 실행하려면 다음 명령을 사용합니다.

```powershell
python -m uvicorn app:app --host 127.0.0.1 --port 8889
```

브라우저 주소:

```text
http://127.0.0.1:8889/
```

서버를 종료하려면 실행 중인 터미널에서 `Ctrl+C`를 누릅니다.

## 웹에서 선택할 수 있는 AI

### 커넥트 4

- GPU Monte Carlo: 후보 열마다 CUDA rollout을 실행
- CPU Solver: 제한 깊이 Negamax/Alpha-Beta 탐색

### 오목

- GPU Monte Carlo: 후보 위치별 CUDA rollout
- CPU Solver: 휴리스틱을 포함한 제한 깊이 Alpha-Beta 탐색

웹 API는 과도한 실행 시간을 방지하기 위해 요청값에 상한을 적용합니다.

- 커넥트 4 Monte Carlo: 최대 1,000,000 simulations
- 오목 Monte Carlo: 최대 20,000 simulations
- AI 실행 제한 시간: 60초

## CLI 실행 예시

### 커넥트 4 GPU Monte Carlo

```powershell
.\build\Release\play_monte_carlo_cuda.exe `
  --moves 343 `
  --once `
  --simulations-per-move 1000000 `
  --threads 256
```

`--moves`는 0부터 6까지의 열 번호를 착수 순서대로 이어 붙인 문자열입니다.

### 커넥트 4 CPU Solver

```powershell
.\build\Release\perfect_solver_cpu.exe --moves 343 --max-depth 15
```

### 오목 GPU Monte Carlo

```powershell
.\build\Release\gomoku_monte_carlo_cuda.exe `
  --moves 7_7,8_8 `
  --simulations-per-move 10000 `
  --threads 256
```

오목 착수는 `행_열` 형식이며 각 착수를 쉼표로 구분합니다.

### 오목 CPU Solver

```powershell
.\build\Release\gomoku_solver_cpu.exe --moves 7_7,8_8 --max-depth 4
```

## 벤치마크 실행

빠른 동작 확인:

```powershell
python benchmark.py --quick
```

1,000만 회 Monte Carlo 정식 측정:

```powershell
python benchmark.py --mc-simulations 10000000
```

결과는 기본적으로 `benchmark_results.json`에 저장됩니다. 벤치마크는 CPU와 GPU 결과 카운트가 일치하는지도 함께 검사합니다.

## 최근 측정 결과

커넥트 4 Flat Monte Carlo 10,000,000회 기준:

| 구현 | 중앙값 |
|---|---:|
| OpenMP CPU 16스레드 | 약 4.17초 |
| CUDA 전체 실행 | 약 0.261초 |
| CUDA 커널 | 약 85.3ms |

- CUDA 전체 실행은 OpenMP 16스레드 대비 약 16.0배 빨랐습니다.
- CUDA 커널 시간만 비교하면 약 48.9배 빨랐습니다.
- 동일 seed에서 CPU/GPU의 흑 승·백 승·무승부 카운트가 일치했습니다.

측정값은 GPU 드라이버, 전원 설정, 백그라운드 프로세스에 따라 달라질 수 있습니다.

## 문제 해결

### `Executable not found` 오류

Release 빌드가 완료됐는지 확인합니다.

```powershell
cmake --build build --config Release
```

### `No module named uvicorn` 오류

```powershell
python -m pip install -r requirements.txt
```

### CUDA 컴파일러를 찾지 못하는 경우

- CUDA Toolkit 설치 여부를 확인합니다.
- Visual Studio C++ 개발 도구가 설치됐는지 확인합니다.
- 새 터미널을 열고 다시 CMake 설정을 실행합니다.

CPU 프로그램만 확인하려면 다음과 같이 빌드합니다.

```powershell
cmake -S . -B build-cpu -DBUILD_CUDA=OFF
cmake --build build-cpu --config Release
```

## 참고

- 커넥트 4 비트 인덱스는 `row + col * 7`을 사용하며 각 열의 7번째 비트는 sentinel입니다.
- 오목은 CPU, CUDA, 웹에서 exact-five 규칙을 사용합니다. 정확히 5개가 연결되어야 승리이며 6목 이상의 장목은 승리로 처리하지 않습니다.
- 전체 깊이 완전 탐색은 매우 오래 걸릴 수 있으므로 작은 `--max-depth`부터 실행하는 것을 권장합니다.
