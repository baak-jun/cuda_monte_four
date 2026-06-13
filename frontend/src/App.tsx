import React, { useState, useEffect } from 'react';
import { SettingsPanel } from './components/SettingsPanel';
import { StatusBar } from './components/StatusBar';
import { Board } from './components/Board';
import type { GameType, SideToMove, BoardState } from './utils/gameLogic';
import {
  checkC4Win,
  checkGomokuWin,
  sendGameLog,
  getApiHost
} from './utils/gameLogic';

export const App: React.FC = () => {
  const [game, setGame] = useState<GameType>('connect4');
  const [moves, setMoves] = useState<string>('');
  const [sideToMove, setSideToMove] = useState<SideToMove>('black');
  const [isWaitingForAi, setIsWaitingForAi] = useState<boolean>(false);
  const [isGameOver, setIsGameOver] = useState<boolean>(false);
  const [winner, setWinner] = useState<number>(0);
  const [isDraw, setIsDraw] = useState<boolean>(false);

  // Derive Board State synchronously during render to prevent layout height collapsing
  const isC4 = game === 'connect4';
  const boardRows = isC4 ? 6 : 15;
  const boardCols = isC4 ? 7 : 15;
  const boardState: BoardState = Array(boardRows)
    .fill(null)
    .map(() => Array(boardCols).fill(0));

  if (isC4) {
    let turn = 1; // 1 = black (human), 2 = white (ai)
    for (let i = 0; i < moves.length; i++) {
      const col = parseInt(moves[i]);
      if (!isNaN(col) && col >= 0 && col < 7) {
        for (let r = 5; r >= 0; r--) {
          if (boardState[r][col] === 0) {
            boardState[r][col] = turn;
            break;
          }
        }
      }
      turn = turn === 1 ? 2 : 1;
    }
  } else {
    if (moves) {
      const moveList = moves.split(',');
      let turn = 1;
      moveList.forEach((m) => {
        const [r, c] = m.split('_').map((x) => parseInt(x));
        if (!isNaN(r) && !isNaN(c) && r >= 0 && r < 15 && c >= 0 && c < 15) {
          boardState[r][c] = turn;
          turn = turn === 1 ? 2 : 1;
        }
      });
    }
  }

  // Settings
  const [c4Mode, setC4Mode] = useState<string>('cuda_mc');
  const [c4Simulations, setC4Simulations] = useState<number>(1000000);
  const [c4Threads, setC4Threads] = useState<number>(256);
  const [gomokuMode, setGomokuMode] = useState<string>('cuda_mc');
  const [statsOutput, setStatsOutput] = useState<string>('대기 중...');
  const [clientIpv4, setClientIpv4] = useState<string>('');

  // Fetch client IPv4 on load
  useEffect(() => {
    async function fetchIp() {
      try {
        const response = await fetch('https://api.ipify.org?format=json');
        const data = await response.json();
        if (data.ip) {
          setClientIpv4(data.ip);
        }
      } catch (err) {
        console.warn('Failed to fetch client IPv4:', err);
      }
    }
    fetchIp();
  }, []);

  // Update body class for layout adjustments
  useEffect(() => {
    document.body.className = `game-${game}`;
  }, [game]);

  // Check Win/Draw on moves or game change
  useEffect(() => {
    const isC4Active = game === 'connect4';
    const rows = isC4Active ? 6 : 15;
    const cols = isC4Active ? 7 : 15;
    const tempBoard: BoardState = Array(rows)
      .fill(null)
      .map(() => Array(cols).fill(0));

    if (isC4Active) {
      let turn = 1;
      for (let i = 0; i < moves.length; i++) {
        const col = parseInt(moves[i]);
        if (!isNaN(col) && col >= 0 && col < 7) {
          for (let r = 5; r >= 0; r--) {
            if (tempBoard[r][col] === 0) {
              tempBoard[r][col] = turn;
              break;
            }
          }
        }
        turn = turn === 1 ? 2 : 1;
      }
    } else {
      if (moves) {
        const moveList = moves.split(',');
        let turn = 1;
        moveList.forEach((m) => {
          const [r, c] = m.split('_').map((x) => parseInt(x));
          if (!isNaN(r) && !isNaN(c) && r >= 0 && r < 15 && c >= 0 && c < 15) {
            tempBoard[r][c] = turn;
            turn = turn === 1 ? 2 : 1;
          }
        });
      }
    }

    const gameWinner = isC4Active ? checkC4Win(tempBoard) : checkGomokuWin(tempBoard);
    if (gameWinner !== 0) {
      setWinner(gameWinner);
      setIsGameOver(true);
      setIsWaitingForAi(false);
      const activeMode = isC4Active ? c4Mode : gomokuMode;
      sendGameLog(
        game,
        activeMode,
        gameWinner === 1 ? 'black_win' : 'white_win',
        moves,
        clientIpv4
      );
      return;
    }

    let isGameDraw = false;
    if (isC4Active && moves.length === 42) {
      isGameDraw = true;
    } else if (!isC4Active && moves !== '' && moves.split(',').length === 225) {
      isGameDraw = true;
    }

    if (isGameDraw) {
      setIsDraw(true);
      setIsGameOver(true);
      setIsWaitingForAi(false);
      const activeMode = isC4Active ? c4Mode : gomokuMode;
      sendGameLog(game, activeMode, 'draw', moves, clientIpv4);
    }
  }, [moves, game, clientIpv4, c4Mode, gomokuMode]);

  // Handle cell clicks (player moves)
  const handleCellClick = (row: number, col: number) => {
    if (isWaitingForAi || isGameOver || sideToMove !== 'black') return;

    if (game === 'connect4') {
      if (boardState[0][col] !== 0) return; // Column is full
      const newMoves = moves + col;
      setMoves(newMoves);
      setSideToMove('white');
    } else {
      if (boardState[row][col] !== 0) return; // Spot is taken
      const moveCoord = `${row}_${col}`;
      const newMoves = moves ? moves + ',' + moveCoord : moveCoord;
      setMoves(newMoves);
      setSideToMove('white');
    }
  };

  // Trigger AI Move
  useEffect(() => {
    if (sideToMove === 'white' && !isGameOver && !isWaitingForAi) {
      setIsWaitingForAi(true);
      setStatsOutput('AI 연산 중...');
      if (game === 'connect4') {
        getC4AiMove(moves);
      } else {
        getGomokuAiMove(moves);
      }
    }
  }, [sideToMove, isGameOver, isWaitingForAi, game]);

  // Connect 4 AI API fetch
  const getC4AiMove = async (currentMoves: string) => {
    try {
      const host = getApiHost();
      const response = await fetch(host + '/api/connect4', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          moves: currentMoves,
          mode: c4Mode,
          simulations: c4Simulations,
          threads: c4Threads
        })
      });

      const data = await response.json();

      if (data.error) {
        alert('에러 발생: ' + data.error);
        revertLastUserMove(game, currentMoves);
        return;
      }

      setStatsOutput(
        `[AI 연산 완료]\n소요 시간: ${data.duration_sec.toFixed(4)}s\n` +
          (data.extra_info ? data.extra_info : '')
      );

      if (data.best_move !== undefined && data.best_move >= 0) {
        setMoves(currentMoves + data.best_move);
      }
      setSideToMove('black');
      setIsWaitingForAi(false);
    } catch (err) {
      console.error(err);
      alert('서버 연결 실패. app.py가 실행 중인지 확인하세요.');
      revertLastUserMove(game, currentMoves);
    }
  };

  // Gomoku AI API fetch
  const getGomokuAiMove = async (currentMoves: string) => {
    try {
      const host = getApiHost();
      const response = await fetch(host + '/api/gomoku', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          moves: currentMoves,
          mode: gomokuMode,
          simulations: 10000,
          threads: 256
        })
      });

      const data = await response.json();

      if (data.error) {
        alert('에러 발생: ' + data.error);
        revertLastUserMove(game, currentMoves);
        return;
      }

      setStatsOutput(
        `[Gomoku AI 연산 완료]\n소요 시간: ${data.duration_sec.toFixed(4)}s\n` +
          (data.extra_info ? data.extra_info : '')
      );

      if (data.best_move !== undefined && data.best_move !== '-1_-1') {
        setMoves(currentMoves ? currentMoves + ',' + data.best_move : data.best_move);
      }
      setSideToMove('black');
      setIsWaitingForAi(false);
    } catch (err) {
      console.error(err);
      alert('서버 연결 실패. app.py가 실행 중인지 확인하세요.');
      revertLastUserMove(game, currentMoves);
    }
  };

  // Helper to revert user's last move in case of network/AI error
  const revertLastUserMove = (currentGameType: GameType, currentMoves: string) => {
    if (currentGameType === 'connect4') {
      if (currentMoves.length > 0) {
        setMoves(currentMoves.slice(0, -1));
      }
    } else {
      if (currentMoves) {
        const moveList = currentMoves.split(',');
        moveList.pop();
        setMoves(moveList.join(','));
      }
    }
    setSideToMove('black');
    setIsWaitingForAi(false);
  };

  const resetGame = () => {
    setMoves('');
    setSideToMove('black');
    setIsWaitingForAi(false);
    setIsGameOver(false);
    setWinner(0);
    setIsDraw(false);
    setStatsOutput('대기 중...');
  };

  const switchGame = (gameType: GameType) => {
    if (isWaitingForAi) return;
    setGame(gameType);
    setMoves('');
    setSideToMove('black');
    setIsWaitingForAi(false);
    setIsGameOver(false);
    setWinner(0);
    setIsDraw(false);
    setStatsOutput('대기 중...');
  };

  return (
    <div className="main-wrapper-container">
      <header>
        <div className="logo">AI GAME PLATFORM</div>
        <div className="game-selector">
          <button
            className={`btn-select ${game === 'connect4' ? 'active' : ''}`}
            onClick={() => switchGame('connect4')}
            disabled={isWaitingForAi}
          >
            Connect Four (사목)
          </button>
          <button
            className={`btn-select ${game === 'gomoku' ? 'active' : ''}`}
            onClick={() => switchGame('gomoku')}
            disabled={isWaitingForAi}
          >
            Gomoku (오목)
          </button>
        </div>
      </header>

      <div className="main-container">
        <SettingsPanel
          game={game}
          c4Mode={c4Mode}
          setC4Mode={setC4Mode}
          c4Simulations={c4Simulations}
          setC4Simulations={setC4Simulations}
          c4Threads={c4Threads}
          setC4Threads={setC4Threads}
          gomokuMode={gomokuMode}
          setGomokuMode={setGomokuMode}
          statsOutput={statsOutput}
          onReset={resetGame}
          isWaitingForAi={isWaitingForAi}
        />

        <div className="game-wrapper">
          <StatusBar
            sideToMove={sideToMove}
            isGameOver={isGameOver}
            winner={winner}
            isDraw={isDraw}
          />
          <Board
            game={game}
            boardState={boardState}
            onCellClick={handleCellClick}
            isWaitingForAi={isWaitingForAi}
            isGameOver={isGameOver}
          />
        </div>
      </div>
    </div>
  );
};
