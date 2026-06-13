import React from 'react';
import type { SideToMove } from '../utils/gameLogic';

interface StatusBarProps {
  sideToMove: SideToMove;
  isGameOver: boolean;
  winner: number; // 0 = none, 1 = black (human), 2 = white (ai)
  isDraw: boolean;
}

export const StatusBar: React.FC<StatusBarProps> = ({
  sideToMove,
  isGameOver,
  winner,
  isDraw
}) => {
  let dotClass = `turn-dot ${sideToMove}`;
  let turnText = '';
  let statusText = '진행 중';
  let statusColor = '#a0aec0';

  if (isGameOver) {
    if (isDraw) {
      turnText = '게임 종료 (무승부)';
      statusText = 'DRAW';
      statusColor = '#a0aec0';
    } else {
      const winnerName = winner === 1 ? '흑돌 승리' : '백돌 승리';
      turnText = `게임 종료 (${winnerName})`;
      statusText = winner === 1 ? 'BLACK WINS' : 'WHITE WINS';
      statusColor = winner === 1 ? '#ecc94b' : '#e53e3e';
    }
  } else {
    if (sideToMove === 'black') {
      turnText = '당신의 턴 (흑돌)';
    } else {
      turnText = 'AI 연산 중... (백돌)';
    }
  }

  return (
    <div className="status-bar">
      <div className="status-turn">
        {!isGameOver && <div className={dotClass}></div>}
        <span>{turnText}</span>
      </div>
      <div id="game-status-label" style={{ color: statusColor }}>
        {statusText}
      </div>
    </div>
  );
};
