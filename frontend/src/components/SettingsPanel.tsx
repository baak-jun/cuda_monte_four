import React from 'react';
import type { GameType } from '../utils/gameLogic';

interface SettingsPanelProps {
  game: GameType;
  c4Mode: string;
  setC4Mode: (m: string) => void;
  c4Simulations: number;
  setC4Simulations: (s: number) => void;
  c4Threads: number;
  setC4Threads: (t: number) => void;
  gomokuMode: string;
  setGomokuMode: (m: string) => void;
  statsOutput: string;
  onReset: () => void;
  isWaitingForAi: boolean;
}

export const SettingsPanel: React.FC<SettingsPanelProps> = ({
  game,
  c4Mode,
  setC4Mode,
  c4Simulations,
  setC4Simulations,
  c4Threads,
  setC4Threads,
  gomokuMode,
  setGomokuMode,
  statsOutput,
  onReset,
  isWaitingForAi
}) => {
  return (
    <div className="settings-panel">
      <div className="settings-title">
        {game === 'connect4' ? 'Connect Four Settings' : 'Gomoku Settings'}
      </div>

      {game === 'connect4' ? (
        <div id="c4-settings" style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
          <div className="setting-group">
            <label className="setting-label">AI 난이도 / 모델</label>
            <select
              value={c4Mode}
              onChange={(e) => setC4Mode(e.target.value)}
              disabled={isWaitingForAi}
            >
              <option value="cuda_mc">GPU Monte Carlo AI (RTX 4070)</option>
              <option value="cpu_solver">CPU Negamax Perfect AI (수 읽기)</option>
            </select>
          </div>

          <div className="setting-group">
            <label className="setting-label">GPU 시뮬레이션 수</label>
            <select
              value={c4Simulations}
              onChange={(e) => setC4Simulations(parseInt(e.target.value))}
              disabled={isWaitingForAi || c4Mode !== 'cuda_mc'}
            >
              <option value="100000">100,000 simulations (초고속)</option>
              <option value="1000000">1,000,000 simulations (기본)</option>
              <option value="10000000">10,000,000 simulations (최고지능)</option>
            </select>
          </div>

          <div className="setting-group">
            <label className="setting-label">CUDA 스레드 수</label>
            <select
              value={c4Threads}
              onChange={(e) => setC4Threads(parseInt(e.target.value))}
              disabled={isWaitingForAi || c4Mode !== 'cuda_mc'}
            >
              <option value="128">128 threads</option>
              <option value="256">256 threads</option>
              <option value="512">512 threads</option>
            </select>
          </div>
        </div>
      ) : (
        <div id="gomoku-settings" style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
          <div className="setting-group">
            <label className="setting-label">Gomoku AI Mode</label>
            <select
              value={gomokuMode}
              onChange={(e) => setGomokuMode(e.target.value)}
              disabled={isWaitingForAi}
            >
              <option value="cuda_mc">GPU Monte Carlo AI (RTX 4070)</option>
              <option value="cpu_solver">CPU Alpha-Beta Heuristic AI</option>
            </select>
          </div>
        </div>
      )}

      <div className="setting-group">
        <label className="setting-label">AI 연산 정보</label>
        <div className="stats-box" id="stats-output">
          {statsOutput}
        </div>
      </div>

      <button className="btn-action btn-reset" onClick={onReset} disabled={isWaitingForAi}>
        게임 초기화
      </button>
    </div>
  );
};
