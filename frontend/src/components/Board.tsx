import React from 'react';
import type { GameType, BoardState } from '../utils/gameLogic';

interface BoardProps {
  game: GameType;
  boardState: BoardState;
  onCellClick: (row: number, col: number) => void;
  isWaitingForAi: boolean;
  isGameOver: boolean;
}

export const Board: React.FC<BoardProps> = ({
  game,
  boardState,
  onCellClick,
  isWaitingForAi,
  isGameOver
}) => {
  const isC4 = game === 'connect4';
  const boardClass = isC4 ? 'board-c4' : 'board-gomoku';

  const handleC4Click = (colIndex: number) => {
    if (isWaitingForAi || isGameOver) return;
    // Connect 4 click only needs column. We can represent it as click on column.
    // Row doesn't matter for the click, backend determines row.
    onCellClick(0, colIndex);
  };

  const handleGomokuClick = (rowIndex: number, colIndex: number) => {
    if (isWaitingForAi || isGameOver) return;
    onCellClick(rowIndex, colIndex);
  };

  if (isC4) {
    return (
      <div id="board-container">
        <div className={boardClass}>
          {boardState.map((rowArr, rowIndex) =>
            rowArr.map((state, colIndex) => {
              return (
                <div
                  key={`c4-${rowIndex}-${colIndex}`}
                  className="cell-c4"
                  onClick={() => handleC4Click(colIndex)}
                >
                  {state === 1 && <div className="stone black"></div>}
                  {state === 2 && <div className="stone white"></div>}
                </div>
              );
            })
          )}
        </div>
      </div>
    );
  }

  return (
    <div id="board-container">
      <div className={boardClass}>
        {boardState.map((rowArr, rowIndex) =>
          rowArr.map((state, colIndex) => {
            return (
              <div
                key={`gomoku-${rowIndex}-${colIndex}`}
                className="cell-gomoku"
                onClick={() => handleGomokuClick(rowIndex, colIndex)}
              >
                {state === 1 && <div className="stone black"></div>}
                {state === 2 && <div className="stone white"></div>}
              </div>
            );
          })
        )}
      </div>
    </div>
  );
};
