export type GameType = 'connect4' | 'gomoku';
export type SideToMove = 'black' | 'white';
export type BoardState = number[][];

export function checkC4Win(boardState: BoardState): number {
  for (let r = 0; r < 6; r++) {
    for (let c = 0; c < 7; c++) {
      const color = boardState[r][c];
      if (color === 0) continue;
      if (c <= 3 && color === boardState[r][c + 1] && color === boardState[r][c + 2] && color === boardState[r][c + 3]) return color;
      if (r <= 2 && color === boardState[r + 1][c] && color === boardState[r + 2][c] && color === boardState[r + 3][c]) return color;
      if (r <= 2 && c <= 3 && color === boardState[r + 1][c + 1] && color === boardState[r + 2][c + 2] && color === boardState[r + 3][c + 3]) return color;
      if (r >= 3 && c <= 3 && color === boardState[r - 1][c + 1] && color === boardState[r - 2][c + 2] && color === boardState[r - 3][c + 3]) return color;
    }
  }
  return 0;
}

export function checkGomokuWin(boardState: BoardState): number {
  const directions = [
    [0, 1],   // Horizontal
    [1, 0],   // Vertical
    [1, 1],   // Diagonal down-right
    [-1, 1]   // Diagonal up-right
  ];

  for (let r = 0; r < 15; r++) {
    for (let c = 0; c < 15; c++) {
      const color = boardState[r][c];
      if (color === 0) continue;

      for (const [dr, dc] of directions) {
        const endR = r + 4 * dr;
        const endC = c + 4 * dc;
        if (endR < 0 || endR >= 15 || endC < 0 || endC >= 15) continue;

        let match = true;
        for (let i = 1; i <= 4; i++) {
          if (boardState[r + i * dr][c + i * dc] !== color) {
            match = false;
            break;
          }
        }

        if (match) {
          // Check for overline (6-in-a-row)
          // 1. Check cell before start
          const prevR = r - dr;
          const prevC = c - dc;
          const hasPrev = prevR >= 0 && prevR < 15 && prevC >= 0 && prevC < 15;
          if (hasPrev && boardState[prevR][prevC] === color) continue;

          // 2. Check cell after end
          const nextR = r + 5 * dr;
          const nextC = c + 5 * dc;
          const hasNext = nextR >= 0 && nextR < 15 && nextC >= 0 && nextC < 15;
          if (hasNext && boardState[nextR][nextC] === color) continue;

          return color; // Exact 5-in-a-row
        }
      }
    }
  }
  return 0;
}

export function getApiHost(): string {
  return window.location.protocol === 'file:' ? 'http://127.0.0.1:8889' : '';
}

export async function sendGameLog(
  game: GameType,
  mode: string,
  outcome: string,
  moves: string,
  clientIpv4: string
): Promise<void> {
  try {
    const host = getApiHost();
    await fetch(host + '/api/log', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        game,
        mode,
        outcome,
        moves,
        client_ipv4: clientIpv4
      })
    });
  } catch (err) {
    console.warn('Log transmission failed:', err);
  }
}
