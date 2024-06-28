## Game Board Description

The game board is an \(N \times N\) grid, where N is an odd number, resulting in \(N \times N\) squares for players to move on.

### Board Indexing

- **Player Positions**:
  - The board index for players is given by \((x, y)\):
    - \(x\) is the row index (\(0 \leq x < N\)); 0 is the top row.
    - \(y\) is the column index (\(0 \leq y < N\)); 0 is the left column.

- **Wall Placements**:
  - Wall placement indices are given by \((i, j)\):
    - \(i\) is the row index (\(0 \leq i < N-1\)); 0 is the top intersection row (below the top row that the player can stand on).
    - \(j\) is the column index (\(0 \leq j < N-1\)); 0 is the left intersection column (right of the left column that the player can stand on).

### Player Information

- **Player 1**:
  - Starts at the bottom.
  - Moves first.
  - The starting square index is \((N-1, \lfloor N/2 \rfloor)\).

- **Player 2**:
  - Starts at the top.
  - Moves second.
  - The starting square index is \((0, \lfloor N/2 \rfloor)\).

**Note:** This description may differ from the C++ implementation due to optimizations and specific details of each implementation. Python is preferred for connecting with ML techniques due to its flexibility and integration capabilities.
