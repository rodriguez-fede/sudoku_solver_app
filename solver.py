def solve(board, validate=True):
    """
    Solves the sudoku in-place and returns False if the puzzle
    is invalid. Returns True otherwise. Numbers are represented
    as int from 1 to 9. Blank cells are represented as 0.
    Sample input:
        [[5, 3, 0, 0, 7, 0, 0, 0, 0],
         [6, 0, 0, 1, 9, 5, 0, 0, 0],
         [0, 9, 8, 0, 0, 0, 0, 6, 0],
         [8, 0, 0, 0, 6, 0, 0, 0, 3],
         [4, 0, 0, 8, 0, 3, 0, 0, 1],
         [7, 0, 0, 0, 2, 0, 0, 0, 6],
         [0, 6, 0, 0, 0, 0, 2, 8, 0],
         [0, 0, 0, 4, 1, 9, 0, 0, 5],
         [0, 0, 0, 0, 8, 0, 0, 7, 9]]
    """
    memo = set()
    blank = []
    for i, row in enumerate(board):
        for j, x in enumerate(row):
            if x == 0:
                blank.append((i, j))
            else:
                rcg = _generate_rcg(i, j, x)
                if validate:
                    if any(x in memo for x in rcg):
                        return False
                memo.add(rcg[0])
                memo.add(rcg[1])
                memo.add(rcg[2])

    return _solve(board, memo, blank)


def is_valid(board):
    """Checks whether the sudoku is valid."""
    memo = set()
    for i, row in enumerate(board):
        for j, x in enumerate(row):
            if x != 0:
                rcg = _generate_rcg(i, j, x)
                if any(x in memo for x in rcg):
                    return False
                memo.add(rcg[0])
                memo.add(rcg[1])
                memo.add(rcg[2])
    return True


def _generate_rcg(i, j, x):
    """
    Returns the row, column, and grid string to save in memory.
    Args:
        i (int): Row index
        j (int): Column index
        x (int): Value in grid at row i and column j
    Returns:
        Tuple[str]: Row, column, and grid are represented as
                    i(x), (x)j, gi(x)gj respectively where
                    1 <= i, j <= 9, 1 <= gi, gj <= 3.
    """
    return ('{}({})'.format(i, x), '({}){}'.format(x, j),
            '{}({}){}'.format(i // 3, x, j // 3))


def _solve(board, memo, blank):
    """Solves the puzzle recursively using backtracking."""
    if not blank:
        return True
    i, j = blank[-1]
    for x in range(1, 10):
        rcg = _generate_rcg(i, j, x)
        if all(x not in memo for x in rcg):
            board[i][j] = x
            memo.add(rcg[0])
            memo.add(rcg[1])
            memo.add(rcg[2])
            blank.pop()
            if _solve(board, memo, blank):
                return True
            board[i][j] = '.'
            memo.discard(rcg[0])
            memo.discard(rcg[1])
            memo.discard(rcg[2])
            blank.append((i, j))
    return False