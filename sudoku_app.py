import streamlit as st
import numpy as np

st.title("🧩 Sudoku Solver")

# Create 9x9 grid
grid = []
for i in range(9):
    row = []
    cols = st.columns(9)
    for j in range(9):
        val = cols[j].text_input(f"{i}{j}", "", max_chars=1)
        row.append(int(val) if val.isdigit() else 0)
    grid.append(row)

board = np.array(grid)

def find_empty(board):
    for i in range(9):
        for j in range(9):
            if board[i][j] == 0:
                return i, j
    return None

def valid(board, num, pos):
    row, col = pos
    if num in board[row]: return False
    if num in board[:, col]: return False
    box_x, box_y = col // 3, row // 3
    for i in range(box_y*3, box_y*3+3):
        for j in range(box_x*3, box_x*3+3):
            if board[i][j] == num: return False
    return True

def solve(board):
    empty = find_empty(board)
    if not empty: return True
    row, col = empty
    for num in range(1, 10):
        if valid(board, num, (row, col)):
            board[row][col] = num
            if solve(board): return True
            board[row][col] = 0
    return False

if st.button("Solve Sudoku"):
    b = board.copy()
    if solve(b):
        st.success("✅ Solved Sudoku!")
        st.table(b)
    else:
        st.error("❌ No solution found")
        
