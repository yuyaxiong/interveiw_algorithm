from typing import List

# 37. 解数独
class Solution:
    def solveSudoku(self, board: List[List[str]]) -> None:
        """
        Do not return anything, modify board in-place instead.
        """
        self.recur_solve(board)
        return board

    def recur_solve(self, board):
        status = False
        for i in range(9):
            for j in range(9):
                if board[i][j] == '.':
                    row_idx_list = self.get_row_list(i, j)
                    col_idx_list = self.get_col_list(i, j)
                    row_col_list = self.get_row_col_list(i, j)
                    remain_n_list = self.get_remain_n(row_idx_list + col_idx_list + row_col_list, board)
                    # 有符合的数字
                    status = False
                    for n in remain_n_list:
                        board[i][j] = n
                        next_status = self.recur_solve(board)
                        if next_status is False:
                            continue
                        else:
                            status = True
                            break
                    # 没有符合的数字
                    if status is False:
                        board[i][j] = "."
                        return status 
        return True

    def get_remain_n(self, idx_list, board):
        n_list = [str(i) for i in range(1, 10)]
        for row_col in idx_list:
            i, j = row_col[0], row_col[1]
            if board[i][j]  != "." and board[i][j] in n_list:
                n_list.remove(board[i][j])
        return n_list

    def get_row_list(self, i, j):
        tmp = []
        for n in range(9):
            if n != j:
                tmp.append([i, n])
        return tmp 
    
    def get_col_list(self, i, j):
        tmp = []
        for n in range(9):
            if n != i:
                tmp.append([n, j])
        return tmp

    def get_row_col_list(self, i, j):
        row_list, col_list = None, None
        if i < 3:
            row_list = [0, 3]
        elif i >= 3 and i < 6:
            row_list = [3, 6]
        elif i >= 6:
            row_list = [6, 9]
            
        if j < 3:
            col_list = [0, 3]
        elif j >= 3 and j < 6:
            col_list = [3, 6]
        elif j >= 6:
            col_list = [6, 9]

        row_col_list = []
        for row in range(row_list[0], row_list[1]):
            for col in range(col_list[0], col_list[1]):
                if i == row and j == col:
                    continue
                row_col_list.append([row, col])
        return row_col_list


def testCase():
    boards = [["5","3",".",".","7",".",".",".","."],["6",".",".","1","9","5",".",".","."],[".","9","8",".",".",".",".","6","."],
    ["8",".",".",".","6",".",".",".","3"],["4",".",".","8",".","3",".",".","1"],["7",".",".",".","2",".",".",".","6"],
    [".","6",".",".",".",".","2","8","."],[".",".",".","4","1","9",".",".","5"],[".",".",".",".","8",".",".","7","9"]]
    sol = Solution()
    ret = sol.solveSudoku(boards)
    for n in ret:
        print(n)
    # print(ret)
    print("---" * 10)
    ans = [["5","3","4","6","7","8","9","1","2"],["6","7","2","1","9","5","3","4","8"],["1","9","8","3","4","2","5","6","7"],
    ["8","5","9","7","6","1","4","2","3"],["4","2","6","8","5","3","7","9","1"],["7","1","3","9","2","4","8","5","6"],
    ["9","6","1","5","3","7","2","8","4"],["2","8","7","4","1","9","6","3","5"],["3","4","5","2","8","6","1","7","9"]]
    for n in ans:
        print(n)

if __name__ == "__main__":
    testCase()