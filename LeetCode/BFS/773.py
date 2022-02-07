# 773. 滑动谜题
from typing import List

class Solution:
    def slidingPuzzle(self, board: List[List[int]]) -> int:
        row, col = len(board), len(board[0])
        start_row, start_col = None, None
        for i in range(row):
            for j in range(col):
                if board[i][j] == 0:
                    start_row, start_col = i, j

        q_list = [[[start_row, start_col, board]]]
        board_dict = dict()

        counter = 0
        while len(q_list) > 0:
            cur_list =  q_list.pop(0)
            tmp_list = []
            for cur in cur_list:
                cur_x, cur_y, cur_board = cur[0], cur[1], cur[2]
                if self.checkBoard(cur_board):
                    return counter                
                if cur_x + 1 < row:
                    tmp_board = [b[::] for b in cur_board]
                    tmp_board[cur_x][cur_y], tmp_board[cur_x+1][cur_y] = tmp_board[cur_x+1][cur_y], tmp_board[cur_x][cur_y]
                    tmp_board_str = self.castStr(tmp_board)
                    if board_dict.get(tmp_board_str) is None:
                        tmp_list.append([cur_x+1, cur_y, tmp_board])
                        board_dict[tmp_board_str] = 1
                if cur_x - 1 >= 0:
                    tmp_board = [b[::] for b in cur_board]
                    tmp_board[cur_x-1][cur_y], tmp_board[cur_x][cur_y] = tmp_board[cur_x][cur_y], tmp_board[cur_x-1][cur_y]
                    tmp_board_str = self.castStr(tmp_board)
                    if board_dict.get(tmp_board_str) is None:
                        tmp_list.append([cur_x-1, cur_y, tmp_board])
                        board_dict[tmp_board_str] = 1
                if cur_y + 1 < col:
                    tmp_board = [b[::] for b in cur_board]
                    tmp_board[cur_x][cur_y], tmp_board[cur_x][cur_y+1] = tmp_board[cur_x][cur_y+1], tmp_board[cur_x][cur_y]
                    tmp_board_str = self.castStr(tmp_board)
                    if board_dict.get(tmp_board_str) is None:
                        tmp_list.append([cur_x, cur_y+1, tmp_board])
                        board_dict[tmp_board_str] = 1
                if cur_y - 1 >= 0:
                    tmp_board = [b[::] for b in cur_board]
                    tmp_board[cur_x][cur_y], tmp_board[cur_x][cur_y-1] = tmp_board[cur_x][cur_y-1], tmp_board[cur_x][cur_y]
                    tmp_board_str = self.castStr(tmp_board)
                    if board_dict.get(tmp_board_str) is None:
                        tmp_list.append([cur_x, cur_y-1, tmp_board])
                        board_dict[tmp_board_str] = 1
            if len(tmp_list) > 0:
                q_list.append(tmp_list)
                counter += 1
        return -1

    def checkBoard(self, board):
        status = True
        for n1, n2 in zip(board[0], [1, 2, 3]):
            if n1 != n2:
                status = False
        for n1, n2 in zip(board[1],[4, 5, 0]):
            if n1 != n2:
                status = False
        return status

    def castStr(self, board):
        n_list = []
        for n in board:
            n_list.append("".join([str(i) for i in n]))
        return "".join(n_list)

def testCase():
    board = [[1,2,3],[4,0,5]]
    sol = Solution()
    ret = sol.slidingPuzzle(board)
    print(ret)

def testCase1():
    board = [[1,2,3],[5,4,0]]
    sol = Solution()
    ret = sol.slidingPuzzle(board)
    print(ret)

if __name__ == "__main__":
    testCase1()
    testCase()
