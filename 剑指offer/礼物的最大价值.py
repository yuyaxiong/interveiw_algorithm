"""
在一个mXn的棋盘的每一格斗放有一个礼物，每个礼物都有一定的价值（价值大于0）。
你可以从棋盘的左上角开始拿格子里的礼物，并每次向左或者向下移动一格，直到到达棋盘的
右下角。给定一个棋盘机器上面的礼物，请计算你最多能拿到多少价值的礼物？

"""
class Solution(object):
    def max_gift_value(self, gifts, rows, cols):        
        matrixA = [[0 for _ in range(cols)] for _ in range(rows)]
        matrixA[0][0] = gifts[0][0]
        for row in range(1, rows):
            matrixA[row][0] = gifts[row][0] + matrixA[row-1][0]
        for col in range(1, cols):
            matrixA[0][col] += matrixA[0][col-1] + gifts[0][col]
        for row in range(1, rows):
            for col in range(1, cols):
                matrixA[row][col] = max(matrixA[row-1][col], matrixA[row][col-1]) + gifts[row][col]
        return matrixA[-1][-1]


if __name__ == '__main__':
    s = Solution()
    gifts=[[1, 10, 3, 8], [12, 2, 9, 6],[5, 7, 4, 11], [3, 7, 16, 5]]
    print(s.max_gift_value(gifts, 4, 4))

