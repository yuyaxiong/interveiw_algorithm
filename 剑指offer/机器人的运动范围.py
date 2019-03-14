"""
地上有一个m行n列的方格。一个机器人从坐标(0,0)的格子开始移动，
它每次可以向左右上下移动一格，但不能进入行坐标和列坐标的数位之和
大于k的格子。例如，当k为18时，机器人能够进入方格(35, 37)，因为
3+5+3+7=18。但它不能进入放格(35, 38),因为3+5+3+8=19。请问
该机器人能够到达多少个格子。
"""


class Solution(object):
    def moving_count(self, threshold, rows, cols):
        visited = [[False for _ in range(rows+1)] for _ in range(cols+1)]
        return self.robot(0, 0, rows, cols, threshold, visited)

    def robot(self, row, col, rows, cols, threshold, visited):
        if row >= 0 and row <= rows and col >= 0 and col <= cols and self.get_digit_sum(col, row) <= threshold and visited[row][col] is False:
            visited[row][col] = True
            return 1 + self.robot(row-1, col, rows, cols, threshold, visited) + self.robot(row+1, col, rows, cols, threshold, visited) \
                    + self.robot(row, col-1, rows, cols, threshold, visited) + self.robot(row, col+1, rows, cols, threshold, visited)
        else:
            return 0

    def get_digit_sum(self, col, row):
        sum = 0
        # for n in str(col):
        #     sum += int(n)
        # for m in str(row):
        #     sum += int(m)
        while col > 0:
            sum += col % 10
            col //= 10
        while row > 0:
            sum += row % 10
            row //= 10
        return sum

if __name__ == '__main__':
    s = Solution()
    print(s.moving_count(4, 3, 3))
    print(s.get_digit_sum(3, 3))