"""
输入一个矩阵，按照从外向里以顺时针的顺序依次打印出每个数字。
例如如果输入以下矩阵：
1 2 3 4
5 6 7 8
9 10 11 12
13 14 15 16
"""
class Solution(object):
    def print_matrix_clock_wisely(self, numbers, columns, rows):
        if numbers is None and columns <=0 and rows <= 0:
            return
        start = 0
        while(columns > start * 2 and rows > start * 2):
            self.print_matrix_in_circle(numbers, columns, rows, start)
            start += 1

    def print_matrix_in_circle(self, numbers, columns, rows, start):
        endX = columns - 1 - start
        endY = rows - 1 - start

        # 从左到右
        for i in range(start, endX+1):
            number = numbers[start][i]
            self.print_number(number)

        # 从上到下
        if start < endY:
            for i in range(start+1, endY+1):
                number = numbers[i][endX]
                self.print_number(number)

        # 从右到左
        if start < endX and start < endY:
            for i in range(endX-1, start, -1):
                number = numbers[endY][i]
                self.print_number(number)

        # 从下到上
        if start < endX and start < endY-1:
            for i in range(endY-1, start+1, -1):
                numbers = numbers[i][start]
                self.print_number(number)

    def print_number(self, number):
        print(number)


if __name__ == '__main__':
    mat = [[1, 2,3, 4],[5, 6,7,8],[9, 10, 11, 12],[13, 14, 15,16]]
    s = Solution()
    s.print_matrix_clock_wisely(mat,4, 4)