"""
请设计一个函数，用来判断在一个矩阵中是否存在一条包含
某字符串所有字符的路径。路径可以从矩阵中的任意一格开始，
每一步可以再矩阵中向左右上下移动一格。如果一条路径经过了
矩阵的某一格，那么该路径不能再次进入该格子。例如，在下面的
3*4的矩阵中包含一条字符串"bfce"的路径（路径中的字母用下划线标出）。
但矩阵中不包含字符串"abfb"的路径，因为字符串的第一个字符b占据了
矩阵中的第一行第二个格子之后，路径不能再次进入这个格子。
"""


class Solution1(object):
    def has_path(self, mat, rows, cols, strings):
        if mat == None or rows < 1 or cols < 1 or strings is None:
            return False
        visited = [[False for _ in range(cols+1)] for _ in range(rows+1)]
        path_legnth = 0
        for row in range(rows):
            for col in range(cols):
                # 遍历所有启点
                if self.has_path_core(mat, rows, cols, row, col, strings, path_legnth, visited):
                    return True
        return False

    def has_path_core(self, mat, rows, cols, row, col, strings, path_length, visited):
        if len(strings) == path_length:
            return True
        has_path = False
        if row >= 0 and row <= rows and col >= 0 and col <= cols and mat[col][row] == strings[path_length] and visited[col][row] is False:
            path_length += 1
            visited[col][row] = True
            has_path = self.has_path_core(mat, rows, cols, row, col-1, strings, path_length, visited)\
                       or self.has_path_core(mat, rows, cols, row-1, col, strings, path_length, visited)\
                        or self.has_path_core(mat, rows, cols, row, col+1, strings, path_length, visited)\
                        or self.has_path_core(mat, rows, cols, row+1, col, strings, path_length, visited)

            if has_path is False:
                path_length -= 1
                visited[col][row] = False
        return has_path


if __name__ == '__main__':
    s = Solution1()
    mat = [['a','b','t','g'],['c', 'f','c', 's'],['j', 'd', 'e','h']]
    path = 'bfce'
    n, m = len(mat), len(mat[0])
    print(s.has_path(mat, n-1, m-1, path))