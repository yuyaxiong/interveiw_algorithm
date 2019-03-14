""""
在一个二维数组中，每一行都按照从左到右递增的顺序排序，
每一列都按照从上到下递增的顺序排序。请完成一个函数，
输入这样的一个二维数组和一个整数，判断数组中是否含有该数组。
"""


class Solution:
    def Find(self, array, num):
        if len(array) < 0 :
            return False
        n, m = len(array), len(array[0])
        i, j = 0, m-1
        while i < n and j >= 0:
            if array[i][j] == num:
                return True
            elif array[i][j] > num:
                j = j-1
            else:
                i = i+1
        return False


    def Find1(self, array, num):
        if len(array) < 0 :
            return False
        n, m = len(array), len(array[0])
        i, j = n-1, 0
        while i >= 0 and j < m:
            if array[i][j] == num:
                return True
            elif array[i][j] < num:
                j = j+1
            else:
                i = i-1
        return False

if __name__ == '__main__':
    mat = [[1, 2, 8, 9], [2, 4, 9, 12], [4, 7, 10, 13], [6, 8, 11, 15]]
    s = Solution()
    print(s.Find1(mat, 22))