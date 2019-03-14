"""
把一个数组最开始的若干个元素搬到数组的末尾，我们称之为数组的旋转。
输入一个递增排序的数组的一个旋转，输出旋转数组的最小元素。
例如，数组[3,4,5,1,2]为[1,2,3,4,5]的一个旋转，该数组的最小值为1。
"""


class Solution(object):
    def get_min(self, nList):
        length = len(nList)
        begin, mid, end = 0, length//2, length-1
        while True:
            if nList[begin] < nList[mid] and nList[mid] > nList[end]:
                begin = mid
                mid = (begin + end)//2
            elif nList[begin] > nList[mid] and nList[mid] < nList[end]:
                end = mid
                mid = (begin + end)//2
            else:
                return nList[end]


if __name__ == '__main__':
    nlist = [3, 4, 5, 6, 1, 2]
    s = Solution()
    print(s.get_min(nlist))
