""""
输入一个排序数组，查找某个数是否在其中,并返回序号.
"""

class Solution(object):
    def find(self, nList, n):
        s, e = 0, len(nList)-1
        while s <= e:
            mid = (s+e) // 2
            if nList[mid] == n:
                return mid
            elif nList[mid] > n:
                e = mid-1
            else:
                s = mid+1
        return -1


if __name__ == '__main__':
    s = Solution()
    nList = [1, 2, 3, 4, 5, 6, 7]
    print(nList[s.find(nList, 3)])