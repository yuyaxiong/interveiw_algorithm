"""
二分查找最小值
给出一个排序的nList,查找其中nList >= x 的最小的index
例如：nlist=[1,3,5,7,9], x=2 则返回 为1，
"""

class Solution(object):
    def find_n(self, nList, n, s, e):
        if len(nList) == 0 or nList[0] > n or nList[-1] < n:
            return False
        mid = (s + e)//2+1
        if mid -1 >= 0:
            if nList[mid] >= n and nList[mid-1] < n:
                return mid
            elif nList[mid] >= n and nList[mid-1] >= n:
                return self.find_n(nList[:mid], n, s, mid-1)
            elif nList[mid] <= n and nList[mid-1] <= n:
                return self.find_n(nList[mid:], n, mid, e)
        else:
            if nList[mid] >= n:
                return mid
            else:
                return self.find_n(nList[mid:], n, mid, e)

if __name__ == '__main__':
     s = Solution()
     nList1 = [1, 3, 5, 7, 9]
     nList2 = [1,3,5,7,9,11]
     nList3 = [1, 3]
     print(s.find_n(nList3, 1, 0, len(nList3)-1))




