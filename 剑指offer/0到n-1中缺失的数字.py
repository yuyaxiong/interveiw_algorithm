"""
一个长度为n-1的递增排序数组中的所有数字都是唯一的，
并且每个数字都在范围0-n-1之内。在范围0-n-1内的n个
数字中有且只有一个数字不在该数组中，请找出这个数字。
"""

class Solution(object):
    def get_missing_num(self, nList , length, s, e):
        if s > e:
            return -1
        mid = (s + e)//2
        if mid > 0:
            if nList[mid] != mid and nList[mid-1] == mid-1:
                return mid
            elif nList[mid] != mid and nList[mid-1] != mid-1:
                return self.get_missing_num(nList, length, s, mid-1)
            # elif nList[mid] == mid and
        # else:
        if nList[mid] == mid:
            return self.get_missing_num(nList, length, mid+1, e)
        else:
            return mid

if __name__ == "__main__":
     s = Solution()
     nList1=[0,1,2,3,5,6,7]
     nList2 = [1,2,3,4,5,6]
     nList4 = [0,1,2,3,4,5,7]
     nList3 = [0]
     print(s.get_missing_num(nList4, len(nList4), 0, len(nList4)-1))



