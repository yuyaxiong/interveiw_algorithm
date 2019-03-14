"""
数组中数值和下标相等的元素
假设一个单调递增的数组里的每个元素都是整数并且是唯一的。
请编程实现一个函数，找出数组中任意一个数值等于其下标的元素。
例如：在数组[-3, -1, 1, 3, 5]
"""

class Solution(object):
    def get_num_same_as_index(self, num, length):
        if num is None or length <= 0:
            return -1
        left = 0
        right = length -1
        while left <= right:
            mid = (left + right) // 2
            if num[mid] == mid:
                return mid
            if num[mid] > mid:
                right = mid -1
            else:
                left = mid + 1
        return -1

class Solution1(object):
    def get_num_same_as_index(self, num, length, s, e):
        if s > e:
            return -1
        mid = (s + e) //2
        if nList1[mid] == mid:
            return mid
        elif num[mid] > mid:
            e = mid -1
        else:
            s = mid + 1
        return self.get_num_same_as_index(num , length, s, e)



class Solution3(object):
    def get_idx(self, nList, n):
        left, right = 0, len(nList)-1
        while left < right:
            mid = (left + right) // 2
            if nList[mid] > n:
                right = mid -1
            elif nList[mid] < n:
                left = mid + 1
            else:
                return mid
        return left


if __name__ == '__main__':
    nList1 = [-3, -1, 1, 3, 5]
    s = Solution3()
    # print(s.get_num_same_as_index(nList1, len(nList1), 0,len(nList1)-1))
    print(s.get_idx(nList1, 2))