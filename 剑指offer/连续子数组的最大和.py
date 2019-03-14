"""
输入一个整型数组，数组里有正数也有负数。数组中的一个或连续多个正数组成一个子数组。
求所有子数组的和的最大值。要求时间复杂度为O(n)
"""

class Solution(object):
    def find_greatest_sum(self, nList):
        sum_val = 0
        max_val = -1 << 16
        for n in nList:
            if sum_val < 0:
                sum_val = 0
            sum_val += n
            max_val = max(sum_val, max_val)
        return max_val


if __name__ == '__main__':
    s = Solution()
    print(s.find_greatest_sum([1, -2, 3, 10, -4, 7, 2, -5]))


