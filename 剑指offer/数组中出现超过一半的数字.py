"""
数组中有一个数字出现的次数超过数组长度的一半，请找出这个数字。
例如：输入一个长度为9的数组[1,2,3,2,2,2,5,4,2]。由于数字2
在数组中出现了5次，超过数组长度的一半，因此输出2。
"""

class Solution(object):
    def more_than_half_num(self, nums):
        counter, num = 0, nums[0]
        for n in nums[1:]:
            if counter == 0:
                num = n
                counter = 1
            elif n == num:
                counter += 1
            else:
                counter -= 1
        return counter



if __name__ == '__main__':
    s = Solution()
    nums = [1,2,3,2,2,2,5,4,2]
    print(s.more_than_half_num(nums))