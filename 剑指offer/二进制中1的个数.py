"""
请实现一个函数，输入一个整数，输出该数二进制表示中1的个数。
例如，把9表示成二进制是1001，有2位是1。因此，如果输入9，则
该函数输出2。
"""

class Solution(object):
    def binary_count(self, n):
        counter = 0
        if n<0:
            n = ~n + 1
        while n > 1:
            m = n % 2
            counter += m
            n = n // 2
        counter += n
        return counter

class Solution1(object):
    def binary_count(self, n):
        counter = 0
        flag = 1
        while flag < (1 << 31):
            if n & flag:
                counter += 1
            flag = (flag << 1)
        return counter

if __name__ == '__main__':
    s1 = Solution1()
    s = Solution()
    print(s1.binary_count(n=-16666))
    print(s.binary_count(n=-16666))