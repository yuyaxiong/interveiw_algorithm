"""
写一个函数，求两个整数之和，要求在函数体内不得使用+-*/四则运算符号。
"""


class Solution(object):
    def add(self, num1, num2):
        while num2 != 0:
            sum, carry = num1^num2, (num1&num2) << 1
            num1, num2 = sum, carry
        return num1

if __name__ == '__main__':
    s = Solution()
    n1, n2 = 10, 23
    print(s.add(n1, n2))
    assert s.add(n1, n2) == n1+n2

