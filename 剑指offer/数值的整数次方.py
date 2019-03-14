"""
实现函数 power(base, exponent)求base的exponent次方。不得使用库函数。
同时不需要考虑大数问题。
"""


class Solution(object):
    def power(self, base, exp):
        if base == 0 and exp < 0:
            raise ZeroDivisionError('0.0 canont be raised to a negative power.')
        result = base
        for i in range(abs(exp)-1):
            result *= base
        if exp > 0:
            return result
        else:
            return 1/result




if __name__ == '__main__':
    s = Solution()
    base, exp = 0, -2
    print(s.power(base, exp))
    # print(base ** exp)