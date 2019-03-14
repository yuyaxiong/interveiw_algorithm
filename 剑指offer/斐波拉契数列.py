"""
求斐波拉契数列的第n项。
写一个函数，输入n，求斐波拉契数列的第N项。
斐波拉切数列的定义如下：
f(n)=
{
    0 , n== 0
    1, n == 1
    f(n-1) + f(n-2), n> 1
}
"""

class Solution(object):
    def Fibonacci(self, n):
        if n == 0:
            return 0
        elif n == 1:
            return 1
        res = [0, 1]
        for i in range(1, n):
            res.append(res[i] + res[i-1])
        return res[-1]


if __name__ == '__main__':
    s = Solution()
    for i in range(6):
        print(s.Fibonacci(n=i))