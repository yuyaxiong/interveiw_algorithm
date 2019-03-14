"""
圆圈中最后剩下的数字
题目：0,1...,n-1这n个数字排成一个圆圈，从数字0开始，
每次从这个圆圈里删除第m个数字。求出这个圆圈里剩下的最后一个数字。
"""

class Solution1(object):
    def last_remaining(self, n, m):
        nList = [i for i in range(n)]
        return self.last_remaining_help(nList, m)

    def last_remaining_help(self, nList, m):
        if len(nList) == 1:
            return nList[0]
        idx = 0
        for _ in range(m):
            idx = 0 if idx + 1 == len(nList) else idx + 1
        return self.last_remaining_help(nList[idx+1:] + nList[:idx], m)

class Solution2(object):
    def last_remaining(self, n, m):
        if n < 1 and m < 1:
            return -1
        last = 0
        for i in range(2, n):
            last = (last + m) % i
        return last


if __name__ == '__main__':
    s = Solution2()
    print(s.last_remaining(3, 4))
