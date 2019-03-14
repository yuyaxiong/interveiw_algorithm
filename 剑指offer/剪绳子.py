"""
给你一个长度为n的绳子，请把绳子剪成m段（m,n都是整数，n>1并且m>1),
每段绳子的长度记为k[0],k[1],...k[m]。请问k[0]*k[1]*...k[m]可能
的最大乘积是多少？例如，当绳子的长度为8时，我们把它剪成长度分别为2，3，3的三段，
此时得到的最大乘积是18。
"""

class Solution1(object):
    def cut(self, length):
        result = [0, 1, 2, 3]
        div = [0,0,1,2]
        if length <= 3:
            return div[length]
        for i in range(4, length+1):
            max_val = 0
            n, m = 1, i-1
            while n < m:
                max_val = max(result[n] * result[m], i, max_val)
                n += 1
                m -= 1
            result.append(max_val)
        return result[-1]





if __name__ == '__main__':
    s = Solution1()
    print(s.cut(8))