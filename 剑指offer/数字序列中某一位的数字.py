"""
数字以01234567891011121315...的格式化到一个字符序列中。
在这个序列中，第5位（从0开始计数）是5，第13位是1，第19位
是4，等等。请写一个函数，求任意第n为对应的数字。
"""

class Solution(object):
    def digitAtIndex(self, n):
        # 前10位是0-9这10个只有一位的数字
        if n < 10:
            return n
        n -= 10
        m = 1
        while n - 9 * 10 ** m * (m+1) > 0:
            n -= 9 * 10 ** m * (m+1)
            m += 1

        remainder = n % m
        res = 1 * 10 ** m + n // (m+1)
        return str(res)[remainder]
    


if __name__ == "__main__":
    s = Solution() 
    print(s.digitAtIndex(n=1001))
