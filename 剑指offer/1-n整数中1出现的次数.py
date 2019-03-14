"""
输入一个整数n,求1-n这n个整数的十进制表示中1出现的次数。
例如：输入12， 1-12这些整数中包含1的数字有1，10，11和12，
1一共出现了5次。
"""

class Solution(object):
    def NumberOf1Between1AndN(self, n):
        if n <= 0:
            return 0
        # stringsN = str(n)
        return self.NumberOf1(str(n))

    def NumberOf1(self, strN):
        if strN is None or strN[0] < '0' or strN[0] > '9' or len(strN) == 0:
            return 0
        first = int(strN[0])
        length = len(strN)
        if length == 1 and first == 0:
            return 0
        if length == 1 and first > 0:
            return 1
        # 假设 strN 是 21345
        # numFirstDigit 是数字 10000~19999的第一位中的数目
        numFirstDigit = 0
        if first > 1:
            # 将数字分为 21345-1346和1-1345部分（递归）
            numFirstDigit = 10 ** (length - 1)
        elif first == 1:
            # 第一位不是1 则1的和是剩余位数数字+1
            numFirstDigit = int(strN[1:]) + 1
        # numOtherDigit 是 1346-21345除第一位之外的数位中的数目
        numOtherDigit = first * (length - 1) * 10 ** (length - 2)
        # numRecursive 是1-1345中的数目
        numRecursive = self.NumberOf1(strN[1:])

        return numFirstDigit + numOtherDigit + numRecursive

if __name__ == '__main__':
    s = Solution()
    print(s.NumberOf1Between1AndN(21345))

