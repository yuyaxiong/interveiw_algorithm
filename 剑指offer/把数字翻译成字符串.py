"""
给定一个数字，我们按照如下规则把它翻译为字符串：0翻译成“a”， 1翻译成“b”, ..... 11 翻译成“l”，...... 25 翻译成"z"。
一个数字可能有多个翻译。例如， 12258有5种不同的翻译，分别是bccfi,bwfi, bczi, mcfi, mzi。
请编程实现一个函数，用来计算一个数字有多少种不同的翻译方法。
"""
class Solution(object):
    def translation_count(self, num):
        if len(num) < 1:
            return 0
        elif len(num) == 1:
            return 1
        elif len(num) == 2:
            return 2 if int(num) <= 25 else 1
        double = 0
        if int(num[:1]) <= 25:
            double = self.translation_count(num[2:])
        signal = self.translation_count(num[1:])
        return signal + double

if __name__ == '__main__':
    s = Solution()
    print(s.translation_count(num=str(12258)))