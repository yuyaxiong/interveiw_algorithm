"""
请实现一个函数，用来找出字符流中第一个只出现一次的字符。
例如，当从字符流中只读出前两个字符“go”时，第一个只出现
一次的字符是“g”；当从该字符流中读出前6个字符“google”时，
第一个只出现一次的字符是“I”
"""

class Solution(object):
    def __init__(self):
        self.strings = ''
        self.strs_count = {}
        self.first_idx = None

    def insert(self, s):
        self.strings += s
        if self.strs_count.get(s) is None:
            self.strs_count[s] = 1
        else:
            self.strs_count[s] += 1

    def first_appearing_once(self):
        for s in self.strings:
            if self.strs_count[s] == 1:
                return s


if __name__ == '__main__':
    s = Solution()
    for q in "oooooo":
        s.insert(q)
    print(s.first_appearing_once())