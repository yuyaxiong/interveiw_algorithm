"""
第一个只出现一次的字符。
在字符串中找出第一个只出现一次的字符。如输入："abaccdeff"，则输出"b"
"""
class Solution(object):
    def first_not_repeating(self, pStrings):
        if pStrings is None:
            return None
        s_count = {}
        for s in pStrings:
            if s_count.get(s) is None:
                s_count[s] = 1
            else:
                s_count[s] += 1
        for s in pStrings:
            if s_count[s] == 1:
                return s


if __name__ == '__main__':
    s = Solution()
    print(s.first_not_repeating("abaccdeff"))
