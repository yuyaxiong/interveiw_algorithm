"""
请实现一个函数用来匹配包含'.'和'*'的正则表达式。
模式中的字符'.'表示任意一个字符，而'*'表示它前面的
字符可以出现任意次(含0次)。在本题中，匹配是指字符串
的所有所有字符匹配整个模式。例如，字符串"aaa"与模式
"a.a"和"ab*ac*a"匹配，但与"aa.a"和"ab*a"均不匹配。
"""

class Solution(object):
    def match(self, strings, pattern):
        if strings is None or pattern is None:
            return False
        return self.match_core(strings, pattern)

    def match_core(self, strings, pattern):
        if len(strings) == 0 and len(pattern) == 0:
            return True
        if len(strings) != 0 and len(pattern) == 0:
            return False

        if len(pattern) >= 2 and pattern[1] == '*':
            if pattern[0] == strings[0] or (pattern[0] == '.' and len(strings) != 0):
                # parttern[0] 在string[0]中出现1次， 出现N次
                return self.match_core(strings[1:], pattern[2:]) or self.match_core(strings[1:], pattern)
            else:
                #pattern[0] 在string[0]中出现 出现0次
                return self.match_core(strings, pattern[2:])

        if (strings[0] == pattern[0]) or (pattern[0] == '.' and len(strings) != 0):
            return self.match_core(strings[1:], pattern[1:])

        return False


if __name__ == '__main__':
     s = Solution()
     print(s.match('aaa', 'ab*a'))
