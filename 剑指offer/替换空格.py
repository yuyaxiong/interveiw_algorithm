"""
请实现一个函数，把字符串中的每个空格替换成"%20"。
例如，输入"we are happy.",则输出"we%20are%20happy."
"""


class Solution(object):
    def replace_blank(self, strings):
        new_strings = ""
        for s in strings:
            if s == ' ':
                s = "%20"
            new_strings += s
        return new_strings

if __name__ == "__main__":
    strings= 'we are happy'
    s = Solution()
    print(s.replace_blank(strings))