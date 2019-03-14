"""
输入一个英文句子，翻转句子中单词的顺序，但单词内字符的顺序不变。
为了简单起见，标点符号和普通字母一样处理。例如输入字符串
“I am a student.”，则输出"student. a am I"
"""


class Solution(object):
    def reversed(self, strings):
        str_list = strings.split(' ')
        rev_list = []
        rev_list = self.reversed_help(str_list, rev_list)
        return ' '.join(rev_list)


    def reversed_help(self, str_list, rev_list):
        if len(str_list) == 0:
            return rev_list
        rev_list = self.reversed_help(str_list[1:], rev_list)
        rev_list.append(str_list[0])
        return rev_list


if __name__ == '__main__':
    s = Solution()
    print(s.reversed('I am a student.'))
