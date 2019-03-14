"""
字符串的左旋转操作是把字符串前面的若干个字符转移到字符串的尾部。
请定义一个函数实现字符串左旋转操作的功能。比如，输入字符串"abcdefg"和
数字2，该函数将返回左旋转两位得到的结果"cdefgab"
"""

class Solution(object):
    def left_rotate_strings(self, strings, n):
        left, right = [s for s in strings[:n]], [s for s in strings[n:]]
        left, right = self.reverse_help(left, []), self.reverse_help(right, [])
        str_list = self.reverse_help(left + right, [])
        return ''.join(str_list)


    def reverse_help(self, str_list, rev_list):
        if len(str_list) == 0:
            return rev_list
        rev_list = self.reverse_help(str_list[1:], rev_list)
        rev_list.append(str_list[0])
        return rev_list


if __name__ == '__main__':
    s = Solution()
    strings = 'abcdefg'
    str_list = [s for s in strings]
    # print(s.reverse_help(str_list, []))
    print(s.left_rotate_strings(str_list, 2))