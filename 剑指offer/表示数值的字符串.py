"""
请实现一个函数用来判断字符串是否表示数值（包括：整数和小数）。
例如，字符串“+100”、“5e2”、“-123”,"3.1416"及“-1E-16”都
表示数值，但“12e”,"la3.14","1.2.3", '+-5'及’12e+5.4’
"""

class Solution(object):
    def is_numeric(self, n):
        num_str = [str(i) for i in range(1, 11)]
        if len(n)<=0:
            return False
        if n[0] in ['+', '-']:
            n = n[1:]
        if n[0] == '.':
            n = n[1:]
            for num in n:
                if num not in num_str:
                    return False
        else:
            if 'e' in n or 'E' in n:
                n_list = n.split('e') if 'e' in n else n.split('E')
                front, end = n_list[0], n_list[1]
                for i in front:
                    if i not in num_str + ['.']:
                        return False
                if len(end) == 0:
                    return False
                if end[0] in ['+', '-']:
                    end = end[1:]
                for i in end:
                    if i not in num_str:
                        return False
            else:
                for i in n:
                    if i not in num_str:
                        return False
        return True


if __name__ == "__main__":
    s = Solution()
    print(s.is_numeric('12e+5.4'))