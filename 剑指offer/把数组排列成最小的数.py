"""
输入一个正整数数组，把数组里所有数字凭借起来排成一个数，
打印能凭借出的所有数字中最小的一个。例如， 输入数组[3,32,321],
则打印出这3个数字能排序成的最小数字321323
"""

class Solution(object):
    def print_min_num(self, n):
        res = str(n[0])
        for i, s in enumerate(n[1:]):
            flag = self.compare(res, str(s))
            if flag is '>':
                res = str(s) + res
            else:
                res = res + str(s)
            # else:
            #     res = res + str(i)
        return res



    def compare(self, n1, n2):
        if str(n1)+str(n2) > str(n2) + str(n1):
            return '>'
        elif str(n1)+str(n2) < str(n2) + str(n1):
            return '<'
        else:
            return '='


if __name__ == '__main__':
    s = Solution()
    n = [3, 32, 321]
    print(s.print_min_num(n))


