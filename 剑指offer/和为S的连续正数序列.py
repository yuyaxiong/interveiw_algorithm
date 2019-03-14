"""
输入一个正数s，打印出所有和为s的连续正数序列(至少含有两个数)。
例如：输入15，由于1+2+3+4+5=4+5+6=7+8=15，所以打印出3个连续
序列1-5，4-6和7-8。
"""
class Solution(object):
    def find_continuous_sequence(self, sum):
        small, bigger = 1, 2
        while small < bigger:
            if (small + bigger) * (bigger - small+1) / 2 == sum:
                self.print_num(small, bigger)
                bigger += 1
            elif (small + bigger) * (bigger - small+1) / 2 > sum:
                small += 1
            else:
                bigger += 1

    def print_num(self, small, bigger):
        strings = ''
        for n in range(small, bigger+1):
            strings += '%s, ' % n
        print(strings)

if __name__ == '__main__':
    s = Solution()
    s.find_continuous_sequence(15)