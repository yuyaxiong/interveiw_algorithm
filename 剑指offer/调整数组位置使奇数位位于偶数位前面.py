"""
输入一个整数数组，实现一个函数来调整该数组中数字的顺序，
使得所有奇数位于数组的前半部分，所有偶数位于数组后半部分。

"""
class Solution(object):
    def reorder_odd_event(self, nList):
        sp = 0
        ep = len(nList)-1
        while sp < ep:
            if nList[sp] % 2 == 0:
                sp += 1
            else:
                if nList[ep] % 2 == 1:
                    ep -= 1
                else:
                    nList[sp], nList[ep] = nList[ep], nList[sp]
                    sp += 1
                    ep -= 1
        return


class Solution1(object):
    def reorder_odd_event(self, nList):
        sp = 0
        ep = len(nList)-1
        while sp < ep:
            while nList[sp] % 2 == 0:
                sp += 1
            while nList[ep] % 2 == 1:
                ep -= 1
            if sp < ep:
                nList[sp], nList[ep] = nList[ep], nList[sp]

class Solution2(object):
    def reorder_odd_event(self, nList):
        sp = 0
        ep = len(nList)-1
        while sp < ep:
            while self.is_event(nList[sp]):
                sp += 1
            while not self.is_event(nList[ep]):
                ep -= 1
            if sp < ep:
                nList[sp], nList[ep] = nList[ep], nList[sp]

    def is_event(self, num):
        return num % 2 == 0


if __name__ == '__main__':
    s = Solution()
    s1 = Solution1()
    s2 = Solution2()
    nList = [1,2,3,4,5]
    # print(s.reorder_odd_event(nList))
    s2.reorder_odd_event(nList)
    print(nList)

