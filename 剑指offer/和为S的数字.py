"""
和为s的两个数
输入一个递增排序的数组和一个数字s，在数组中查找两个数，使得它们的和正好是s。
如果有多对数字的和等于s，则输出任意一对即可。
"""

class Solution(object):
    def find_num_with_sum(self, nList, s):
        if nList is None:
            return False
        sp = 0
        ep = len(nList)-1
        while sp < ep:
            if nList[sp] + nList[ep] == s:
                return nList[sp], nList[ep]
            elif nList[sp] + nList[ep] > s:
                ep -= 1
            else:
                sp += 1
        return False

if __name__ == '__main__':
     s = Solution()
     nList =[1, 2, 4, 7, 11, 15]
     print(s.find_num_with_sum(nList, 23))



