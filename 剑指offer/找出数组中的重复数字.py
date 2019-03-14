"""
在一个长度为n的数组里的所有数字都在0-n-1的范围内。数组中某些数字是重复的，
但不知道有几个数字重复了，也不知道每个数字的重复了几次。请找出数组中任意一个
重复的数字。例如，如果输入长度为7的数组[2,3,1,0,2,5,3]
"""


class Solution(object):
    def find_duplicates(self, nList):
        for i in range(len(nList)):
            while nList[i] != i:
                if nList[nList[i]] == nList[i]:
                    return nList[i]
                else:
                    nList[nList[i]], nList[i] = nList[i], nList[nList[i]]
        return None

if __name__ == '__main__':
    s = Solution()
    nList = [2, 1, 3, 0, 2, 5, 3]
    print(s.find_duplicates(nList))