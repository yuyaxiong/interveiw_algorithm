"""
从扑克牌中随机抽5张牌，判断是不是一个顺子，即这5张牌
是不是连续的。2-10位数字本身，A为1，J为11，Q为12，K为13，
而大小王可以看做任意数。
"""
class Solution(object):
    def is_continuous(self, nList, length):
        zero_counter = 0
        missing_counter = 0
        nList = sorted(nList)
        idx = 0
        for i in range(len(nList)):
            if nList[i] == 0:
                zero_counter += 1
            else:
                idx = i
                break

        for i in range(idx+1, len(nList)):
            missing_counter += nList[i] - nList[i-1] -1
        if zero_counter >= missing_counter:
            return True
        else:
            return False

if __name__ == '__main__':
    s = Solution()
    nList = [0, 0, 1, 4, 5]
    print(s.is_continuous(nList, len(nList)))