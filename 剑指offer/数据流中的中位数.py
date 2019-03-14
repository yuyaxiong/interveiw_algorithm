""""
如何得到一个数据流中的中位数，如果从数据流中读出奇数个数值，那么
中位数就是所有数值排序之后位于中间的数值。如果从数据流中读出偶数
个数值，那么中位数就是所有数值排序之后中间两个数的平均值

"""

class Solution(object):
    def __init__(self):
        self.lList = []
        self.rList = []

    def steam_median(self, d):
        if len(self.lList) == len(self.rList):
            if len(self.lList) == 0:
                self.lList.append(d)
            else:
                if self.lList[-1] > d:
                    self.lList.append(d)
                    self.max_heap(self.lList)
                else:
                    self.rList.append(d)
                    self.min_heap(self.rList)
                    val = self.rList.pop()
                    self.lList.append(val)
                    self.max_heap(self.lList)
        elif len(self.lList) > len(self.rList):
            if self.lList[-1] > d:
                self.lList.append(d)
                self.max_heap(self.lList)
                left_max = self.lList.pop()
                self.rList.append(left_max)
                self.min_heap(self.rList)
            else:
                self.rList.append(d)
                self.min_heap(self.rList)
        else:
            if self.rList[-1] < d:
                self.rList.append(d)
                self.min_heap(self.rList)
                min_val = self.rList.pop()
                self.lList.append(min_val)
                self.min_heap(self.lList)
            else:
                self.lList.append(d)
                self.max_heap(self.lList)
        return

    def get_median(self):
        if (len(self.lList) + len(self.rList)) % 2 == 0:
            return (self.lList[-1] + self.rList[-1])/2
        else:
            return self.lList[-1] if len(self.lList) > len(self.rList) else self.rList[-1]


    def max_heap(self, nList):
        for i in range(len(nList) - 1, -1, -1):
            while i > 0:
                if nList[i] > nList[(i - 1) // 2]:
                    nList[(i - 1) // 2], nList[i] = nList[i], nList[(i - 1) // 2]
                i = (i - 1) // 2
        nList[0], nList[-1] = nList[-1], nList[0]
        return

    def min_heap(self, nList):
        for i in range(len(nList) - 1, -1, -1):
            while i > 0:
                if nList[i] < nList[(i - 1) // 2]:
                    nList[(i - 1) // 2], nList[i] = nList[i], nList[(i - 1) // 2]
                i = (i - 1) // 2
        nList[0], nList[-1] = nList[-1], nList[0]
        return

if __name__ == '__main__':
    s = Solution()
    for i in [2,3,4,6,7,8,10]:
        s.steam_median(i)
    s.steam_median(24)
    print(s.get_median())