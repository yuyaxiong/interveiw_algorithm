"""
假设把某股票的价格按照时间先后循序储存在数组中，
请问买卖该股票一次可能获得的最大利润是多少？
例如：一只股票在某个时间节点的价格为[9,11,8,5,7,12,16,14]。
如果我们能在价格为5的时候买入并在价格为16的时候卖出，则能够
收获最大的利润11。
"""
class Solution(object):
    def max_diff(self, nList):
        delta_list = []
        for i in range(1, len(nList)):
            delta_list.append(nList[i]-nList[i-1])
        max_val = 0
        cumsum = 0
        for n in delta_list:
            cumsum += n
            if cumsum < 0:
                cumsum = 0
            max_val = max(cumsum, max_val)
        return max_val

class Solution1(object):
    def max_diff(self, nList):
        if len(nList) <= 0 or nList is None:
            return 0
        min_val = nList[0]
        max_diff = nList[1] - min_val
        #
        for i in range(2, len(nList)):
            min_val = min(nList[i-1], min_val)
            current_diff = nList[i] - min_val
            max_diff = max(current_diff, max_diff)
        return max_diff



if __name__ == '__main__':
    s = Solution1()
    nList = [9, 11, 8, 5, 7, 12, 16, 14]
    print(s.max_diff(nList))


