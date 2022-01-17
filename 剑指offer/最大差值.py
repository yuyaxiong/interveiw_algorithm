
class Solution(object):
    """
    已知一个乱序数组，现找出一对a1,a2的差值a2-a1最大，并且a1.index < a2.index,
    需要在o(n)的时间内求出
    """
    def max_delta(self, nList):
        if len(nList) <2:
            return 0
        i, j, max_idx = 0, 1, 1
        max_delta = nList[1] - nList[0]
        while j < len(nList):
            if nList[j] - nList[0] > max_delta:
                max_idx = j
                max_delta = nList[j] - nList[0]
            j += 1
        min_val = nList[0]
        min_idx = 0
        while i < max_idx:
            i += 1
            if nList[i] < min_val:
                min_val = nList[i]
                min_idx = i
        max_delta = nList[max_idx] - nList[min_idx]
        return max_delta, max_idx, min_idx

if __name__ == "__main__":
    s = Solution()
    nList = [8,7,2,1,2,3,0,5,9,6]
    delta, max_idx, min_idx = s.max_delta(nList)
    print(delta, max_idx, min_idx)
