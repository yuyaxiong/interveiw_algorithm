from functools import cmp_to_key
from typing import List

#删除被覆盖区间
class Solution:
    def removeCoveredIntervals(self, intervals: List[List[int]]) -> int:
        def cmp(t1, t2):
            if t1[0] != t2[0] :
                return t1[0] - t2[0]
            else:
                return t2[1] - t1[1]
        intervals = sorted(intervals, key=cmp_to_key(cmp))
        print(intervals)
        j = 0
        while j < len(intervals) -1:
            if intervals[j][0] == intervals[j+1][0]:
                del intervals[j+1]
            else:
                if intervals[j][1] >= intervals[j+1][1]:
                    del intervals[j+1]
                else:
                    j += 1
        print(intervals)
        return len(intervals)


if __name__ == "__main__":
    # intervals = [[1,4],[3,6],[2,8]]
    intervals = [[34335,39239],[15875,91969],[29673,66453],[53548,69161],[40618,93111]]
    sol = Solution()
    ret = sol.removeCoveredIntervals(intervals)
    print(ret)

