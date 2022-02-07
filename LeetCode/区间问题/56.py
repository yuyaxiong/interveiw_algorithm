# 56.合并区间
from functools import cmp_to_key
from typing import List

class Solution:
    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        def cmp(t1, t2):
            if t1[0] != t2[0]:
                return t1[0] - t2[0]
            else:
                return t2[1] - t1[1]
        intervals = sorted(intervals, key=cmp_to_key(cmp))
        i = 0
        while i < len(intervals) -1:
            if intervals[i][0] == intervals[i+1][0]:
                del intervals[i+1]
            else:
                if intervals[i][1] >= intervals[i+1][1]:
                    del intervals[i+1]
                else:
                    if intervals[i][1] < intervals[i+1][0]:
                        i += 1
                    elif intervals[i][1] < intervals[i+1][1]:
                        intervals[i] = [intervals[i][0], intervals[i+1][1]]
                        del intervals[i+1]
        return intervals
