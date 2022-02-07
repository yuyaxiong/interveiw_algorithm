# 986.区间列表的交集
from typing import List


class Solution:
    def intervalIntersection(self, firstList: List[List[int]], secondList: List[List[int]]) -> List[List[int]]:
        i,j = 0, 0
        result_list = []
        while i < len(firstList) and j < len(secondList):
            if firstList[i][0] > secondList[j][0]:
                if firstList[i][0] > secondList[j][1]:
                    j += 1
                else:
                    if firstList[i][1] < secondList[j][1]:
                        result_list.append([firstList[i][0], firstList[i][1]])
                        i += 1
                    else:
                        result_list.append([firstList[i][0], secondList[j][1]])
                        j += 1
            else:
                if secondList[j][0] > firstList[i][1]:
                    i += 1
                else:
                    if secondList[j][1] > firstList[i][1]:
                        result_list.append([secondList[j][0], firstList[i][1]])
                        i += 1
                    else:
                        result_list.append([secondList[j][0], secondList[j][1]])
                        j += 1
        return result_list
