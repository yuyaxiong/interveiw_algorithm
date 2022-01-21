from typing import List

# 用最少数量的箭引爆气球
class Solution:
    def findMinArrowShots(self, points: List[List[int]]) -> int:
        if len(points) <= 0:
            return 0
        points = sorted(points, key=lambda x: x[1])
        counter = 0
        i =  0
        piv = points[i]
        while i < len(points):
            counter += 1
            while i < len(points) and piv[1] >= points[i][0]:
                i += 1
            if i < len(points):
                piv = points[i]
            else:
                piv = None
        return counter if piv is None else counter + 1
        
if __name__ == "__main__":
    points = [[1,2]]
    sol = Solution()
    ret = sol.findMinArrowShots(points)
    print(ret)