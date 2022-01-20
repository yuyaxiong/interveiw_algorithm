from typing import List


class Solution:
    def maxArea(self, height: List[int]) -> int:
        i, j = 0, len(height)-1
        area = 0
        while i < j:
            cur_area = min(height[i], height[j]) * (j - i)
            area = max(cur_area, area)
            if height[i] < height[j]:
                i += 1
            else:
                j -= 1
        return area

height = [1,8,6,2,5,4,8,3,7]
sol = Solution()
assert 49 == sol.maxArea(height)
