#接雨水
from typing import List


class Solution:
    def trap(self, height: List[int]) -> int:
        if len(height) <= 2:
            return 0
        right_idx_list = sorted(range(len(height)), key=lambda i: height[i], reverse=True)
        left_idx_list = []
        water = 0
        right_idx_list.remove(0)
        for i in range(1, len(height)-1):
            left_idx_list = self.update_idx_list(left_idx_list, i-1, height)
            right_idx_list.remove(i)
            left_max, right_max = height[left_idx_list[0]], height[right_idx_list[0]]
            h = min(left_max, right_max) - height[i]
            print("left:", left_idx_list, "right:", right_idx_list, "cur_idx:", i)
            if h > 0:

                water += h
        return water

    def update_idx_list(self, idx_list, idx, height):
        if len(idx_list) == 0:
            idx_list.append(idx)
            return idx_list
        for i, cur_idx in enumerate(idx_list):
            if height[cur_idx] < height[idx]:
                idx_list.insert(i, idx)
                break
        return idx_list


class Solution1:
    def trap(self, height: List[int]) -> int:
        if len(height) <= 2:
            return 0
        water = 0
        for i in range(1, len(height)-1):
            h = min(max(height[:i]), max(height[i+1:])) - height[i]
            if h > 0:
                water += h
        return water

class Solution2:
    def trap(self, height: List[int]) -> int:
        ans = 0
        h1 = 0
        h2 = 0
        for i in range(len(height)):
            h1 = max(h1,height[i])
            h2 = max(h2,height[-i-1])
            ans = ans + h1 + h2 -height[i]
        return  ans - len(height)*h1

if __name__ == "__main__":
    height = [0,1,0,2,1,0,1,3,2,1,2,1]
    sol = Solution()
    ret = sol.trap(height)
    ret1 = sol.trap(height)
    print(ret)
    print(ret1)

