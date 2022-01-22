
# 370.区间加法
from typing import List
from winreg import REG_RESOURCE_LIST


class Solution:
    def zoneAdd(self, updates: List[List[int]], length: int) -> int:
        # max_val = max([item[1] for item in updates])
        delta_list = [0 for _ in range(length)]
        for item in updates:
            start, end , inc = item[0], item[1], item[2]
            delta_list[start] += inc
            if end + 1 < len(delta_list):
                delta_list[end+1] -= inc
        res_list = []
        for delta in delta_list:
            val = res_list[-1] + delta if len(res_list) > 0 else delta
            res_list.append(val)
        return res_list


if __name__ == "__main__":
    updates = [[1,3,2],[2,4,3],[0,2,-2]]
    length = 5
    s = Solution()
    ret = s.zoneAdd(updates, length)
    print(ret)
    
    