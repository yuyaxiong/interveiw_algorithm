# 移除元素
class Solution:
    def removeElement(self, nums: List[int], val: int) -> int:
        s_p = 0
        for f_p, n in enumerate(nums):
            if nums[f_p] != val:
                nums[s_p] = nums[f_p]
                s_p += 1
        return s_p
