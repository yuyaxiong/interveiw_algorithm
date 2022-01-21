from typing import List

class Solution:
    def removeDuplicates(self, nums: List[int]) -> int:
        if len(nums) <= 0:
            return 0
        s_p = 1
        pre_val = nums[0]
        for f_p in range(1, len(nums)):
            if pre_val != nums[f_p]:
                nums[s_p]= nums[f_p]
                s_p += 1
            pre_val = nums[f_p]
        return s_p

if __name__ == "__main__":
    nums = [1,1,2,2,2,3,3]
    sol = Solution()
    ret = sol.removeDuplicates(nums)
    print(ret)