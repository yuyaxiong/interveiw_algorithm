from typing import List

# 四数之和
class Solution:
    def fourSum(self, nums: List[int], target: int) -> List[List[int]]:
        nums = sorted(nums)
        return self.nSums(target, 4, nums)

    def nSums(self, target, n, nums):
        if n == 2:
            return self.twoSum(target, nums)
        else:
            res_list = []
            for i, val in enumerate(nums):
                tmp_nums = nums[i+1:]
                if i >= 1 and nums[i] == nums[i-1]:
                    continue
                sub_list = self.nSums(target-val, n-1, tmp_nums)
                for sub in sub_list:
                    sub.append(val)
                    res_list.append(sub)
            return res_list

    def twoSum(self, target, nums):
        i, j = 0, len(nums)-1
        res_list = []
        while i < j:
            if nums[i] + nums[j] == target:
                res_list.append([nums[i], nums[j]])
                while i< j and nums[i] == nums[i+1]:
                    i += 1
                while i< j and nums[j] == nums[j-1]:
                    j -= 1
                i += 1
                j -= 1
            elif nums[i] + nums[j] < target:
                i += 1
            elif nums[i] + nums[j] > target:
                j -= 1
        return res_list

if __name__ == "__main__":
    nums = [1, 0, -1, 0, -2, 2]
    target = 0
    sol = Solution()
    ret = sol.fourSum(nums, target)
    # ret = sol.twoSum(0, sorted(nums))
    print(ret)