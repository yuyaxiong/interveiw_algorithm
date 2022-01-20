from typing import List

#三数之和
class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        nums = sorted(nums)
        pair_list = []
        for idx, n in enumerate(nums):
            tmp_nums = nums[idx+1:]
            if nums[idx] > 0:
                break
            if idx >= 1 and nums[idx] == nums[idx-1]:
                continue
            pair_list.extend(self.twoSum(-n, tmp_nums))
        return pair_list

    def twoSum(self, target, nums):
        i, j = 0, len(nums)-1
        pair_list = []
        while i < j:
            if nums[i] + nums[j] > target:
                j -= 1
            elif nums[i] + nums[j] < target:
                i += 1
            elif nums[i] + nums[j] == target:
                pair_list.append([-target, nums[i], nums[j]])
                while i < j and nums[i] == nums[i+1]:
                    i += 1
                while i < j and nums[j] == nums[j-1]:
                    j -= 1
                i += 1
                j -= 1
        return pair_list



if __name__ == "__main__":
    # nums = [-1,0,1,2,-1,-4]
    nums = [-2,0,1,1,2]
    sol = Solution()
    # ret = sol.threeSum(nums)
    # nums = sorted(nums)
    # nums = [0, 0, 0]
    # ret = sol.twoSum(0, nums)
    ret = sol.threeSum(nums)
    print(ret)
    print("---------------------")
    # sol1 = Solution1()
    # ret1 = sol1.threeSum(nums)
    # # print(ret)
    # print(ret1)


