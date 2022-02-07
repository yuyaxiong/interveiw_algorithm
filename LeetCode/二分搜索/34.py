from typing import List

# 34. 在排序数组中查找元素的第一个和最后一个位置
class Solution:
    def searchRange(self, nums: List[int], target: int) -> List[int]:
        return self.findZone(nums, target, 0, len(nums)-1)

    def findZone(self, nums, target, start, end):
        if start > end:
            return [-1, -1]
        else:
            mid = int((start + end)/2)
            if nums[mid] == target:
                left = mid
                while left >= 0 and nums[left] == target :
                    left -= 1
                right = mid
                while right < len(nums) and  nums[right] == target:
                    right += 1
                return [left+1, right-1]
            elif nums[mid] > target:
                return self.findZone(nums, target, start, mid-1)
            elif nums[mid] < target:
                return self.findZone(nums, target, mid+1, end)



# 非递归的方式
class Solution:

    def searchRange(self, n_list, target):
        return [self.leftBound(n_list, target), self.rightBound(n_list, target)]

    def leftBound(self, n_list, target):
        low, hight = 0, len(n_list)-1
        while low <= hight:
            mid = (low + hight)//2
            if n_list[mid] > target:
                hight = mid -1
            elif n_list[mid] < target:
                low = mid + 1
            elif n_list[mid] == target:
                hight = mid -1
        if low >= len(n_list):
            return -1
        elif low < len(n_list):
            return  low if n_list[low] == target else -1

    def rightBound(self, n_list, target):
        low, hight = 0, len(n_list) - 1
        while low <= hight:
            mid = (low + hight)//2
            if n_list[mid] > target:
                hight = mid-1
            elif n_list[mid] < target:
                low = mid + 1
            elif n_list[mid] == target:
                low = mid + 1
        if hight < 0:
            return -1
        else:
            return hight if n_list[hight] == target else -1


def testCase():
    nums = [5,7,7,8,8,10]
    target = 8
    sol = Solution()
    ret = sol.searchRange(nums, target)
    print(ret)

def testCase1():
    nums = [5,7,7,8,8,10]
    target = 10
    sol = Solution()
    ret = sol.searchRange(nums, target)
    print(ret)

def testCase2():
    nums = [1]
    target = 1
    sol = Solution()
    ret = sol.searchRange(nums, target)
    print(ret)

if __name__ == "__main__":
    testCase1()
    testCase2()