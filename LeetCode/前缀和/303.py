from typing import List

# Your NumArray object will be instantiated and called as such:
# obj = NumArray(nums)
# param_1 = obj.sumRange(left,right)
# 303.区域和检索-数组不可变
class NumArray:
    
    def __init__(self, nums: List[int]):
        self.acc_list = []
        acc = 0
        for n in nums:
            acc += n
            self.acc_list.append(acc)
        
    def sumRange(self, left: int, right: int) -> int:
        if left > 0:
            return self.acc_list[right] - self.acc_list[left-1]
        else:
            return self.acc_list[right]

    
