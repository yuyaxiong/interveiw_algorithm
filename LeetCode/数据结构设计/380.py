
# 380. O(1) 时间插入、删除和获取随机元素
import random
class RandomizedSet:

    def __init__(self):
        self.nums = []
        self.valToIdx = dict()

    def insert(self, val: int) -> bool:
        if self.valToIdx.get(val) is not None:
            return False
        self.valToIdx[val] = len(self.nums)
        self.nums.append(val)
        return True

    def remove(self, val: int) -> bool:
        if self.valToIdx.get(val) is None:
            return False
        
        idx = self.valToIdx.get(val)
        last_val = self.nums[-1]
        self.valToIdx[last_val] = idx
        self.nums[idx] = last_val

        self.nums.pop()
        del self.valToIdx[val]
        return True

    def getRandom(self) -> int:
        return self.nums[random.randint(0, len(self.nums)-1)]


# Your RandomizedSet object will be instantiated and called as such:
# obj = RandomizedSet()
# param_1 = obj.insert(val)
# param_2 = obj.remove(val)
# param_3 = obj.getRandom()

def testCase():
    rs_sol = RandomizedSet()
    # ret = [rs_sol.insert(1), rs_sol.remove(2), rs_sol.insert(2), rs_sol.getRandom(), rs_sol.remove(1), rs_sol.insert(2), rs_sol.getRandom()] 
    ret = [rs_sol.insert(0), rs_sol.insert(1), rs_sol.remove(0), rs_sol.insert(2), rs_sol.remove(1), rs_sol.getRandom()]
    # print(rs_sol.nums)
    # print(rs_sol.valToIdx)
    print(ret)



if __name__ == "__main__":
    testCase()
