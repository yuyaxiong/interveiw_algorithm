class Solution:
    def subarraySum(self, nums: List[int], k: int) -> int:
        acc_list, acc_dict = self.getAccList(nums)
        base = acc_dict[k] if acc_dict.get(k) is not None else 0
        count = 0
        for n in acc_list[::-1]:
            acc_dict[n] -= 1
            if acc_dict.get(n - k) is not None and acc_dict.get(n-k) > 0:
                count += acc_dict.get(n-k)
        return count + base

    def getAccList(self, nums):
        acc_dict = {}
        acc_list = []
        for n in nums:
            val = acc_list[-1] + n if len(acc_list) > 0 else  n
            acc_list.append(val)
            if acc_dict.get(val) is None:
                acc_dict[val] = 1
            else:
                acc_dict[val] += 1
        return acc_list, acc_dict

