"""
在一个长度为N+1的数组里的所有数字都在1-n的范围内，所以数组中至少有一个数字是重复的。
请找出数组中任意一个重复的数字，但不能修改输入数组。例如，如果输入长度为8的数组
[2,3,5,4,3,2,6,7],那么对应的输出事重复的数字2或者3。
"""


class Solution(object):
    def find_duplicates(self, nList, begin, end):
        mid = (begin + end)// 2
        left_count, right_count = 0, 0
        for n in nList:
            if n <= mid and n > begin:
                left_count += 1
            elif n > mid and n <= end:
                right_count += 1
        if mid - begin == 1 and left_count > 1:
            return mid
        elif end - mid == 1 and right_count > 1:
            return end
        if left_count > mid:
            end = mid
        else:
            begin = mid
        return self.find_duplicates(nList, begin, end)

if __name__ == '__main__':
    s = Solution()
    nList = [2,3,5,4,3,2,6,7]
    print(s.find_duplicates(nList, begin=0, end=len(nList)-1))