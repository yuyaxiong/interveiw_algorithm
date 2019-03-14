"""
数字在排序数组中出现的次数。
统计一个数字在排序数组中出现的次数。
例如：输入排序数组[1,2,3,3,3,3,4,5]和数字3，由于3在这个数组中出现了4次，因此输出4。
"""


class Solution1(object):
    def get_first_k(self,data, length, k, s, e):
        if s > e:
            return -1
        mid_idx = (s+e)//2
        mid_data = data[mid_idx]
        if mid_data == k:
            if (mid_idx > 0 and data[mid_idx - 1] != k) or mid_idx == 0:
                return mid_idx
            else:
                e = mid_idx -1
        elif mid_data > k:
            e = mid_idx -1
        else:
            s = mid_idx + 1
        return self.get_first_k(data, length, k, s, e)

    def get_last_k(self, data, length, k, s, e):
        if s > e:
            return -1
        mid_idx = (s + e)//2
        mid_data = data[mid_idx]

        if mid_data == k:
            if (mid_idx < length -1 and data[mid_idx+1] != k) or mid_idx == length -1:
                return mid_idx
            else:
                s = mid_idx + 1
        elif mid_data < k:
            s = mid_idx + 1
        else:
            e = mid_idx - 1
        return self.get_last_k(data, length, k, s, e)

    def get_num_of_k(self, data, length, k):
        num = 0
        if data is not None and length > 0:
            first = self.get_first_k(data, length, k, 0, length-1)
            last = self.get_last_k(data, length, k, 0, length-1)
            if first >- 1 and last > -1:
                num = last - first +1
        return num




if __name__ == '__main__':
    nList = [1, 2, 3, 3, 3, 3, 4, 5]
    s = Solution1()
    print(s.get_num_of_k(nList, len(nList),3))







