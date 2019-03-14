"""
数组中只出现一次的两个数字。
一个整型数组里除两个数字之外，其他数字都出现了两次。请写程序找出这两个只出现一次的数字。
"""

class Solution(object):
    def find_nums_appear_once(self, data):
        xor = data[0]
        for n in data[1:]:
            xor ^= n
        first1 = 0
        for i, n in enumerate(bin(xor)[2:][::-1]):
            if n == '1':
                first1 = -i-1
                break
        left = list(filter(lambda x: self.is_left(x, first1), data))
        right = list(filter(lambda x: not self.is_left(x, first1), data))
        ln = left[0]
        rn = right[0]
        for n in left[1:]:
            ln ^= n
        for n in right[1:]:
            rn ^= n
        return ln, rn

    def is_left(self, n, idx):
        n_binary = bin(n)[2:] if n >0 else bin(n)[3:]
        if n_binary[idx] == '1':
            return True
        else:
            return False


if __name__ == '__main__':
    s = Solution()
    data = [2, 4, 3, 6, 3, 2, 5, 5]
    print(s.find_nums_appear_once(data))

