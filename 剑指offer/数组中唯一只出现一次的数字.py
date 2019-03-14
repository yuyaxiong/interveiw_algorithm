"""
在一个数组中除了一个数字只出现一次之外，其他数字都出现了三次。
请找出那个只出现一次的数字。
"""
class Solution(object):
    def find_num_appearing_once(self, nList, length):
        if nList is None or length <= 0:
            return None
        tmp = []
        for n in nList:
            tmp.append(self.dec_to_bin(n))

        result = []
        for i in range(32):
            cumsum = 0
            for j in range(len(tmp)):
                # print(i, j)
                cumsum += tmp[j][i]
            result.append(cumsum)
        n = 0
        for i, m in enumerate(result[1:][::-1]):
            if m % 3 == 1:
                n += 2 ** i
        return result, n


    def dec_to_bin(self, n):
        binary = [0 for _ in range(32)]
        idx = len(binary)-1
        for i, s in enumerate(bin(n)[2:][::-1] if n >=0 else bin(n)[3:][::-1]):
            binary[idx - i] = int(s)
        binary[0] = 0 if n >= 0 else 1
        return binary


if __name__ == '__main__':
    s = Solution()
    # print(s.dec_to_bin())
    nList = [1,2,2,2,3,3,3,4,4,4,7,7,7]
    print(s.find_num_appearing_once(nList, len(nList)))
