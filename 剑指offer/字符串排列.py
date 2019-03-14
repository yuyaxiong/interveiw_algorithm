"""
输入一个字符串，打印该字符串中字符的所有排列，
例如：输入字符串 abc, 则打印出由字符a,b,c所能排列出来的所有字符串
abc,acb,bac,bca,cab和cba
"""

class Solution(object):
    def permutation(self, strings):
        str_list = [s for s in strings]
        result = [str_list]
        return self.permutation_help(result, 0, len(str_list))


    def permutation_help(self, result, idx, end):
        if idx >= end-1:
            return result
        tmp1 = []
        for res in result:
            for j in range(idx, end):
                res1 = res[::]
                res1[idx], res1[j] = res1[j], res1[idx]
                if res1 not in tmp1:
                    tmp1.append(res1)
        return self.permutation_help(tmp1, idx+1, end)

if __name__ == '__main__':
    s = Solution()
    strings = 'abc'
    print(s.permutation(strings))

