"""
给定文本串text和模式串pattern，从文本串text中找出模式串pattern第一次出现的位置。

例如：主串：a b a c a a b a c a b a c a b a a b b
模式串：a b a c a b
"""
class Solution(object):
    def KMP(self, text_strings, pattern_strings):
        i, j = 0, 0
        next = self.get_next(pattern_strings)
        while i < len(text_strings) and j < len(pattern_strings):
            # 当j为-1时,要移动的是i，当然j也要归0
            if j == -1 or text_strings[i] == pattern_strings[j]:
                i += 1
                j += 1
            else:
                # 不相等则回溯j的位置
                j = next[j]

        if j == len(pattern_strings):
            return i - j
        else:
            return -1

    def get_next(self, pattern_strings):
        next = [None for _ in range(len(pattern_strings))]
        next[0] = -1
        j = 0
        k = -1
        while j < len(pattern_strings) - 1:
            # next[j+1] == next[j] + 1, if p[k] == p[j]
            # k = next[k]， if p[k] != p[j] 不相等则利用next前面的计算结果跳转
            if k == -1 or pattern_strings[j] == pattern_strings[k]:
                j += 1; k +=1
                next[j] = k
            else:
                k = next[k]
        return next

if __name__ == '__main__':
    text = 'abacaabacabacabaabb'
    pattern = 'abacab'
    s = Solution()
    # print(s.KMP(text, pattern))
    print(s.get_next(pattern))
