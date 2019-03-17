"""
给定一个字符串，求它的最长回文子串
回文串就是正反读都一样的字符串，例如："abba","aba"
例如："abbbcdcbbaaae" 则最长回文子串是"bbcdcbb"

https://www.jianshu.com/p/c82cada7e5b0
"""
# 中心扩展法 O(n^2)
class Solution(object):
    def longest_parlindrome(self, strings):
        length = len(strings)
        max_len = 1
        s = 0
        # 长度为奇数的最长回文子串
        for i in range(length):
            j, k = i-1, i+1
            while j >= 0 and k < length and strings[j] == strings[k]:
                if k -j +1 > max_len:
                    max_len = k -j + 1
                    s = j
                j -= 1
                k += 1
        # 长度为偶数的最长回文子串
        for i in range(length):
            j, k = i, i+1
            while j >= 0 and k <length and strings[j] == strings[k]:
                if k - j + 1 > max_len:
                    max_len = k - j + 1
                    s = j
                j -= 1
                k += 1
        return strings[s: s+ max_len], max_len


# 动态规划
"""
dp[j][i] = True if j == i
           strings[i] == strings[j]  if i-j == 1
           strings[i] == strings[j] and dp[j+1][i-1] if  i-j > 1
"""
class Solution1(object):
    def longest_parlindrome(self, strings):
        length = len(strings)
        dp = [[0 for _ in range(length)] for _ in range(length)]
        max_len = 1
        s = 0
        # i 为字符串的尾指针
        for i in range(length):
            # j为字符串的头指针
            for j in range(0, i+1):
                # 这个是因为 dp[j][i] == dp[j+1][i-1] if strings[j] == strings[i]
                if i-j < 2:
                    dp[j][i] = (strings[j] == strings[i])
                else:
                    dp[j][i] = (strings[i] == strings[j]) and dp[j+1][i-1]

                if dp[j][i] and max_len < i - j +1:
                    max_len = i - j + 1
                    s = j

        return strings[s: s + max_len]


# Manacher
# https://www.cnblogs.com/grandyang/p/4475985.html
# 链接是唯一看懂的一个说明，然后在博主的代码基础上稍微改了一点。
class Solution2(object):
    def longest_parlindrome(self, strings):
        t = '$#'
        for s in strings:
            t += s
            t += '#'
        # p 中保存的是以当前为中心的最长回文子串的长度
        p = [0 for _ in range(len(t))]
        mx, idx = 0, 0 # idx 是字符串的中心idx, mx是Max的length
        result_length, result_center = 0, 0
        for i in range(0, len(t)):
            # j = 2 * idx - 1
            # 当mx > i 由前面求得的P[j]知识来推测后面的P[i]知识
            p[i] = min(p[2*idx - 1] if 2*idx - 1 >= 0 and 2*idx - 1 < len(t) else 1, mx - i) if mx > i else 1
            # 当 mx <= i 的时候，没有先验知识，只能老老实实的匹配
            while i+p[i] < len(t) and i-p[i] >= 0 and t[i + p[i]] == t[i - p[i]]:
                p[i] += 1

            # 更新 idx, mx， result_length, result_center
            if mx < i + p[i]:
                mx = i + p[i]
                idx = i
            if result_length < p[i]:
                result_length = p[i]
                result_center = i
        # 计算这个同样很关键
        s = (result_center - result_length)/2
        length = result_length -1
        return strings[int(s): int(s) + length]







if __name__ == '__main__':
    s = Solution()
    s1 = Solution1()
    s2 = Solution2()
    print(s2.longest_parlindrome(strings='abbbcdcbbaaae'))
    print(s1.longest_parlindrome(strings='abbbcdcbbaaae'))
    print(s.longest_parlindrome(strings='abbbcdcbbaaae'))