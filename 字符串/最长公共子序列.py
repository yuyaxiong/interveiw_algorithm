"""
一个给定序列的子序列就是该给定序列中去掉零个或者多个元素的序列。
在两个字符串中，有些字符会一样，可以形成的子序列也有可能相等,因此，长度最长的相等子序列
便是两者间的最长公共子序列。
例如：S1='13456778',S2='357486782'
https://blog.csdn.net/someone_and_anyone/article/details/81044153
"""
# 动态规划
class Solution(object):
    """
    dp[i][j] = 0 , if i == 0 and j == 0
               dp[i-1][j-1] +1 , if i >0 and j >0 and s1[i] == s2[j]
               max(dp[i][j-1], dp[i-1][j]), if i>0 and j >0 and s1[i] != s2[j]
    """

    def longest_common_strings(self, S1, S2):
        s1_length, s2_length = len(S1), len(S2)
        dp = [[0 for _ in range(s1_length+1)] for _ in range(s2_length+1)]
        for i in range(1, s2_length+1):
            for j in range(1, s1_length+1):
                if S2[i-1] == S1[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        # 找出最后其中某一个子序列
        j,i = s1_length, s2_length
        result = []
        while i > 0 and j > 1:
            while dp[i][j]-1 != dp[i-1][j-1]:
                if j > 1:
                    j -= 1
                else:
                    break
            result.append(S1[j-1])
            j = j-1
            i = i-1

        return result[::-1]

if __name__ == '__main__':
    s = Solution()
    S1 = '13456778'
    S2 = '357486782'
    print(s.longest_common_strings(S1, S2))


