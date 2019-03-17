"""
编辑距离是指两个字符串之间，由一个转成另一个所需的最小编辑操作次数。
许可的编辑操作包括将一个字符替换成另一个字符，插入一个字符，删除一个字符。
例如：对于字符串"if"和"iff"，可以通过插入一个"f"或者删除一个"f"来达到目的。
给定两个字符串A和B，求字符串A至少经过多少步字符操作编程字符串B。
例如S1='michaelxy', S2='michaelab'
https://blog.csdn.net/baodream/article/details/80417695
"""
# 动态规划
class Solution(object):
    """
    dp[i][j] = 0, if i ==0 and j == 0
               j, if i == 0 and j>0
               i, if j == 0 and i>0
               min(dp[i-1][j]+1, dp[i][j-1]+1,dp[i-1][j-1] + flag), if i>0 and j>0
    其中：
    flag = 0 , if S1[i] == S2[j]
           1,  if S1[i] != S2[j]

    删除，插入操作都是: min(dp[i-1][j]+1, dp[i][j-1]+1)
    替换操作是：dp[i-1][j-1] + flag
    """
    def edit_distance(self, S1, S2):
        s1_length = len(S1)
        s2_length = len(S2)
        dp = [[0 for _ in range(s1_length+1)] for _ in range(s2_length+1)]
        dp[0] = [i for i in range(s1_length+1)]
        for i in range(s2_length+1):
            dp[i][0] = i

        for i in range(1, s1_length+1):
            for j in range(1, s2_length+1):
                flag = 0 if S1[i-1] == S2[j-1] else 1
                dp[i][j] = min(dp[i-1][j] + 1, dp[i][j-1] + 1, dp[i-1][j-1] + flag)

        return min(dp[-1][1:])



if __name__ == '__main__':
    s = Solution()
    S1 = 'michaelxy'
    S2 = 'michaelab'
    print(s.edit_distance(S1, S2))

