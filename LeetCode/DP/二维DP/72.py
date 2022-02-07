# 72.编辑距离
class Solution:
    def minDistance(self, word1: str, word2: str) -> int:
        m , n = len(word1), len(word2)
        # m * n 的数组保存的是，对应的word1[:i] 字符串和 对应的word2[:j]字符串的编辑距离
        dp = [[0 for _ in range(n+1)] for _ in range(m+1)]
        for i in range(1, m+1):
            dp[i][0] = i
        for j in range(1, n+1):
            dp[0][j] = j
        for i in range(1, m+1):
            for j in range(1, n+1):
                if word1[i-1] == word2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    # i-1, j 删除; i,j-1 插入； i-1,j-1 替换
                    # i-1, j 删除，是指：word1[:i]删除 对应的i的字符，所以操作+1 并取上一步的编辑距离
                    # i,j-1 插入是指，插入当前word1[i]的位置字符为对应Word2[j] 所以操作+1,插入之前的word1的字符串保持不变，
                    # 相当于插入到之前字符串的i+1的位置，因此取上一步的编辑距离，i,j-1
                    # i-1, j-1 替换操作，则+1，然后直接取上一步的编辑距离。
                    dp[i][j] = min(dp[i-1][j]+1, min(dp[i][j-1]+1, dp[i-1][j-1] + 1))
        return dp[m][n]



