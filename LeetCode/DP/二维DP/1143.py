# 1143. 最长公共子序列
class Solution:
    def longestCommonSubsequence(self, text1: str, text2: str) -> int:
        self.memo =[[0 for _ in text2] for _ in text1]
        return self.dp(text1, 0, text2, 0)

    def dp(self, text1, i, text2, j):
        if i == len(text1) or j == len(text2):
            return 0
        if self.memo[i][j] != 0:
            return self.memo[i][j]
        if text1[i] == text2[j]:
            self.memo[i][j] = 1 + self.dp(text1, i+1, text2, j+1)
        else:
            self.memo[i][j] = max(self.dp(text1, i+1, text2, j), self.dp(text1, i, text2, j+1))
        return self.memo[i][j]