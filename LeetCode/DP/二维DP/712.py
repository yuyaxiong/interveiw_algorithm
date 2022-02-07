# 712. 两个字符串的最小ASCII删除和
class Solution:
    def minimumDeleteSum(self, s1: str, s2: str) -> int:
        self.memo = [[0 for _ in s2] for _ in s1]
        return self.dp(s1, 0, s2, 0)

    def dp(self, s1, i, s2, j):
        if i == len(s1) and j == len(s2):
            return 0
        elif i == len(s1) and j < len(s2):
            val = 0
            while j < len(s2):
                val += ord(s2[j])
                j += 1
            return val
        elif i < len(s1) and j == len(s2):
            val = 0 
            while i < len(s1):
                val += ord(s1[i])
                i += 1
            return val

        if self.memo[i][j] != 0:
            return self.memo[i][j]

        if s1[i] == s2[j]:
            self.memo[i][j] = self.dp(s1, i+1, s2, j+1)
        else:
            left = ord(s1[i]) + self.dp(s1, i+1, s2, j)
            right = ord(s2[j]) + self.dp(s1, i, s2, j+1)
            self.memo[i][j] = min(left, right)
        return self.memo[i][j]
