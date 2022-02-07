
# 583. 两个字符串的删除操作
class Solution:
    def minDistance(self, word1: str, word2: str) -> int:
        self.memo = [[-1 for _ in range(len(word2))] for _ in range(len(word1))]
        res = self.dp(word1, 0, word2, 0)
        print(self.memo)
        return res

    def dp(self, word1, i, word2, j):
        if i == len(word1) and j == len(word2):
            print("mark1")
            return 0
        elif i == len(word1) and j < len(word2):
            return len(word2) - j
        elif i < len(word1) and j == len(word2):
            return len(word1) - i

        if self.memo[i][j] != -1:
            return self.memo[i][j]
        if word1[i] == word2[j]:
            self.memo[i][j] = self.dp(word1, i+1, word2, j+1)
        else:
            self.memo[i][j] = min(self.dp(word1, i+1, word2, j), self.dp(word1, i, word2, j+1)) + 1

        return self.memo[i][j]

if __name__ == "__main__":
    word1 = "sea"
    word2 = "eat"  
    sol = Solution()
    ret = sol.minDistance(word1, word2)
    print(ret)