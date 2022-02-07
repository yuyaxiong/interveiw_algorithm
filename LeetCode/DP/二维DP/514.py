# 514.自由之路
import sys

class Solution:
    def findRotateSteps(self, ring: str, key: str) -> int:
        m, n = len(ring), len(key)
        self.charToIndex = dict()
        self.memo = [[0 for _ in range(n)] for _ in range(m)]
        # 索引初始化
        for i in range(m):
            if self.charToIndex.get(ring[i]) is None:
                self.charToIndex[ring[i]] = [i]
            else:
                self.charToIndex[ring[i]].append(i)
        return self.dp(ring, 0, key, 0)

    def dp(self, ring, i, key, j):
        if j == len(key):
            return 0
        if self.memo[i][j] != 0:
            return self.memo[i][j]
        n = len(ring)
        res = sys.maxsize
        # 找到最小的字母
        for k in self.charToIndex.get(key[j]):
            delta = abs(k - i)
            # 正向和逆向都可以，找到最小的
            delta = min(delta, n - delta)
            subProblem = self.dp(ring, k, key, j+1)
            res = min(res, 1 + delta +subProblem)
        self.memo[i][j] = res
        return res

if __name__ == "__main__":
    rings = "godding"
    key = "godding"
    sol = Solution()
    ret = sol.findRotateSteps(rings, key)
    print(ret)
