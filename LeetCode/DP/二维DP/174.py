
# 174. 地下城游戏
from typing import List

import sys
class Solution1:
    def calculateMinimumHP(self, dungeon: List[List[int]]) -> int:
        m , n = len(dungeon), len(dungeon[0])
        self.memo = [[-1 for _ in range(n)] for _ in range(m)]
        return self.dp(dungeon, 0, 0)

    def dp(self, dungeon, i, j):
        m, n = len(dungeon), len(dungeon[0])
        if i == m-1 and j == n-1:
            return 1 if dungeon[i][j] >= 0 else 1 - dungeon[i][j]

        if i == m or j == n:
            return sys.maxsize

        if self.memo[i][j] != -1:
            return self.memo[i][j]

        res = min(self.dp(dungeon, i, j+1), self.dp(dungeon, i+1, j)) - dungeon[i][j]
        
        self.memo[i][j] = 1 if res <= 0 else res

        return self.memo[i][j]
        
def testCase():
    dungeon = [[-2,-3,3],[-5,-10,1],[10,30,-5]]
    # dungeon = [[3,-20,30],[-3,4,0]]
    # print(du)
    sol = Solution1()
    ret = sol.calculateMinimumHP(dungeon)
    print(ret)

if __name__ == "__main__":
    testCase()

    