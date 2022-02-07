import sys
sys.setrecursionlimit(100000) #例如这里设置为十万 

# 887. 鸡蛋掉落/810

class Solution:
    def superEggDrop(self, k: int, n: int) -> int:
        dp = [[0 for _ in range(n+1)] for _ in range(k+1)]
        m = 0
        while dp[k][m] < n:
            m += 1
            for i in range(1, k+1):
                dp[i][m] = dp[i][m-1] + dp[i-1][m-1] + 1
        return m




def testCase():
    k = 4 
    n = 2000
    sol = Solution()
    res = sol.superEggDrop(k, n)
    print(res)

if __name__ == "__main__":
    testCase()

