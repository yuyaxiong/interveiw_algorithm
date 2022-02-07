#123. 买卖股票的最佳时机 III
from typing import List


class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        max_k = 2
        dp = [ [[0, 0] for _ in range(max_k+1)] for _ in prices]
        for k in range(max_k, 0, -1):
            dp[0][k][0] = 0
            dp[0][k][1] = -prices[0]
        
        for i in range(1, len(prices)):
            for k in range(max_k, 0, -1):
                dp[i][k][0] = max(dp[i-1][k][0], dp[i-1][k][1] + prices[i])
                dp[i][k][1] = max(dp[i-1][k][1], dp[i-1][k-1][0] - prices[i])

        return dp[-1][max_k][0]




def testCase():
    prices = [3,3,5,0,0,3,1,4]
    sol = Solution()
    ret = sol.maxProfit(prices)
    print(ret)

def testCase1():
    prices = [1,2,4,2,5,7,2,4,9,0]
    sol = Solution()
    ret = sol.maxProfit(prices)
    print(ret)

if __name__ == "__main__":
    testCase1()
