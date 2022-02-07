# 188. 买卖股票的最佳时机 IV
from typing import List


class Solution:
    def maxProfit(self, k: int, prices: List[int]) -> int:
        n = len(prices)
        if n <= 0:
            return 0

        if k > n/2:
            return self.profitNoLimit(prices) 

        dp = [[[0, 0] for _ in range(k+1)] for _ in prices]
        for i in range(k, 0, -1):
            dp[0][i][0] = 0
            dp[0][i][1] = -prices[0]
        for i in range(1, len(prices)):
            for j in range(k, 0, -1):
                dp[i][j][0] = max(dp[i-1][j][0], dp[i-1][j][1] + prices[i])
                dp[i][j][1] = max(dp[i-1][j][1], dp[i-1][j-1][0] - prices[i])
        return dp[-1][k][0]

    def profitNoLimit(self, prices):
        dp = [[0, 0] for _ in prices]
        dp[0][0] = 0
        dp[0][1] = -prices[0]
        for i in range(1, len(prices)):
            dp[i][0] = max(dp[i-1][0], dp[i-1][1] + prices[i])
            dp[i][1] = max(dp[i-1][1], dp[i-1][0] - prices[i])
        return dp[-1][0]



