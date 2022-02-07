#122. 买卖股票的最佳时机 II
from typing import List

class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        delta_list = []
        for p1, p2 in zip(prices[:-1], prices[1:]):
            delta_list.append(p2-p1)
        val_list = list(filter(lambda x: x> 0 , delta_list))
        return sum(val_list)


class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        dp = [[0, 0] for _ in prices]
        dp[0][0] = 0
        dp[0][1] = -prices[0]
        for i in range(1, len(prices)):
            dp[i][0] = max(dp[i-1][0], dp[i-1][1] + prices[i])
            dp[i][1] = max(dp[i-1][1], dp[i-1][0] - prices[i])
        return dp[-1][0]