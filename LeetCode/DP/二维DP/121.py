# 121. 买卖股票的最佳时机
from typing import List


class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        delta_list = []
        for p1, p2 in zip(prices[:-1], prices[1:]):
            delta_list.append(p2-p1)
        cum_val = 0
        max_val = 0
        for delta in delta_list:
            if cum_val + delta > 0:
                cum_val += delta
                max_val = max(cum_val, max_val)
            else:
                cum_val = 0
        return max_val
        

# DP的方式
class Solution1:
    def maxProfit(self, prices: List[int]) -> int:
        dp = [[0, 0] for _ in prices]
        # 卖出
        dp[0][0] = 0
        # 买入
        dp[0][1] = -prices[0]
        for i in range(1, len(prices)):
            # 卖出   =  max(保持 ，  前一天买入 + 当天卖出  )      
            dp[i][0] = max(dp[i-1][0], dp[i-1][1] + prices[i])
            # 买入 =    max( 保持,  当天买入）
            dp[i][1] = max(dp[i-1][1], -prices[i])
        return dp[-1][0]

