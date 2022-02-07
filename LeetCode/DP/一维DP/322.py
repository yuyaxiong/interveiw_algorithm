# 322. 零钱兑换
import sys
from typing import List
class Solution:
    def coinChange(self, coins: List[int], amount: int) -> int:
        if amount == 0:
            return 0
        dp = [sys.maxsize for _ in range(amount+1)]
        for i in range(1, len(dp)):
            for coin in coins:
                if i == coin:
                    dp[i] = 1
                else:
                    if i > coin:
                        dp[i] = min(dp[i-coin] + 1, dp[i])
        return -1 if dp[-1] == sys.maxsize else dp[-1]
            