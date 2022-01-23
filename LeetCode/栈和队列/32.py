
# 32. 最长有效括号
class Solution:
    def longestValidParentheses(self, s: str) -> int:
        left_idx_list = []
        dp = [0 for i in range(len(s) +1)]
        for i, a in enumerate(s):
            if a == "(":
                left_idx_list.append(i)
                dp[i+1] = 0
            elif a == ")":
                if len(left_idx_list) > 0:
                   left_idx = left_idx_list.pop()
                   length = i - left_idx + 1 + dp[left_idx]
                   dp[i+1] = length
                   i += 1
                else:
                    dp[i+1] = 0
        return max(dp)