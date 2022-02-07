# 3. 无重复字符的最长子串
class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        idx_dict = dict()
        cur_idx = 0
        max_len = 0
        for j, alpha in enumerate(s):
            if idx_dict.get(s[j]) is None:
                idx_dict[s[j]] = [j]
                max_len = max(max_len, j-cur_idx+1)
            else:
                pre_idx = idx_dict.get(s[j])[-1]
                if cur_idx > pre_idx:
                    max_len = max(max_len, j-cur_idx+1)
                    idx_dict[s[j]].append(j)
                else:
                    max_len = max(max_len, j-pre_idx)
                    cur_idx = pre_idx + 1
                    idx_dict[s[j]].append(j)
        return max_len