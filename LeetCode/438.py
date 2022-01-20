from collections import defaultdict
class Solution:
    def findAnagrams(self, s: str, p: str) -> List[int]:
        if len(s) < len(p):
            return []
        p_dict, s_dict = dict(), dict()
        left, right = 0, -1
        res_list = []
        for a in p:
            if p_dict.get(a) is None:
                p_dict[a] = 0
            p_dict[a] += 1
            right += 1
            if s_dict.get(s[right]) is None:
                s_dict[s[right]] = 0
            s_dict[s[right]] += 1
        if self.cmp_dict(p_dict, s_dict):
            res_list.append(left)
        right += 1
        while right < len(s):
            if s_dict.get(s[right]) is None:
                s_dict[s[right]] = 0
            s_dict[s[right]] += 1
            if s_dict.get(s[left]) is None:
                s_dict[s[left]] = 0
            s_dict[s[left]] -= 1
            left += 1
            if self.cmp_dict(p_dict, s_dict):
                res_list.append(left)
            right += 1
        return res_list

    def cmp_dict(self, p1_dict, p2_dict):
        for k, v in p1_dict.items():
            if v == 0:
                continue
            if p2_dict.get(k) != v:
                return False
        for k, v in p2_dict.items():
            if v == 0:
                continue
            if p1_dict.get(k) != v:
                return False
        return True

s = "cbaebabacd"
p = "abc"
slo = Solution()
res = slo.findAnagrams(s, p)
print(res)
print(s[6:9])