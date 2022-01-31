# 第一部分

## 1.1.二分搜索

### 1011. 在 D 天内送达包裹的能力

```python
from typing import List

# 1011. 在 D 天内送达包裹的能力
class Solution:
    def shipWithinDays(self, weights: List[int], days: int) -> int:
        max_val = sum(weights)
        min_val = max(weights)
        return self.carrayWeight(weights, min_val, max_val, days)

    def carrayWeight(self, weights, s, e, days):
        if s == e:
            return s
        mid = (s + e) // 2
        if self.carrayDays(weights, mid) > days:
            return self.carrayWeight(weights, mid + 1, e, days)
        else:
            return self.carrayWeight(weights, s, mid, days)

    def carrayDays(self, weights, limitWeight):
        days = 0
        cumWeight = 0
        for w in weights:
            if cumWeight + w > limitWeight:
                days += 1
                cumWeight = w
            elif cumWeight + w == limitWeight:
                days += 1
                cumWeight = 0
            else:
                cumWeight += w
        if cumWeight != 0:
            days += 1
        return days

if __name__ == "__main__":
    s = Solution()
    weights = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    limitWeight = 11
    print(s.carrayDays(weights, limitWeight))
```

## 1.2.滑动窗口

### 76. 最小覆盖子串

```python

from collections import defaultdict
class Solution(object):
    def minWindow(self, s, t):
        """
        :type s: str
        :type t: str
        :rtype: str
        """
        mem = defaultdict(int)
        for char in t:
            mem[char]+=1
        t_len = len(t)

        minLeft, minRight = 0,len(s)
        left = 0

        for right,char in enumerate(s):
            if mem[char]>0:
                t_len-=1
            mem[char]-=1

            if t_len==0:
                while mem[s[left]]<0:
                    mem[s[left]]+=1
                    left+=1

                if right-left<minRight-minLeft:
                    minLeft,minRight = left,right

                mem[s[left]]+=1
                t_len+=1
                left+=1
        return '' if minRight==len(s) else s[minLeft:minRight+1]
```

### 239.滑动窗口最大值

```python
from typing import List


class Solution:
    def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
        win, ret = [], []
        for i, v in enumerate(nums):
            if i >= k and win[0] <= i - k:
                win.pop(0)
            while win and nums[win[-1]] <= v:
                win.pop()
            win.append(i)
            if i >= k - 1:
                ret.append(nums[win[0]])
        return ret
```

### 438. 找到字符串中所有字母异位词

```python
# 438. 找到字符串中所有字母异位词
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
```

### 567.字符串的排列

```python
class Solution:
    def checkInclusion(self, s1: str, s2: str) -> bool:
        if len(s2) < len(s1):
            return False

        s2_dict, s1_dict = dict(), dict()
        left, right = 0, 0
        for s in s1:
            if s1_dict.get(s) is None:
                s1_dict[s] = 0
            s1_dict[s] += 1
            if s2_dict.get(s2[right]) is None:
                s2_dict[s2[right]] = 0
            s2_dict[s2[right]] += 1
            right += 1
        if self.cmp_dict(s1_dict, s2_dict):
            return True
        print(s2_dict)
        print(right)
        while right < len(s2):
            if s2_dict.get(s2[right]) is None:
                s2_dict[s2[right]] = 0
            s2_dict[s2[right]] += 1
            if s2_dict.get(s2[left]) is None:
                s2_dict[s2[left]] = 0
            s2_dict[s2[left]] -= 1
            left += 1
            if self.cmp_dict(s2_dict, s1_dict):
                return True
            right += 1
            print(s2_dict)
        return False

    def cmp_dict(self, t1_dict, t2_dict):
        for k, v in t1_dict.items():
            if v == 0:
                continue
            if t2_dict.get(k) != v:
                return False
        for k, v in t2_dict.items():
            if v == 0:
                continue
            if t1_dict.get(k) != v:
                return False
        return True

s1 = "adc"
s2 = "dcda"

sol = Solution()
res = sol.checkInclusion(s1, s2)
print(res)
```




### 20.有效的括号

```python
# 20.有效的括号
class Solution:
    def isValid(self, s: str) -> bool:
        n_list = []
        a_dict = {"(": ")", "[":"]", "{": "}"}
        status = True
        for a in s:
            if a not in [")", "}", "]"]:
                n_list.append(a)
            else:
                if len(n_list) > 0:
                    left = n_list.pop()
                    if a_dict.get(left) != a:
                        status = False
                        break
                else:
                    status = False
                    break
        if len(n_list) > 0:
            status = False
        return status
```
