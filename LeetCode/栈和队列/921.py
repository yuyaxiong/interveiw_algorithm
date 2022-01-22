#921. 使括号有效的最少添加

# 骚操作
class Solution:
    def minAddToMakeValid(self, s: str) -> int:
        while  "()" in s :
            s = s.replace("()", "")
        return len(s)

# 正常版本
class Solution1:
    def minAddToMakeValid(self, s: str) -> int:
        left = []
        for a in s:
            if a == "(":
                left.append(a)
            elif a == ")":
                if len(left) > 0:
                    if left[-1] == "(":
                        left.pop()
                    else:
                        left.append(a)
                else:
                    left.append(a)
        return len(left)

