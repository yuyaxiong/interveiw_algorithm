# 1541.平衡括号字符串的最少插入次数
class Solution:
    def minInsertions(self, s: str) -> int:
        left = []
        counter = 0
        i = 0
        while i < len(s):
            if s[i] == "(":
                left.append(s[i])
                i += 1
            elif s[i] == ")":
                if len(left) > 0:
                    left.pop()
                    if i+1 < len(s):
                        if s[i+1] == "(":
                            counter += 1
                            i += 1
                        elif s[i+1] == ")":
                            i += 2
                    else:
                        counter += 1
                        i += 1
                else:
                    counter += 1
                    if i+1 < len(s):
                        if s[i+1] == ")":
                            i += 2
                        elif s[i+1] == "(":
                            counter += 1
                            i += 1
                    else:
                        counter += 1
                        i += 1
        
        return counter + len(left) * 2


def is_exists(s):
    while "())" in s:
        s = s.replace("())", "")
        print(s)


if __name__ == "__main__":
    # s = "()())))()"
    s = "(()))(()))()())))"
    # s = "((())))))"
    # s = "()"  

    sol = Solution1()
    ret = sol.minInsertions(s)
    print(ret)
    print(is_exists(s))


