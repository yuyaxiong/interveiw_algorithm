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



class Solution1:
    def minInsertions(self, s: str) -> int:
        while "())" in s:
            s = s.replace("())", "|")
        left = []
        i = 0
        counter = 0
        print(s)
        while i < len(s):
            if s[i] == "(":
                left.append(s[i])
                i += 1
            elif s[i] == ")":
                if len(left) > 0:
                    left.pop()
                    counter += 1
                    i += 1
                else:
                    if i + 1 < len(s):
                        if s[i+1] == ")": 
                            counter += 1
                            i += 2
                        elif s[i+1] == "(":
                            counter += 2
                            i += 1
                    else:
                        counter += 2
                        i += 1
            elif s[i] == "|":
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


