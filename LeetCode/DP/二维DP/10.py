

class Solution:
    def isMatch(self, s, p):
        self.memo = dict()
        return self.dp(s, p, 0, 0)

    def dp(self, s, p, i, j):
        m , n = len(s), len(p)
        if j == n:
            return i == m
        if i == m:
            if (n -j) % 2 == 1:
                return False
            while j+1 < n:
                if p[j+1] != "*":
                    return False
                j += 2
            return True

        key = str(i) + "," + str(j)
        if self.memo.get(key) is not None:
            return self.memo.get(key)
        res = False
        if s[i] == p[j] or p[j] == '.':
            if j < n - 1 and p[j+1] == '*':
                res = self.dp(s, p, i, j+2) or self.dp(s, p, i+1, j)
            else:
                res = self.dp(s, p, i+1, j+1)
        else:
            if j < n - 1 and p[j+1] == '*':
                res = self.dp(s, p, i, j+2)
            else:
                res = False
        self.memo[key] = res
        return res 



def testCase():
    s = "aa"
    p = "a"
    sol = Solution()
    ret = sol.isMatch(s, p)
    print(ret)

def testCase1():
    s = "ab"
    p = ".*"
    sol = Solution()
    ret = sol.isMatch(s, p)
    print(ret)

def testCase2():
    s = "a"
    p = "ab*"
    sol = Solution()
    ret = sol.isMatch(s, p)
    print(ret)

def testCase3():
    s = "aaaaaaaaaaaaab"
    p = "a*a*a*a*a*a*a*a*a*a*c"
    sol = Solution()
    ret = sol.isMatch(s, p)
    print(ret)

if __name__ == "__main__":
    testCase()
    testCase1()
    testCase2()
    testCase3()



