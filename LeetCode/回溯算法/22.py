# 22. 括号生成
from typing import List

class Solution:
    def generateParenthesis(self, n: int) -> List[str]:
        cur_list = [['(', 1, 0]]
        cur_list = self.tran(n, 1, cur_list)
        result = [n[0] for n in cur_list]
        return result

    def tran(self, n, i , cur_list):
        if i == 2 * n:
            return cur_list
        tmp = []
        for ele in cur_list:
            out, left, right = ele[0], ele[1], ele[2]
            if left == n:
                out += ')'
                right += 1
                tmp.append([out, left, right])
            elif left == right:
                out += "("
                left += 1
                tmp.append([out, left, right])
            elif left > right:
                tmp.append([out + "(", left + 1, right])
                tmp.append([out + ")", left, right+1])
        cur_list = tmp
        return self.tran(n , i+1, cur_list)

def testCase():
    n = 3
    sol = Solution()
    ret = sol.generateParenthesis(n)
    print(ret)

if __name__ == "__main__":
    testCase()


