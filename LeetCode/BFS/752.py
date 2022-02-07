# 752. 打开转盘锁
from typing import List

class Solution:
    def openLock(self, deadends: List[str], target: str) -> int:
        dead_dict = {d:1 for d in deadends}
        visited_dict = dict()
        q = []
        step = 0
        q.append(['0', '0', '0', '0'])
        visited_dict["0000"] = 1
        while len(q) > 0:
            sz = len(q)
            for i in range(sz):
                cur = q.pop(0)
                cur_str = ''.join(cur)
                if dead_dict.get(cur_str) is not None:
                    continue
                if cur_str == target:
                    return step
                
                for j in range(4):
                    up = self.plusOne(cur[::], j)
                    up_str = ''.join(up)
                    if visited_dict.get(up_str) is None:
                        q.append(up)
                        visited_dict[up_str] = 1

                    down = self.minusOne(cur[::], j)
                    down_str = ''.join(down)
                    if visited_dict.get(down_str) is None:
                        q.append(down)
                        visited_dict[down_str] = 1
            step += 1
        return -1

    def plusOne(self, s, j):
        if s[j] == '9':
            s[j] = '0'
        else:
            s[j] = str(int(s[j]) + 1)
        return s

    def minusOne(self, s, j):
        if s[j] == '0':
            s[j] = '9'
        else:
            s[j] = str(int(s[j]) -1)
        return s


def testCase():
    deadends = ["0201","0101","0102","1212","2002"]
    target = "0202"
    sol = Solution()
    ret = sol.openLock(deadends, target)
    print(ret)

if __name__ == "__main__":
    testCase()

