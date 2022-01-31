import sys
from typing import List
import sys

# 1631. 最小体力消耗路径
class State:
    def __init__(self, x, y, delta):
        self.x = x
        self.y = y
        self.delta = delta

class Solution:
    def minimumEffortPath(self, heights: List[List[int]]) -> int:
        if len(heights) == 1 and len(heights[0]) == 1:
            return 0
        row, col = len(heights), len(heights[0])
        graph = self.buildGraph(heights)
        pq = []
        delta_list = [[sys.maxsize for _ in range(col)] for _ in range(row)]

        delta_list[0][0] = 0
        pq.append(State(0, 0, 0))
        while len(pq) > 0:
            cur_state = pq.pop(0)
            if delta_list[cur_state.x][cur_state.y] < cur_state.delta:
                continue                
            for neighbor in graph[cur_state.x][cur_state.y]:
                next_x, next_y, height = neighbor[0], neighbor[1], neighbor[2]
                val = max(height, delta_list[cur_state.x][cur_state.y])
                if delta_list[next_x][next_y] > val:
                    delta_list[next_x][next_y] = val
                    state = State(next_x, next_y, val)
                    pq.append(state)
        return delta_list[-1][-1]

    def buildGraph(self, heights):
        graph = [[[] for _ in range(len(heights[0]))] for _ in range(len(heights))]
        for i in range(len(heights)):
            for j in range(len(heights[0])):
                val = heights[i][j]
                left, right, top, bot = None, None, None, None
                if j - 1 >= 0:
                    left = heights[i][j-1]
                    delta = abs(left - val)
                    graph[i][j].append([i, j-1, delta])
                if j + 1 < len(heights[0]):
                    right = heights[i][j+1]
                    delta = abs(right - val)
                    graph[i][j].append([i, j+1, delta])
                if i -1 >= 0:
                    top = heights[i-1][j]
                    delta = abs(top - val)
                    graph[i][j].append([i-1, j, delta])
                if i + 1 < len(heights):
                    bot = heights[i+1][j]
                    delta = abs(bot - val)
                    graph[i][j].append([i+1, j, delta])
        return graph


def testCase():
    heights = [[1,2,2],[3,8,2],[5,3,5]]
    sol = Solution()
    ret = sol.minimumEffortPath(heights)
    # assert ret == 2
    print(ret)


def testCase1():
    heights = [[3]]
    sol = Solution()
    ret = sol.minimumEffortPath(heights)
    assert ret == 0
    print(ret)

def testCase2():
    heights = [[1,10,6,7,9,10,4,9]]
    sol = Solution()
    ret = sol.minimumEffortPath(heights)
    # assert ret == 9
    print(ret)

if __name__ == "__main__":
    testCase()
    # testCase1()
    # testCase2()

