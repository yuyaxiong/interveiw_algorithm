"""
滑动窗口的最大值
给定一个数组和滑动窗口的大小，请找出所有滑动窗口里的最大值。
例如：如果输入数组[2,3,4,3,6,3,5,1]及滑动窗口的大小3， 那么
一共存在6个滑动窗口，他们的最大值分别为[4,4,6,6,6,5]
"""

class Solution(object):
    def max_in_windows(self, nList, num):
        # 初始化
        queue, max_windows = [], []
        current_idx = nList[0]
        for i, n in enumerate(nList[1:]):
            if i < num-1 and nList[current_idx] < n:
                current_idx = i
        queue.append(current_idx)
        max_windows.append(nList[current_idx])
        # 程序部分
        for i in range(current_idx+1, len(nList)):
            begin, end = queue[0], queue[-1]
            if i - begin >= num:
                queue = queue[1:]

            if nList[end] > nList[i]:
                queue.append(i)
            else:
                while len(queue) != 0 and nList[queue[-1]] < nList[i]:
                    queue = queue.pop()
                queue.append(i)
            max_windows.append(nList[queue[0]])
        return max_windows


if __name__ == "__main__":
    s = Solution()
    nList = [2,3,4,3,6,3,5,1]
    print(s.max_in_windows(nList, 3))


