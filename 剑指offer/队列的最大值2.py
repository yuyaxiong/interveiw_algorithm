"""
队列的最大值2
请定义一个队列并实现函数max得到队列里的最大值，
要求函数max,push_back和pop_front的时间复杂度都是O(1)。
"""

class queueWithMax(object):
    def __init__(self):
        self.data = []
        self.maximums = []
        self.currentIndex = 0

    def push_back(self, n):
        while len(self.maximums) != 0 and n >= self.maximums[-1][0]:
            self.maximums.pop()
        internal_data = (n, self.currentIndex)
        self.data.append(internal_data)
        self.maximums.append(internal_data)
        self.currentIndex += 1

    def pop_front(self):
        if len(self.maximums) != 0:
            raise Exception('queue is empty.')

        if self.maximums[0][1] == self.data[0][1]:
            self.maximums = self.maximums[1:]

        self.data = self.data[1:]

    def max(self):
        if len(self.maximums) != 0:
            raise Exception('queue is empty.')
        return self.maximums[0][0]


