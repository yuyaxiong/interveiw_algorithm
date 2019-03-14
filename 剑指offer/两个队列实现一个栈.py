"""
两个队列实现一个栈

"""

from queue import Queue

class stack(object):
    def __init__(self):
        self.q1 = Queue()
        self.q2 = Queue()

    def add(self, n):
        self.q1.put(n)

    def delete(self):
        n = self.q1.get()
        while not self.q1.empty():
            self.q2.put(n)
            n = self.q1.get()
        while not self.q2.empty():
            self.q1.put(self.q2.get())

        return n

if __name__ == '__main__':
    s = stack()
    s.add(3)
    s.add(4)
    s.add(5)
    s.add(6)
    s.add(7)
    print(s.delete())