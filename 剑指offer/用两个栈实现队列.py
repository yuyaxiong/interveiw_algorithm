"""
用两个栈实现一个队列。队列的申明如下，请实现它的两个函数
appendTail和DeleteHead,分别完成在队列的尾部插入节点和
在队列的头部删除节点的功能。
"""


class Quene(object):
    def __init__(self):
        self.nList1 = []
        self.nList2 = []

    def append(self, n):
        self.nList1.append(n)

    def delete(self):
        if len(self.nList2) == 0:
            while len(self.nList1) != 0:
                self.nList2.append(self.nList1.pop())
        return self.nList2.pop()


if __name__ == "__main__":
    q = Quene()
    q.append(3)
    q.append(4)
    q.append(5)
    q.append(6)
    q.append(2)
    print(q.delete())
    print(q.delete())
