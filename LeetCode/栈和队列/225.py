import queue


def print_queue(que):
    n_list = []
    while not que.empty():
        n_list.append(que.get())
    print(n_list)

class MyStack:

    def __init__(self):
        self.que = queue.Queue()
        self.top_num = None

    def push(self, x: int) -> None:
        self.que.put(x)
        self.top_num = x

    def pop(self) -> int:
        rev_que = queue.Queue()
        q_size = self.que.qsize()
        if q_size > 2:
            while q_size > 2:
                cur = self.que.get()
                rev_que.put(cur)
                q_size -= 1
            self.top_num = self.que.get()
            pop_val = self.que.get()

            rev_que.put(self.top_num)
            self.que = rev_que
        elif q_size == 2:
            self.top_num = self.que.get()
            pop_val = self.que.get()
            self.que.put(self.top_num)
        elif q_size == 1:
            self.top_num = None
            pop_val = self.que.get()
        elif q_size == 0:
            pop_val = None
        return pop_val

    def top(self) -> int:
        return self.top_num

    def empty(self) -> bool:
        return self.que.empty()


# Your MyStack object will be instantiated and called as such:
# obj = MyStack()
# obj.push(x)
# param_2 = obj.pop()
# param_3 = obj.top()
# param_4 = obj.empty()

def testCase():
    obj = MyStack()
    obj.push(1)
    obj.push(2)
    obj.push(3)
    param2 = obj.pop()
    param3 = obj.pop()
    param4 = obj.pop()
    param5 = obj.empty()
    print(param2)
    print(param3)
    print(param4)
    print(param5)

def testCase1():
    obj = MyStack()
    obj.push(1)
    obj.push(2)
    param2 = obj.top()
    param3 = obj.pop()
    param4 = obj.empty()
    print(param2)
    print(param3)
    print(param4)



if __name__ == "__main__":
    testCase1()
