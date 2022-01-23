from collections import deque

class MyQueue:

    def __init__(self):
        self.stack = deque()
        self.top = None

    def push(self, x: int) -> None:
        if len(self.stack) == 0:
            self.top = x
        self.stack.append(x)

    def pop(self) -> int:
        if len(self.stack) == 0:
            return None
        stack_bak = deque()
        while len(self.stack) > 0:
            stack_bak.append(self.stack.pop())
        pop_val = stack_bak.pop()
        
        if len(stack_bak) > 0:
            self.top = stack_bak.pop()
            self.stack.append(self.top)
            while len(stack_bak) > 0:
                self.stack.append(stack_bak.pop())
        else:
            self.top = None
        return pop_val

    def peek(self) -> int:
        return self.top


    def empty(self) -> bool:
        return len(self.stack) == 0



# Your MyQueue object will be instantiated and called as such:
# obj = MyQueue()
# obj.push(x)
# param_2 = obj.pop()
# param_3 = obj.peek()
# param_4 = obj.empty()