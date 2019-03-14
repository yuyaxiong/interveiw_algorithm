"""
定义栈的数据结构，请在该类型中实现一个能够得到栈的最小元素是min函数。
在该栈中，调用min、push、pop的时间复杂度是O(1)
"""

class Stack(object):
    def __init__(self):
        self.stack = []
        self.min_value = None
        self.min_list = []

    def push(self,n):
        if self.min_value is None:
            self.min_value = n
            self.min_list.append(n)
        else:
            self.min_value = min(self.min_value, n)
            self.min_list.append(self.min_value)
        self.stack.append(n)

    def pop(self):
        self.stack.pop()
        self.min_list.pop()
        self.min_value = self.min_list[-1]

    def min(self):
        return self.min_value

if __name__ == '__main__':
     s=Stack()
     s.push(3)
     s.push(2)
     s.push(1)
     print(s.min())
     s.pop()
     print(s.min())
     s.pop()
     print(s.min())