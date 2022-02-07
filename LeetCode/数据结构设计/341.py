# """
# This is the interface that allows for creating nested lists.
# You should not implement it, or speculate about its implementation
# """

# 341. 扁平化嵌套列表迭代器
class NestedInteger:

   def isInteger(self) -> bool:
       """
       @return True if this NestedInteger holds a single integer, rather than a nested list.
       """

   def getInteger(self) -> int:
       """
       @return the single integer that this NestedInteger holds, if it holds a single integer
       Return None if this NestedInteger holds a nested list
       """

   def getList(self) -> [NestedInteger]:
       """
       @return the nested list that this NestedInteger holds, if it holds a nested list
       Return None if this NestedInteger holds a single integer
       """

class NestedIterator:
    def __init__(self, nestedList: [NestedInteger]):
        self.nested_list = nestedList
    
    def next(self) -> int:
        return self.nested_list.pop(0).getInteger()
    
    def hasNext(self) -> bool:
        while len(self.nested_list) > 0 and not self.nested_list[0].isInteger():
            first_list = self.nested_list.pop(0).getList()
            for n in first_list[::-1]:
                self.nested_list.insert(0, n)
        return len(self.nested_list) > 0 



# Your NestedIterator object will be instantiated and called as such:
# i, v = NestedIterator(nestedList), []
# while i.hasNext(): v.append(i.next())