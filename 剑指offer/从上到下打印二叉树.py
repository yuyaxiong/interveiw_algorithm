"""
从上到下打印出二叉树的每个节点，同一层的节点按照从左到右的顺序打印。
例如下列二叉树，依次打印出8,6,10,5,7,9,11
    8
   | |
  6  10
 ||  ||
5 7 9 11
"""
from queue import Queue

class BinaryTree(object):
    def __init__(self):
        self.value = None
        self.left = None
        self.right = None

class Solution(object):
    def print_from_top_to_bottom(self, pTreeRoot):
        if pTreeRoot is None:
            return None
        q = []
        q.append(pTreeRoot)
        self.print_help(q)
        return

    def print_help(self, qList):
        if len(qList) == 0:
            return
        root = qList[0]
        qList = qList[1:]
        print(root.value)
        if root.left is not None:
            qList.append(root.left)
        if root.right is not None:
            qList.append(root.right)
        self.print_help(qList)

if __name__ == '__main__':
    pRoot = BinaryTree()
    pRoot.value = 8
    pRoot.left = BinaryTree()
    pRoot.right = BinaryTree()
    pRoot.left.value = 6
    pRoot.right.value = 10
    pl = pRoot.left
    pr = pRoot.right
    pl.left = BinaryTree()
    pl.right = BinaryTree()
    pr.left = BinaryTree()
    pr.right = BinaryTree()
    pl.left.value = 5
    pl.right.value = 7
    pr.left.value = 9
    pr.right.value = 11
    s = Solution()
    s.print_from_top_to_bottom(pRoot)