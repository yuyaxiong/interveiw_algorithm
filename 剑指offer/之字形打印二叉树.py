"""
请实现一个函数按照之字型顺序定二叉树，即第一行按照从左到右的顺序打印，
第二层按照从右到左的顺序打印，第三行按照从左到右的顺序打印，其他行依次类推。
例如，按之字形打印图4.8中二叉树的结构

    8
   | |
  6  10
 ||  ||
5 7 9 11
"""

class BinaryTree(object):
    def __init__(self):
        self.value = None
        self.left = None
        self.right = None

class Solution(object):
    # 需要NList是个栈
    def print_bt(self, nList, depth):
        if len(nList) == 0:
            return
        tmp = []
        strings = ''
        for node in nList:
            strings += '%s\t' % node.value
            if depth % 2 == 1:
                if node.left is not None:
                    tmp.append(node.left)
                if node.right is not None:
                    tmp.append(node.right)
            else:
                if node.right is not None:
                    tmp.append(node.right)
                if node.left is not None:
                    tmp.append(node.left)

        depth += 1
        print(strings)
        print('\n')
        self.print_bt(tmp[::-1], depth)

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
    s.print_bt([pRoot], 1)