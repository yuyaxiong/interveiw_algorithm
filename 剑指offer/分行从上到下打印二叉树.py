"""
从上到下按层打印二叉树，同一层的节点按从左到右的顺序打印，
每一层打印到一行。例如，打印图4.7中二叉树的结果为：
"""
class BinaryTree(object):
    def __init__(self):
        self.value = None
        self.left = None
        self.right = None

class Solution(object):
    def print_from_top_to_bottom(self, pTreeRoot):
        if pTreeRoot is None:
            return None
        nodes = []
        nodes.append(pTreeRoot)
        nextLevel = 0
        toBePrinted = 1
        while len(nodes) != 0:
            pNode = nodes[0]
            print(pNode.value)
            if pNode.left is not None:
                nodes.append(pNode.left)
                nextLevel += 1
            if pNode.right is not None:
                nodes.append(pNode.right)
                nextLevel += 1
            nodes = nodes[1:]
            toBePrinted -= 1
            if toBePrinted == 0:
                print('\n')
                toBePrinted = nextLevel
                nextLevel = 0

class Solution1(object):
    def print_from_top_to_bottom(self, nList):
        if len(nList) == 0:
            return
        tmp = []
        strings = ''
        for node in nList:
            strings += '%s\t'% node.value
            if node.left is not None:
                tmp.append(node.left)
            if node.right is not None:
                tmp.append(node.right)
        print(strings)
        print('\n')
        self.print_from_top_to_bottom(tmp)

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
    s = Solution1()
    print(s.print_from_top_to_bottom([pRoot]))

