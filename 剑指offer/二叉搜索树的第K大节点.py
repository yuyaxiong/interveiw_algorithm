"""
给定一颗二叉搜索树，请找出其中第K大的节点，例如，在图中的二叉搜索树里
按节点数值大小顺序，第三大节点的值是4。
    5
  3   7
2  4 6  8
"""
class BinaryTree(object):
    def __init__(self):
        self.value = None
        self.left = None
        self.right = None


class Solution(object):
    def kthNode(self, pRoot, k):
        mid_list = []
        self.kthNode_help(pRoot, mid_list)
        return mid_list[k-1]

    def kthNode_help(self, pRoot, nList):
        if pRoot is None:
            return

        self.kthNode_help(pRoot.left, nList)
        nList.append(pRoot.value)
        self.kthNode_help(pRoot.right, nList)

if __name__ == '__main__':
    pRoot = BinaryTree()
    pRoot.value = 5
    pRoot.left = BinaryTree()
    pRoot.left.value = 3
    pl = pRoot.left
    pRoot.right = BinaryTree()
    pRoot.right.value = 7
    pr = pRoot.right
    pl.left = BinaryTree()
    pl.right = BinaryTree()
    pr.left = BinaryTree()
    pr.right = BinaryTree()
    pl.left.value = 2
    pl.right.value = 4
    pr.left.value = 6
    pr.right.value = 8

    s = Solution()
    print(s.kthNode(pRoot, 3))