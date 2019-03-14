"""
请实现一个函数，用来判断一颗二叉树是不是对称的。如果一颗二叉树和它的镜像一样
，那么它是对称的。例如，在如图中的3颗二叉树
  8
 | |
 6 6
|| ||
57 75

   8
  | |
  6 9
 || ||
 57 75

   7
  | |
  7 7
 || ||
 77 7
"""

class BinaryTree(object):
    def __init__(self):
        self.value = None
        self.left = None
        self.right = None

class Solution(object):
    def is_symmetrical(self, pRoot):
        result1 = []
        result2 = []
        self.DLR(pRoot, result1)
        self.DLR1(pRoot, result2)
        if len(result1) == len(result2) and self.equal(result1, result2):
            return True
        else:
            return False


    def equal(self, l1, l2):
        for i, j in zip(l1, l2):
            if i != j:
                return False
        return True

    def DLR(self, pRoot, result):
        if pRoot is None:
            return
        result.append(pRoot.value)
        self.DLR(pRoot.left, result)
        self.DLR(pRoot.right, result)

    def DLR1(self, pRoot, result):
        if pRoot is None:
            return
        result.append(pRoot.value)
        self.DLR1(pRoot.right, result)
        self.DLR1(pRoot.left, result)


class Solution1(object):
    def is_symmetrical(self, pRoot):
        return self.is_symmetrical_help(pRoot, pRoot)

    def is_symmetrical_help(self, pRoot1, pRoot2):
        if pRoot1.value == pRoot2.value and pRoot2 is None:
            return True
        if pRoot1.value is None or pRoot2.value is None:
            return False
        if pRoot1.value != pRoot2.value:
            return False
        return self.is_symmetrical_help(pRoot1, pRoot2) and self.is_symmetrical_help(pRoot2, pRoot1)


if __name__ == '__main__':
    s = Solution()
    pRoot = BinaryTree()
    pRoot.value = 8
    pRoot.left = BinaryTree()
    pRoot.right = BinaryTree()
    pl = pRoot.left
    pr = pRoot.right
    pl.value = 6
    pr.value = 6
    pl.left = BinaryTree()
    pl.right = BinaryTree()
    pl.left.value = 5
    pl.right.value = 7
    pr.left = BinaryTree()
    pr.left.value = 7
    pr.right = BinaryTree()
    pr.right.value = 5

    print(s.is_symmetrical(pRoot))

