"""
第一个公共父节点为，
"""
class BinaryTree(object):
    def __init__(self):
        self.value = None
        self.left = None
        self.right = None


class Solution(object):
    def common_father(self, pRoot, p, q):
        if pRoot is None:
            return
        s1 = self.subtree(pRoot.left, p)
        s2 = self.subtree(pRoot.right, q)
        if s1 and s2 :
            return pRoot
        elif s1 is False and s2:
            return self.subtree(pRoot.right, p, q)
        elif s1 and s2 is False:
            return self.subtree(pRoot.left, p, q)
        else:
            return False

    def subtree(self, pRoot, s):
        if pRoot is None:
            return False
        if pRoot.value == s:
            return True
        return self.subtree(pRoot.left, s) or self.subtree(pRoot.right, s)



if __name__ == "__main__":
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
    # print(s.subtree(pRoot, 14))
    print(s.common_father(pRoot, 6, 11).value)