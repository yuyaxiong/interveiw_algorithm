"""
请实现两个函数，分别用来序列化和反向序列化二叉树

        1
    2       3
4         5   6

"""
class BinaryTree(object):
    def __init__(self):
        self.value = None
        self.left = None
        self.right = None


class Solution(object):
    def serialize(self, pRoot, steam):
        if pRoot is None:
            steam.append('$')
            return
        steam.append('%s' % pRoot.value)
        self.serialize(pRoot.left, steam)
        self.serialize(pRoot.right, steam)

    def deserialize(self, steam, idx):
        if len(steam) < idx:
            return
        pRoot = None
        if steam[idx] != '$':
            pRoot = BinaryTree()
            pRoot.value = int(steam[idx])
            idx += 1
            pRoot.left, idx = self.deserialize(steam, idx)
            pRoot.right, idx = self.deserialize(steam, idx)
        else:
            idx += 1
        return pRoot, idx



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

    steam = []
    s = Solution()
    s.serialize(pRoot, steam)
    # Root = BinaryTree()
    Root,_ = s.deserialize(steam, 0)
    print(Root.value)
    # print(','.join(steam))