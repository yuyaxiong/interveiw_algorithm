# Definition for a binary tree node.
# 337. 打家劫舍 III
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def rob(self, root: TreeNode) -> int:
        self.memo = dict()
        res = self.robHelp(root)
        return res 
    
    def robHelp(self, root):
        if root is None:
            return 0
        if self.memo.get(root) is not None:
            return self.memo.get(root)
        do_it = root.val
        if root.left is not None:
            do_it += self.robHelp(root.left.left) + self.robHelp(root.left.right)
        if root.right is not None:
            do_it += self.robHelp(root.right.left) + self.robHelp(root.right.right)
        not_do = self.robHelp(root.left) + self.robHelp(root.right)
        res = max(do_it, not_do)
        self.memo[root] = res
        return res


def testCase():
    node1 = TreeNode()
    node2 = TreeNode()
    node3 = TreeNode()
    node4 = TreeNode()
    node5 = TreeNode()
    node1.val = 3
    node1.left = node2
    node1.right = node3
    node2.val = 2
    node3.val = 3
    node2.right = node4
    node4.val = 3
    node3.right = node5
    node5.val = 1
    sol = Solution()
    res = sol.rob(node1)
    print(res)
    print(sol.memo)

if __name__=="__main__":
    testCase()
