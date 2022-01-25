# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
# 226. 翻转二叉树
class Solution:
    def invertTree(self, root: TreeNode) -> TreeNode:
        if root is None:
            return root
        node_list = [root]
        self.invertTreeHelp(node_list)
        return root

    def invertTreeHelp(self, node_list):
        if len(node_list) == 0:
            return 
        tmp = []
        print(node_list)
        for node in node_list:
            left_node, right_node = node.left, node.right
            node.left, node.right = right_node, left_node
            if left_node is not None:
                tmp.append(left_node)
            if right_node is None:
                tmp.append(right_node)
        node_list = tmp
        self.invertTreeHelp(node_list)


def print_node(node_list):
    if len(node_list) == 0:
        return 

    while len(node_list) > 0:
        tmp = []
        nodes = []
        for node in node_list:
            tmp.append(node.val)
            if node.left is not None:
                nodes.append(node.left)
            if node.right is not None:
                nodes.append(node.right)
        print(tmp)
        node_list = nodes
    return 



def testCase():
    tn1 = TreeNode(val=4)
    tn2 = TreeNode(val=2)
    tn3 = TreeNode(val=7)
    tn4 = TreeNode(val=1)
    tn5 = TreeNode(val=3)
    tn6 = TreeNode(val=6)
    tn7 = TreeNode(val=7)
    tn1.left = tn2
    tn1.right = tn3
    tn2.left = tn4
    tn2.right = tn5
    tn3.left = tn6
    tn3.right = tn7

    sol = Solution()
    root = sol.invertTree(tn1)

    print_node([root])


if __name__ == "__main__":
    testCase()
