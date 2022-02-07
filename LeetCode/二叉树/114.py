# Definition for a binary tree node.

# 114. 二叉树展开为链表
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def flatten(self, root: TreeNode) -> None:
        """
        Do not return anything, modify root in-place instead.
        """
        if root is None:
            return []

        node_list = [root]
        while len(node_list) > 0:
            node = node_list.pop(0)
            tmp_nodes = []
            node_left, node_right = node.left , node.right
            node.left = None
            if node_left is not None:
                node.right = node_left
                if node_right is not None:
                    tmp_nodes = [node_left, node_right]
                else:
                    tmp_nodes = [node_left]
                tmp_nodes.extend(node_list)
                node_list = tmp_nodes
                continue
            elif node_right  is not None:
                tmp_nodes = [node_right]
                tmp_nodes.extend(node_list)
                node_list = tmp_nodes
                continue
            elif node_right is None:
                if len(node_list) > 0:
                    node.right = node_list[0]
                else:
                    break
        return root


        

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
    tn1 = TreeNode(val=1)
    tn2 = TreeNode(val=2)
    tn3 = TreeNode(val=5)
    tn4 = TreeNode(val=3)
    tn5 = TreeNode(val=4)
    tn6 = TreeNode(val=6)
    tn1.left = tn2
    tn1.right = tn3
    tn2.left = tn4
    tn2.right = tn5 
    tn3.right = tn6

    sol = Solution()
    linkNode = sol.flatten(tn1)
    print_node([linkNode])


if __name__ == "__main__":
    testCase()






        