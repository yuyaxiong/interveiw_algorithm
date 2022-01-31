# Definition for a binary tree node.
from typing import List
import sys
sys.setrecursionlimit(100000)

# 105.从前序遍历和中序遍历构造二叉树
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
class Solution:
    def buildTree(self, preorder: List[int], inorder: List[int]) -> TreeNode:
        if len(preorder) == 0:
            return None
        val = preorder.pop(0)
        node = TreeNode(val)
        in_idx = inorder.index(val)
        left_inorder = inorder[:in_idx]
        right_inorder = inorder[in_idx+1:]
        left_preorder, right_preorder = self.getLeftRight(preorder, left_inorder)
        node.left = self.buildTree(left_preorder, left_inorder)
        node.right = self.buildTree(right_preorder, right_inorder)
        return node

    def getLeftRight(self, preorder, left_inorder):
        left_preorder = []
        right_preorder = []
        left_inorder_dict = {n:1 for n in left_inorder}
        for o in preorder:
            if o in left_inorder_dict:
                left_preorder.append(o)
            else:
                right_preorder.append(o)
        return left_preorder, right_preorder


def testCase():
    import json
    import time
    line_list = []
    with open("./test_case/105.txt") as f:
        for line in f.readlines():
            n_list = json.loads(line.strip())
            line_list.append([int(n) for n in n_list])
    preorder = line_list[0]
    inorder = line_list[1]

    # print(len(preorder))
    # print(len(inorder))
    t1 = time.time()
    sol = Solution()
    sol.buildTree(preorder, inorder)
    t2 = time.time()
    print((t2-t1)*1000)

if __name__ == "__main__":
    testCase()


