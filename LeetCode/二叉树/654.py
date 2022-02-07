# Definition for a binary tree node.

# 654. 最大二叉树
from typing import List

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
class Solution:
    def constructMaximumBinaryTree(self, nums: List[int]) -> TreeNode:
        if len(nums) == 0:
            return None
        max_val, max_idx = self.getMaxIdx(nums)
        print(max_val)
        left_nums = nums[:max_idx]
        right_nums = nums[max_idx+1:]
        node = TreeNode(max_val)
        node.left = self.constructMaximumBinaryTree(left_nums)
        node.right = self.constructMaximumBinaryTree(right_nums)
        return node
        

    def getMaxIdx(self, nums):
        max_val, max_idx = nums[0], 0
        for i in range(1, len(nums)):
            if max_val < nums[i]:
                max_val = nums[i]
                max_idx = i
        return max_val, max_idx

def print_node(node_list):
    if len(node_list) == 0:
        return
    tmp_list = []
    lines = []
    for node in node_list:
        if node.left is not None:
            tmp_list.append(node.left)
        if node.right is not None:
            tmp_list.append(node.right)
        lines.append(node.val)
    print(lines)
    print_node(tmp_list)
    return 


def testCase():
    nums = [3,2,1,6,0,5]
    sol = Solution()
    node = sol.constructMaximumBinaryTree(nums)
    print_node([node])


if __name__ == "__main__":
    testCase()

