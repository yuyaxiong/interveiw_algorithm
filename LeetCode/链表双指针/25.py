# Definition for singly-linked list.
from typing import Optional

# 25. K 个一组翻转链表
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
class Solution:
    def reverseKGroup(self, head: Optional[ListNode], k: int) -> Optional[ListNode]:
        a, b = head, head 
        for i in range(k):
            if b == None:
                return head
            b = b.next
        new_head = self.reverse(a, b)
        a.next = self.reverseKGroup(b, k)
        return new_head


    def reverse(self, a, b):
        pre, cur, nxt = None, a, a
        while cur != b:
            nxt = cur.next
            cur.next = pre
            pre = cur
            cur = nxt
        return pre

        

def print_node(node):
    val_list = []
    while node is not None:
        print(node.val)
        val_list.append(node.val)
        node = node.next
    print(val_list)


if __name__ == "__main__":
    node1 = ListNode(val=1)
    node2 = ListNode(val=2)
    node3 = ListNode(val=3)
    node4 = ListNode(val=4)
    node5 = ListNode(val=5)
    node1.next = node2
    node2.next = node3
    node3.next = node4
    node4.next = node5

    # print_node(node1)
    sol = Solution()
    ret = sol.reverseKGroup(node1, 2)
    # print(ret)
    print_node(ret)



    

