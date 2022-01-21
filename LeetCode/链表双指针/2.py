#两数相加

# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
class Solution:
    def addTwoNumbers(self, l1: ListNode, l2: ListNode) -> ListNode:
        l1_len = self.getLength(l1)
        l2_len = self.getLength(l2)
        if l2_len > l1_len:
            l1, l2 = l2, l1
        p = 0
        pre_node = None
        l1_head = l1
        while l1 is not None:
            l2_val = 0 if l2 is None else l2.val
            cur_val = l1.val + l2_val + p
            if cur_val // 10 == 1:
                p = 1
                cur_val = cur_val % 10
            else:
                p = 0
            l1.val = cur_val
            pre_node = l1
            l1 = l1.next
            l2 = None if l2 is None else l2.next
        if p != 0:
            pre_node.next = ListNode(val=p)
        return l1_head
            
    def getLength(self, list_node):
        counter = 0
        while list_node is not None:
            counter += 1
            list_node = list_node.next
        return counter
            