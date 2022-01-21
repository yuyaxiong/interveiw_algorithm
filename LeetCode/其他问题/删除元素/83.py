# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class Solution:
    def deleteDuplicates(self, head: ListNode) -> ListNode:
        if head is None:
            return head
        s_p = head
        pre_val = head.val
        f_p = head
        while f_p is not None:
            if pre_val != f_p.val:
                s_p.next = f_p
                s_p = s_p.next
            pre_val = f_p.val
            f_p = f_p.next
        s_p.next = None
        return head