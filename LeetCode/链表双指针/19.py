# Definition for singly-linked list.
# 19.删除链表的倒数第N个节点
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
class Solution:
    def removeNthFromEnd(self, head: ListNode, n: int) -> ListNode:
        p_head = ListNode(next=head)
        p_head_bk = p_head 
        length = self.getLength(head)
        if length == n and n == 1:
            return None
        i = 0
        while i < length -n:
            p_head = p_head.next
            i += 1
        p_head_next = p_head.next
        if p_head_next is not None:
            p_head.next = p_head_next.next
        else:
            p_head.next = None
        return p_head_bk.next

    def getLength(self, head):
        counter = 0
        while head is not None:
            counter += 1
            head = head.next
        return counter

