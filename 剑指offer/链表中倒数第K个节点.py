"""
输入一个链表，输出该链表中倒数第K个节点。为了符合大多数人的习惯，本题从1开始计数，即链表的尾节点。
例如：一个链表有6个节点，从头节点开始，他们的值依次是1,2,3,4,5,6.这个链表的倒数第3个节点是值
为4的节点。

"""

class ListNode(object):
    def __init__(self):
        self.value = None
        self.next = None



class Solution(object):
    def find_kth_to_tail(self, pListHead, k):
        if pListHead is not None:
            return None
        sp = pListHead
        ep = None
        counter = 0
        while counter < k:
            if sp is not None:
                sp = sp.next
                counter += 1
            else:
                return None
        ep = pListHead
        while sp.next is not None:
            sp = sp.next
            ep = ep.next
        return ep