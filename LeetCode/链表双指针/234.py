# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
class Solution:
    def isPalindrome(self, head: ListNode) -> bool:        
        n_list = []
        while head is not None:
            n_list.append(head.val)
            head = head.next
        print(n_list)
        i, j = 0, len(n_list)-1
        status = True
        while i < j:
            if n_list[i] != n_list[j]:
                status = False
                break
            else:
                i +=1 
                j -= 1
        return status


class Solution1:
    def isPalindrome(self, head: ListNode) -> bool:
        p_head = head
        pre_node = ListNode(p_head.val)
        p_head = p_head.next
        while p_head is not None:
            node = ListNode(p_head.val)
            node.next = pre_node
            pre_node = node
            p_head = p_head.next 

        status = True
        while pre_node is not None and head is not None:
            if pre_node.val == head.val:
                pre_node = pre_node.next 
                head = head.next
            else:
                status = False
                break
        return status

