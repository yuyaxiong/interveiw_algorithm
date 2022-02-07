# 23.合并K个升序链表
# Definition for singly-linked list.
from typing import List
import heapq

class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class Solution:
    # 当成n-1个两个升序列表的合并
    def mergeKLists(self, lists: List[ListNode]) -> ListNode:
        if len(lists) == 0:
            return None
        p1 = lists.pop()
        for p2 in lists:
            p1 = self.mergeListNode(p1, p2)
        return p1

    def mergeListNode(self, p1, p2):
        p_head = ListNode()
        p_head_bk = p_head
        while p1 is not None and p2 is not None:
            if p1.val < p2.val:
                p_head.next = p1
                p1 = p1.next
            else:
                p_head.next = p2 
                p2 = p2.next
            p_head = p_head.next
        if p1 is not None:
            p_head.next = p1
        if p2 is not None:
            p_head.next = p2
        return p_head_bk.next
    

class Solution1:
    # 排序利用最小堆替代就是时间复杂度OK的K个升序的列表的合并 
    def mergeKLists(self, lists: List[ListNode]) -> ListNode:
        p_head = ListNode()
        p_head_bk = p_head
        lists = list(filter(lambda node : node is not None, lists))
        if len(lists) == 0:
            return None

        heap_list = sorted(lists, key=lambda node:node.val)
        while len(heap_list) > 0:
            node = heap_list.pop(0)
            p_head.next = node
            p_head = p_head.next

            node = node.next
            if node is not None:
                heap_list.append(node)
                heap_list = sorted(heap_list, key=lambda node: node.val)
        return p_head_bk.next