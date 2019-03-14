"""
输入两个链表，找出它们的第一个公共节点。

1->2->3->6->7
   4->5->6->7

"""
class ListNode(object):
    def __init__(self):
        self.value = None
        self.next = None


class Solution(object):
    def find_first_common_node(self, pHead1, pHead2):
        pList1, pList2 = [], []
        while pHead1.next is not None:
            pList1.append(pHead1)
            pHead1 = pHead1.next
        while pHead2.next is not None:
            pList2.append(pHead2)
            pHead2 = pHead2.next
        p1, p2 = pList1.pop(), pList2.pop()
        last = None
        while p1 == p2:
            last = p1
            p1 = pList1.pop()
            p2 = pList2.pop()
        return last

class Solution1(object):
    def find_first_common_node(self, pHead1, pHead2):
        p1, p2 = pHead1, pHead2
        counter1 = 0
        counter2 = 0
        while pHead1.next is not None:
            counter1 += 1
            pHead1 = pHead1.next

        while pHead2.next is not None:
            counter2 += 1
            pHead2 = pHead2.next

        if counter1 > counter2:
            while counter1 - counter2 > 0:
                counter1 -= 1
                p1 = p1.next
        elif counter1 < counter2:
            while counter2 - counter1 > 0:
                counter2 -= 1
                p2 = p2.next

        while p1 != p2:
            p1 = p1.next
            p2 = p2.next
        return p1






if __name__ == '__main__':
    pHead1 = ListNode()
    pHead1.value = 1
    pHead1.next = ListNode()
    pHead1.next.value = 2
    pHead1.next.next = ListNode()
    pHead1.next.next.value = 3
    pHead1.next.next.next = ListNode()
    pHead1.next.next.next.value = 6
    pHead1.next.next.next.next = ListNode()
    pHead1.next.next.next.next.value = 7
    pHead2 = ListNode()
    pHead2.value = 4
    pHead2.next = ListNode()
    pHead2.next.value = 5
    pHead2.next.next = pHead1.next.next.next


    s = Solution()
    node = s.find_first_common_node(pHead1, pHead2)
    s1 = Solution1()
    node1 = s1.find_first_common_node(pHead1, pHead2)
    print(node.value)
    print(node1.value)


