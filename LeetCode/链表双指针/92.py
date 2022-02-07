# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
# 92.反转链表2
class Solution:
    def reverseBetween(self, head: ListNode, left: int, right: int) -> ListNode:
        p_head = head 
        counter = 1
        pre_node = None
        while counter < left :
            counter += 1
            pre_node = p_head
            p_head = p_head.next
        next_node = self.reverse(p_head, right-left+1)
        if pre_node is None:
            return next_node
        else:
            pre_node.next = next_node
            return head

    def reverse(self, head, right):
        if right == 1:
            return head
        p_head, pre_node = head, None
        counter = 1
        while counter <= right:
            counter += 1
            next_node = p_head.next
            p_head.next = pre_node
            pre_node = p_head 
            p_head = next_node
        head.next = p_head
        return pre_node

def print_node(node):
    val_list = []
    while node is not None:
        val_list.append(node.val)
        node = node.next
    print(val_list)

if __name__ == "__main__":
    # left, right = 2, 4
    # node1 = ListNode(val=1)
    # node2 = ListNode(val=2)
    # node3 = ListNode(val=3)
    # node4 = ListNode(val=4)
    # node5 = ListNode(val=5)

    # node1.next = node2
    # node2.next = node3
    # node3.next = node4
    # node4.next = node5

    node1 = ListNode(val=5)
    left, right = 1, 1

    print_node(node1)
    sol = Solution()
    ret = sol.reverseBetween(node1, left, right)
    # ret = sol.reverse(node1, 5)
    # print(ret)

    print_node(ret)




        

            
        
                
