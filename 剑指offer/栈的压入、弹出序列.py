"""
输入两个整数序列，第一个序列表示栈的压入顺序，请判断第二个序列
是否为该栈的弹出序列。假设压入栈的所有数字均不相等。例如：序列
{1,2,3,4,5}是某栈的压栈序列，序列{4,5,3,2,1}是该压栈序列对应
的一个弹出序列，但{4,3,5,1,2}就不可能是该压栈序列的弹出序列。
"""


class Solution1(object):
    def is_pop_order(self, pPush, pPop, length):
        bPossible = False
        if pPush is not None and pPop is not None and length > 0:
            pNextPush = 0
            pNextPop = 0

            stackData= []
            while pNextPop < length:
                while len(stackData) == 0 or stackData[-1] != pPop[pNextPop]:
                    if pNextPush > length:
                        break
                    stackData.append(pPush[pNextPush])
                    pNextPush += 1

                if stackData[-1] != pPop[pNextPop]:
                    break
                # 相等则pop
                stackData.pop()
                pNextPop += 1

            if len(stackData) == 0 and pNextPop == length -1 :
                bPossible = True

        return bPossible



if __name__ == '__main__':
    s1 = Solution1()
    stack1 = [1,2,3,4,5]
    stack2 = [4,3,5,2,1]
    # print(s.is_pop_order(stack1, stack2))
    print(s1.is_pop_order(stack1, stack2, len(stack1)))


