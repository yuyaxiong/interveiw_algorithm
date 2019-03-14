"""
输入n个整数，找出其中最小的K个数。
例如：输入4，5，1，6，2，7，3，8这8个数字，则最小的4个数字是1，2，3，4
"""
class Solution(object):
    def get_least_num(self, inputs, s, e, k):
        index = self.partition(inputs, s, e)
        if index == k:
            return inputs[:index]
        elif index < k:
            return self.get_least_num(inputs, index+1, e, k)
        else:
            return self.get_least_num(inputs, s, index, k)


    def partition(self,inputs ,s, e):
        i = s-1
        for j in range(s, e):
            if inputs[j] <= inputs[e]:
                i += 1
                inputs[i], inputs[j] = inputs[j], inputs[i]
        inputs[i+1], inputs[e] = inputs[e], inputs[i+1]
        return i

class Solution1(object):
    def get_least_num(self, inputs,k):
        hList = []
        for n in inputs:
            if len(hList) < k:
                hList.append(n)
            else:
                self.reshape_heap(hList)
                if hList[-1] > n:
                    hList.pop()
                    hList.append(n)
        return hList

    def reshape_heap(self, nList):
        for i in range(len(nList)-1, -1, -1):
            while i > 0:
                if nList[i] > nList[(i-1)//2]:
                    nList[(i-1)//2], nList[i] = nList[i], nList[(i-1)//2]
                i = (i-1)//2
        nList[0], nList[-1] = nList[-1], nList[0]
        return

    def heap_sort(self, nList):
        for j in range(len(nList)-1, -1, -1):
            for i in range(j, -1, -1):
                while i > 0:
                    if nList[i] > nList[(i-1)//2]:
                        nList[(i-1)//2], nList[i] = nList[i], nList[(i-1)//2]
                    i = (i-1)//2
            nList[0], nList[j] = nList[j], nList[0]
        return


if __name__ == '__main__':
    s = Solution1()
    nList = [4,6,7,8,2,3,4,8,9]
    # s.quick_sort(nList, 0, len(nList)-1)
    # print(s.partition(nList, 0, len(nList)-1))
    # print(s.get_least_num(nList, 0,len(nList)-1, 3))
    # print(nList)
    # n5 = nList[:5]
    # s.reshape_heap(n5)
    # s.heap_sort(nList)
    print(s.get_least_num(nList, 5))
    # print(n5)