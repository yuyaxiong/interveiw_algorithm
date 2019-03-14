"""
堆排序，归并排序，快速排序，冒泡排序

"""
class SortAlgoritm1(object):
    def quick_sort(self, nList, s, e):
        if s < e:
            mid = self.quick_sort_help(nList, s, e)
            self.quick_sort(nList, s, mid)
            self.quick_sort(nList, mid+1, e)
        else:
            return

    def quick_sort_help(self, nList, s, e):
        i = s-1
        for j in range(s, e):
            if nList[e] >= nList[j]:
                i = i+1
                nList[i], nList[j] = nList[j], nList[i]
        nList[i+1], nList[e] = nList[e], nList[i+1]
        return i

class SortAlgorithm2(object):
    def merge_sort(self, nList, s, e):
        if s < e:
            mid = (s + e)//2
            self.merge_sort(nList, s, mid)
            self.merge_sort(nList, mid+1, e)
            self.merge_help(nList, s, mid, e)
        return

    def merge_help(self, nList, s,mid, e):
        tmp = []
        i, j = s, mid+1
        while i <= mid and j <= e:
            if nList[i] > nList[j]:
                tmp.append(nList[j])
                j += 1
            else:
                tmp.append(nList[i])
                i += 1

        if i <= mid:
            tmp.extend(nList[i:mid+1])
        if j <= e:
            tmp.extend(nList[j:e+1])

        for i in range(len(tmp)):
            nList[s+i] = tmp[i]
        return

class SortAlgorithm3(object):
    def heap_sort(self, nList):
        for i in range(len(nList)-1, -1, -1):
            for j in range(i, -1, -1):
                while j > 0:
                    if nList[j] >= nList[(j-1)//2]:
                        nList[j], nList[(j-1)//2] = nList[(j-1)//2], nList[j]
                    j = (j-1)//2
            nList[0], nList[i] = nList[i], nList[0]
        return nList


class SortAlgorithm4(object):
    def bubble_sort(self, nList):
        if len(nList) <= 1:
            return nList

        for i in range(len(nList), 0,-1):
            for j in range(1, i):
                if nList[j] < nList[j-1]:
                    nList[j], nList[j-1] = nList[j-1], nList[j]
        return


if __name__ == '__main__':
    sa1 = SortAlgoritm1()
    nList = [2, 3, 1, 5, 6, 9, 8, 5]
    # sa.quick_sort(nList, s=0, e=len(nList))
    # print(nList)
    # print(sa.quick_sort_help(nList, 0, len(nList)-1))
    # print(nList)
    # sa1.quick_sort(nList, 0, len(nList)-1)
    # print(nList)
    # sa2 = SortAlgorithm2()
    # sa2.merge_sort(nList, 0, len(nList)-1)
    # sa3 = SortAlgorithm3()
    # sa3.heap_sort(nList)
    sa4 = SortAlgorithm4()
    sa4.bubble_sort(nList)
    print(nList)