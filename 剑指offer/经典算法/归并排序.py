class Solution(object):
    def merge_sort(self, nList):
        pass

    def merge_sort_help(self, nList, s, e):
        if s < e:
            mid = (s + e)// 2
            self.merge_sort_help(nList, s, mid)
            self.merge_sort_help(nList, mid+1, e)
            self.merge_help(nList, s, mid, e)

    def merge_help(self, nList, s, mid, e):
        tmp = []
        i, j = s, mid+1
        while i <= mid and j <= e:
            if nList[i] <= nList[j]:
                tmp.append(nList[i])
                i += 1
            else:
                tmp.append(nList[j])
                j += 1
                
        if i<=mid:
            tmp.extend((nList[i:mid+1]))
        if j <= e:
            tmp.extend(nList[j:e+1])
        for i in range(len(tmp)):
            nList[s+i] = tmp[i]
        return 
        