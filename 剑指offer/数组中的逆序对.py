"""
在数组中的两个数字，如果前面一个数字大于后面的数字，则这两个数字
组成一个逆序对。输入一个数组，求这个数组中的逆序对总数。例如：
在数组[7，5，6，4]中，一共存在5个逆序对，分别是（7，6）、（7，5）、
（7，4）、（6，4）、（5，4）

"""
class Solution(object):
    def inverse_pairs(self, data, lenght):
        if data is None or lenght < 0:
            return 0
        copy = data[::]
        count = self.inverse_pairs_core(data, copy, 0, len(copy)-1)
        return count
        
    def inverse_pairs_core(self, data, copy, s, e):
        if s == e:
            copy[s] = data[s]
            return 0
        length = (e - s)//2
        left = self.inverse_pairs_core(copy, data, s, s+length)
        right = self.inverse_pairs_core(copy, data, s+length+1, e)
        i = s + length
        j = e
        index_copy = e
        count = 0
        # 归并的步骤
        while i >= s and j >= s + length + 1:
            if data[i] > data[j]:
                copy[index_copy] = data[i]
                index_copy -= 1
                i -= 1
                count += j - s - length
            else:
                copy[index_copy] = data[j]
                index_copy -= 1
                j -= 1


        # 要么多余i， 要么多余j，不会又多出i，又多出j
        for i in range(i, s, -1):
            copy[index_copy] = data[i]
            index_copy -= 1

        for j in range(j, s+length+1, -1):
            copy[index_copy] = data[j]
            index_copy -= 1

        return left + right + count

if __name__ == '__main__':
    nList = [7, 5, 6, 4]
    s = Solution()
    print(s.inverse_pairs(nList, len(nList)))
                
        
        
        
        