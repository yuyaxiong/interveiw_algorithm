"""
构建乘积数组
给定一个数组A[0,1,...,n-1],请构建一个数组B[0,1,...n-1]，其中B中 的元素
B[i]=A[0]*A[1]*...*A[i-1]*A[i+1]*...*A[n-1]。不能使用除法。
"""
class Solution(object):
    def multiply(self, nList):
        cumproduct1 = [nList[0]]
        cumproduct2 = [nList[-1]]

        for i, j in zip(nList[1:-1], nList[::-1][1:-1]):
            cumproduct1.append(cumproduct1[-1] * i)
            cumproduct2.append(cumproduct2[-1] * j)
        cumproduct2 = cumproduct2[::-1]
        cumproduct2.append(1)
        cumproduct1.insert(0, 1)
        result = []
        for n1, n2 in zip(cumproduct1, cumproduct2):
            result.append(n1 * n2)
        return result

if __name__ == '__main__':
    s = Solution()
    nList = [1, 2, 3, 4, 5]
    print(s.multiply(nList))
