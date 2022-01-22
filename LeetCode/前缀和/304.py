class NumMatrix:
    
    def __init__(self, matrix: List[List[int]]):
        self.acc_matrix = [self.getAccListt(matrix[0])]
        for i in range(1, len(matrix)):
            acc_line = 0
            acc_list = []
            for j, n in enumerate(matrix[i]):
                acc_line += n
                acc_list.append(acc_line + self.acc_matrix[i-1][j])
            self.acc_matrix.append(acc_list)

    def getAccListt(self, n_list):
        acc = 0
        tmp = []
        for n in n_list:
            acc += n 
            tmp.append(acc)
        return tmp

    def sumRegion(self, row1: int, col1: int, row2: int, col2: int) -> int:
        if col1 > 0 and row1 > 0:
            return self.acc_matrix[row2][col2] - self.acc_matrix[row2][col1-1] - self.acc_matrix[row1-1][col2] + self.acc_matrix[row1-1][col1-1]
        elif col1 == 0  and row1 > 0:
            return self.acc_matrix[row2][col2] - self.acc_matrix[row1-1][col2]
        elif col1 > 0 and row1 == 0:
            return self.acc_matrix[row2][col2] - self.acc_matrix[row2][col1-1]
        elif col1 == 0 and row1 == 0:
            return self.acc_matrix[row2][col2]




# Your NumMatrix object will be instantiated and called as such:
# obj = NumMatrix(matrix)
# param_1 = obj.sumRegion(row1,col1,row2,col2)