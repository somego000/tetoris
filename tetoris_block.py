#ブロックが消去されたことは盤面の情報として扱えばよい。
class block:
    def __init__(self,block_num,rotate_num):
        self.block_num = block_num
        self.rotate_num = rotate_num
        self.active = True
        self.block_board = [[0] * 4 for _ in range(4)]
        if block_num == 0:
            for i in range(4):
                self.block_board[i][1] = 1
        elif block_num == 1:
            positions = [(1, 0), (1, 1), (1, 2), (2, 1)]
            for y, x in positions:
                self.block_board[y][x] = 1
        elif block_num == 2:
            positions = [(1, 1), (1, 2), (2, 1),(2, 2)]
            for y, x in positions:
                self.block_board[y][x] = 1
        elif block_num == 3:
            positions = [(0, 1), (1, 1), (2, 1),(2, 2)]
            for y, x in positions:
                self.block_board[y][x] = 1
        elif block_num == 4:
            positions = [(0, 2), (1, 2), (2, 1),(2, 2)]
            for y, x in positions:
                self.block_board[y][x] = 1
        elif block_num == 5:
            positions = [(0, 1), (1, 1), (1, 2),(2, 2)]
            for y, x in positions:
                self.block_board[y][x] = 1
        elif block_num == 6:
            positions = [(0, 2), (1, 1), (1, 2),(2, 1)]
            for y, x in positions:
                self.block_board[y][x] = 1
        
    def rotate(self):#回転処理をするメソッド
        self.rotate_num = (self.rotate_num + 1) % 4
        self.block_board = [list(row) for row in zip(*self.block_board[::-1])]
    def reverse_rotate(self):
        self.rotate_num = (self.rotate_num - 1) % 4
        self.block_board = [list(row) for row in zip(*self.block_board)][::-1]
    def lock(self):
        self.active=False     
    
