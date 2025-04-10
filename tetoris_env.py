import numpy as np
import gym
import time
from gym import spaces
import pygame
import random
from tetoris_block import block 
class TetrisEnv(gym.Env):
    def __init__(self):
        super(TetrisEnv, self).__init__()

        # Pygame の初期化
        pygame.init()
        self.screen = pygame.display.set_mode((400, 800))
        self.clock = pygame.time.Clock()
        self.start_time = 0
        self.current_time = 0
        # グリッドサイズ
        self.width = 14
        self.height = 26
        self.board = np.zeros((self.height, self.width), dtype=int)
        self.col_start = 0
        self.row_start = 0
        self.fall_block = self.get_new_piece()
        # アクションスペース (0: 左, 1: 右, 2: 回転, 3: 下)
        self.action_space = spaces.Discrete(6)
        self.before_line_judge = 0

        # 状態スペース（盤面をフラットにして入力）
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.height * self.width,), dtype=np.int8)

        self.reset()

    def reset(self):
        """ゲームのリセット"""
        self.board = np.zeros((self.height, self.width), dtype=int)
        self.done = False
        return self.get_state()
    def block_call(self):
        self.fall_block = self.get_new_piece()
    def step(self, action):
        """アクションを適用して次の状態と報酬を取得"""
        reward = 0
        self.div_num = 1
        if action == 0:  # 左移動
            self.move_left()
        elif action == 1:  # 右移動
            self.move_right()
        elif action == 2:  # 回転
            self.rotate_piece()
        elif action == 3:  # 下移動
            self.move_down()
        elif action == 4:  # 下移動
            self.move_up()
            self.div_num = 3
        # ゲームオーバーチェック
        if self.is_game_over():
            self.done = True
        # if self.is_game_over_reach():
        #     reward -= 10000  # ゲームオーバーのペナルティ
        if self.block_judge(0,1) == 0:
            lines_cleared=self.clear_lines_judge()
            if lines_cleared == 0:
                reward += 0
                # if any(self.board[4][j] == -1 for j in range(2,12)):
                #     reward -= 100
                #     self.div_num = 1
            else:
                reward += 20*(1 + 1.4**lines_cleared)
                if self.before_line_judge != 0:
                    reward += 50  + self.before_line_judge*0.1
                self.before_line_judge = lines_cleared
            a,reward_num2,h2 = self.reward_get(1)
            if a == 0:
                reward += 20
                if h2 != 0:
                    reward *= h2
            else:
                reward += a
            reward += reward_num2
            i_max = i_min = 0
            for j in range(2,12):
                for i in range(4,24):
                    if self.board[i][j] == -1:
                        if i_max < i:
                            i_max = i
                    if self.board[i][j] == 1:
                        if i_min > i:
                            i_min = i
                if self.board[23][j] == 0:
                    i_max = 24
            #reward -= 1.1 ** (i_max - i_min)
            #print(f' if reward = {reward}')
            reward *= self.div_num
        else:
            min_i_total = 0
            min_j_total = 0
            reward_min_total = 3000
            self.board_change_reward_disappear(self.row_start,self.col_start)
            for j in range(11):
                temp_row_start = j
                min_i = 0
                min_j = 0
                reward_min = 3000
                for i in range(4):
                    temp_col_start = self.col_start + 1
                    #print("AAAA###")
                    if self.block_judge_reward(temp_row_start,temp_col_start) == 1:
                        while self.block_judge_reward(temp_row_start,temp_col_start+1) == 1:
                            temp_col_start += 1
                        #print("$$$$",i,j,temp_col_start,self.col_start)
                        self.board_change_reward_appear(temp_row_start,temp_col_start)
                        reward_num,reward_num2,_ = self.reward_get(0)
                        reward_num += reward_num2
                        if self.ran_rotate == 0 and reward_num < reward_min or self.ran_rotate == 1 and reward_num <= reward_min:
                            reward_min = reward_num
                            min_i = temp_col_start
                            min_j = temp_row_start
                        self.board_change_reward_disappear(temp_row_start,temp_col_start)
                    self.fall_block.rotate()
                    #else:
                        #print("$$$$",i,j,temp_col_start,self.col_start)
                    #self.render()
                if self.ran_rotate == 0 and reward_min < reward_min_total or self.ran_rotate == 1 and reward_min <= reward_min_total:
                        reward_min_total = reward_min
                        min_i_total = min_i
                        min_j_total = min_j
            reward += 10 - (np.sqrt((min_i_total - self.col_start)**2 + (min_j_total - self.row_start)**2))    
            self.board_change_reward_appear(self.row_start,self.col_start)                
        # ラインクリアの処理
        #self.render()
        self.clear_lines()
        return self.get_state(), reward, self.done, {}
    def reward_get(self,n):
        reward = 0
        reward2 = 0
        h2 = 0
        for j in range(2,12): 
            for i in range(4,24):
                #and ((i >= 5 and j >= 3 and self.board[i-1][j] != 0 and self.board[i][j-1] != 0) or (i >= 5 and j == 2 and self.board[i-1][j] != 0) or (i == 4 and j >= 3 and self.board[i][j-1] != 0)):     
                
                if self.board[i][j] == 1:
                    h = self.hold_surrounding(i,j)
                    if h != 0:
                        reward -= 20
                    elif j in [2,11] or i == 23:
                        h2 += 1
                    reward2 += (2 - self.hold_check_horizon(i,j))
                    reward2 += (2 - self.hold_check_vertical(i,j))
        return reward,reward2,h2
    #     direction = [(0,-1),(0,1),(1,0),(-1,0)]
    #     h = 0
    #     for i1,j1 in direction:
    #         if 4 <= i+i1 <= 23 and 2 <= j+j1 <= 11 and h < 4:
    #             temp_i=i+i1
    #             temp_j=j+j1
    #             while(self.board[temp_i+i1][temp_j+j1] == 0):
    #                 if 4 > temp_i+i1 or temp_i+i1 > 23 or 2 > temp_j+j1 or temp_j+j1 > 11:
    #                     break
    #                 temp_i=temp_i+i1
    #                 temp_j=temp_j+j1
    #                 h += 1
    #                 if h >= 4:
    #                     break               
    #     return h % 4
    def hold_surrounding(self,i,j):
        direction = [(0,-1),(0,1)]
        h = 0
        h2 = 0
        for i1,j1 in direction:
            if 2 <= j+j1 <= 11:
                temp_i=i
                temp_j=j+j1
                while(self.board[temp_i-1][temp_j] == 0):
                    temp_i -= 1
                    if temp_i - 1 < 4:
                        break
                if temp_i != 4 and temp_i != i:
                    h += 1
                if 2 <= j+j1 <= 11:
                    break
        if i != 23 and self.board[i+1][j]  == 0:
            h += 1     
        return h
    def hold_check_vertical(self,i,j):
        #0の場合、周辺4マスとどれだけ違うか判定
        direction = [(1,0)]
        sum_hold = 0
        for i1,j1 in direction:
            if self.board[i+i1][j] == 0:
                if i+i1 <= 23:
                    sum_hold += 1
        
        return sum_hold

    def hold_check_horizon(self,i,j):
        #0の場合、周辺4マスとどれだけ違うか判定
        direction = [(0,-1),(0,1)]
        sum_hold = 0
        for i1,j1 in direction:
            if self.board[i][j+j1] == 0:
                if 2 <= j+j1 <= 11: 
                    sum_hold += 1
        
        return sum_hold
    
    def render(self, mode="human"):
        """画面を描画"""
        for row in range(4,24):
            for col in range(2,12):
                rect = pygame.Rect(col * 30, row * 30, 30, 30)
                if self.board[row][col] == -1:
                    pygame.draw.rect(self.screen, (255, 0, 255), rect)  # 緑のブロック
                if self.board[row][col] == -2:
                    pygame.draw.rect(self.screen, (0, 255, 255), rect)  # 緑のブロック
                    #print("AAA")
                elif self.board[row][col] == 1:
                    pygame.draw.rect(self.screen, (0, 255, 0), rect)  # 緑のブロック
                elif self.board[row][col] == 0:
                    pygame.draw.rect(self.screen, (0, 0, 0), rect)
                pygame.draw.rect(self.screen, (255, 255, 255), rect, 1)  # グリッド線（白）

        pygame.display.flip()
        self.clock.tick(10)
    
    def get_state(self):
        """現在の盤面を1D配列で返す"""
        return self.board.flatten()

    def get_new_piece(self):
        """ランダムなブロックを生成"""
        ran = random.randint(0,6)
        fall_block = block(ran,0)
        self.ran_rotate = random.randint(0,1)
        self.col_start = 0
        self.row_start = 5
        return fall_block

    def move_left(self):
        """ブロックを左に動かす"""
        if self.block_judge(-1,0):
            self.board_change(-1,0)

    def move_right(self):
        """ブロックを右に動かす"""
        if self.block_judge(1,0):
            self.board_change(1,0)

    def move_down(self):
        """ブロックを下に動かす"""
        if self.block_judge(0,1):
            self.board_change(0,1)
    def move_up(self):
        while self.block_judge(0,1) == 1:
            self.board_change(0,1)
    def rotate_piece(self):
        """ブロックを回転"""
        if self.block_judge(5,5):
            self.board_change(5,5)

    def clear_lines_judge(self):
        """揃った行を削除し、スコアを計算"""
        judge_num = 0
        for i,col_board in enumerate(self.board):
            if all(col_board[j] in (-1,1) for j in range(2,12)):
                judge_num += 1
        return judge_num
    def clear_lines(self):
        """揃った行を削除し、スコアを計算"""
        for i,col_board in enumerate(self.board):
            if all(col_board[j] == -1 for j in range(2,12)):
                for j in range(2,12):
                    self.board[i][j] = -2
        n = 0
        for i,col_board in enumerate(self.board):
            if all(col_board[j] == -2 for j in range(2,12)):
                if n == 0:
                    self.render()
                    time.sleep(0.3)
                    n = 1
                for k in range(i-1,-1 ,-1):
                    self.board[k+1] = self.board[k]
    def is_game_over(self):
        """ゲームオーバー判定"""
        if any(x == -1 for x in self.board[3]):
            return 1
        return 0
    
    def board_change(self,row_ch,col_ch):
        for i in range(self.col_start,self.col_start+4):
            for j in range(self.row_start,self.row_start+4):
                if self.fall_block.block_board[i-self.col_start][j-self.row_start] == 1:
                    self.board[i][j] = 0
        if row_ch * col_ch > 4:
            self.fall_block.rotate()
        else:
            self.row_start += row_ch
            self.col_start += col_ch
        for i in range(self.col_start,self.col_start+4):
            for j in range(self.row_start,self.row_start+4):
                if self.fall_block.block_board[i-self.col_start][j-self.row_start] == 1:
                    self.board[i][j] = self.fall_block.block_board[i-self.col_start][j-self.row_start]

    def board_change_reward_disappear(self,row_start_bef,col_start_bef):
        for i in range(col_start_bef,col_start_bef+4):
            for j in range(row_start_bef,row_start_bef+4):
                if self.fall_block.block_board[i-col_start_bef][j-row_start_bef] == 1:
                    self.board[i][j] = 0
    def board_change_reward_appear(self,row_start_aft,col_start_aft):
        for i in range(col_start_aft,col_start_aft+4):
            for j in range(row_start_aft,row_start_aft+4):
                if self.fall_block.block_board[i-col_start_aft][j-row_start_aft] == 1:
                    self.board[i][j] = self.fall_block.block_board[i-col_start_aft][j-row_start_aft]

    def block_judge(self,row_ch,col_ch):
        if row_ch * col_ch > 4:
            self.fall_block.rotate()
        else:
            self.row_start += row_ch
            self.col_start += col_ch
        for i in range(self.col_start,self.col_start+4):
            for j in range(self.row_start,self.row_start+4):
                if self.fall_block.block_board[i-self.col_start][j-self.row_start] == 1:
                    if j < 2 or j > 11 or i > 23: #壁や床についたら動けない
                        if row_ch * col_ch > 4:
                            self.fall_block.reverse_rotate()
                        else:
                            self.row_start -= row_ch
                            self.col_start -= col_ch
                        return 0   
                    elif self.board[i][j] == -1:
                        # 移動した先にブロックがあれば動けない
                        if row_ch * col_ch > 4:
                            self.fall_block.reverse_rotate()
                        else:
                            self.row_start -= row_ch
                            self.col_start -= col_ch
                        return 0
        if row_ch * col_ch > 4:
            self.fall_block.reverse_rotate()
        else:
            self.row_start -= row_ch
            self.col_start -= col_ch
        return 1

    def block_judge_reward(self,row_start_aft,col_start_aft):
        for i in range(col_start_aft,col_start_aft+4):
            for j in range(row_start_aft,row_start_aft+4):
                if self.fall_block.block_board[i-col_start_aft][j-row_start_aft] == 1:
                    if j < 2 or j > 11 or i > 23: #壁や床についたら動けない
                        return 0   
                    elif self.board[i][j] == -1:
                        # 移動した先にブロックがあれば動けない
                        return 0
        return 1

    def reverse_degree(self):
        for i in range(self.col_start,self.col_start+4):
            for j in range(self.row_start,self.row_start+4):
                if self.fall_block.block_board[i-self.col_start][j-self.row_start] == 1:
                    self.board[i][j] = -1
# Gym環境として登録
gym.register("Tetris-v1", entry_point=TetrisEnv)

# env = gym.make("Tetris-v0")
# state = env.reset()
# done = False

# while not done:
#     # ランダムなアクション
#     env.block_call()
#     bef_time = time.time() * 1000
#     while env.fall_block.active == True:
#     #action = 2
#         action = env.action_space.sample() 
#         current_time = time.time() * 1000
#         print(current_time,bef_time)
#         if current_time- bef_time > 333:
#             bef_time = current_time
#             if env.block_judge(0,1):
#                     env.board_change(0,1)
#                     env.col_start += 1
#             else:
#                     env.fall_block.active = False
#                     env.reverse_degree()
#                     break
#         next_state, reward, done, _ = env.step(action)
#         env.render()

# env.close()