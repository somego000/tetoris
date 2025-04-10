import gym
import gc
import tetoris_env
from tetoris_dqn import DQNAgent
import time
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model
import tensorflow.keras.backend as K
# 環境のセットアップ
def plot_loss(agent):
    """損失の推移をプロット"""
    plt.plot(agent.loss_history, label="Loss")
    plt.xlabel("Training Steps")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.legend()
    plt.show()

K.clear_session()
gc.collect() 
log_file = "training_log.txt"
loss_file  ="training_loss.txt"
env = gym.make("Tetris-v1")
state_size = env.observation_space.shape[0]  # 状態（盤面の状態）のサイズ
action_size = env.action_space.n  # アクションの種類（移動・回転）

agent = DQNAgent(state_size, action_size)  # DQNエージェントを作成


#agent.model = load_model("tetris_dqn_model_60.h5")
episodes = 8000  # 学習エピソード数
for e in range(episodes):
    state = env.reset()
    done = False
    total_reward = 0
    start_time = time.time() * 1000
    n = 0
    while not done:
        # ① エージェントがアクションを選択
        env.block_call()
        bef_time = time.time() * 1000
        # ② 環境を更新
        while env.fall_block.active == True:
            if n % 10 == 0: 
                env.render()
            n += 1
            current_time = time.time()*1000
            #env.render()
            if current_time - start_time > 600000:
                 print("###################")
                 done = True
                 env.fall_block.acive = False
                 break
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            #print("Initial state:", state.reshape(env.board.shape))
            #agent.remember(state, action, reward, next_state, done)
            state = next_state
            if current_time- bef_time > 333:
                bef_time = current_time
                env.current_time = current_time
                if env.block_judge(0,1):
                        env.board_change(0,1)
                        env.col_start += 1
                else:
                        env.fall_block.active = False
                        env.reverse_degree()
                        break
        # ③ 経験を記憶
        # ④ 状態を更新
        if len(agent.memory) > 100:
            print(f'agent.memory = {len(agent.memory)}')
            agent.replay(batch_size=64)
            print(agent.epsilon)
            with open(loss_file, "a") as f:
                f.write(f"{e+1},{agent.loss_history[-1]}\n") 
            #plot_loss(agent)
        # total_reward += (current_time - start_time) / 20 
        print(f'total_reward = {reward}')
        total_reward += reward
    print(f"エピソード {e+1}/{episodes}, スコア: {total_reward}")
    with open(log_file, "a") as f:
        f.write(f"{e+1},{total_reward}\n") 
    # ⑤ 学習を実行
    # 100エピソードごとに保存
    # if (e + 1) % 20 == 0:
    #     agent.model.save(f"tetris_dqn_model_{e+1}.h5")
    #     print(f"モデルを保存しました: tetris_dqn_model_{e+1}.h5")


env.close()

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