import numpy as np
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from collections import deque

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)  # 過去の経験を保存するメモリ
        self.gamma = 0.95    # 割引率（長期的な報酬を考慮する）
        self.epsilon = 1.00   # 探索率（最初はランダムに行動）
        self.epsilon_min = 0.005
        self.epsilon_decay = 0.999
        self.learning_rate = 0.001
        self.model = self._build_model()
        self.loss_history = []
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print("GPU が使用可能です。")
            except RuntimeError as e:
                print(e)
        else:
            print("GPU が見つかりません。")
    def _build_model(self):
        """Qネットワークの構築"""
        model = Sequential([
            Dense(256, input_dim=self.state_size, activation="relu"),
            Dense(256, activation="relu"),
            Dense(self.action_size, activation="linear")
        ])
        model.compile(loss="mse", optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        """経験をメモリに保存"""
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        """ε-greedy法で行動を決定"""
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)  # ランダム行動
        q_values = self.model.predict(state.reshape(1, -1), verbose=0)
        print(f'q_values = {q_values},{np.argmax(q_values[0])}')
        return np.argmax(q_values[0])  # 最大のQ値を持つアクションを選択

    def replay(self, batch_size=32):
        epochs = 5
        print("######i#)")
        """経験から学習"""
        minibatch = random.sample(self.memory, batch_size)
        total_loss = 0
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target += self.gamma * np.max(self.model.predict(next_state.reshape(1, -1), verbose=0)[0])
            target_f = self.model.predict(state.reshape(1, -1), verbose=0)
            target_f[0][action] = target
            history = self.model.fit(state.reshape(1, -1), target_f, epochs=epochs, verbose=0)  
            total_loss += history.history["loss"][-1]  #
            self.model.fit(state.reshape(1, -1), target_f, epochs=epochs, verbose=0)
        avg_loss = total_loss / batch_size  # 🔹 平均損失
        self.loss_history.append(avg_loss)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
