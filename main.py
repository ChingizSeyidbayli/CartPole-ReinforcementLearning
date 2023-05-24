import gym
import numpy as np
import random
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import skfuzzy as fuzz
import matplotlib.pyplot as plt

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Dense(24, input_shape=(self.state_size,), activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        # Convert the state to fuzzy representation
        fuzzy_state = self.fuzzify_state(state)

        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_size)
        act_values = self.model.predict(fuzzy_state)
        return np.argmax(act_values[0])

    def fuzzify_state(self, state):
        fuzzy_state = np.zeros((state.shape[0], self.state_size * 3))

        for i in range(self.state_size):
            state_min = np.min(state[:, i])
            state_max = np.max(state[:, i])
            mid = (state_min + state_max) / 2

            fuzzy_state[:, i*3] = fuzz.trimf(state[:, i], [state_min, mid, state_max])
            fuzzy_state[:, i*3+1] = fuzz.trimf(state[:, i], [state_min+(mid-state_min)/2, mid, mid+(state_max-mid)/2])
            fuzzy_state[:, i*3+2] = fuzz.trimf(state[:, i], [mid, state_max-(state_max-mid)/2, state_max])

        return fuzzy_state

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self, name):
        self.model.save_weights(name)

    def load(self, name):
        self.model.load_weights(name)


def plot_fuzzy_membership_functions(state):
    state_size = state.shape[0]
    x = np.linspace(-10, 10, 1000)

    # Define membership functions
    membership_functions = [
        ['Low', [-10, -5, 0]],
        ['Medium', [-5, 0, 5]],
        ['High', [0, 5, 10]]
    ]

    # Plot the membership functions
    fig, ax = plt.subplots(state_size, figsize=(8, 6))

    for i in range(state_size):
        ax[i].set_title('State Element {}'.format(i+1))
        ax[i].spines['top'].set_visible(False)
        ax[i].spines['right'].set_visible(False)
        ax[i].get_xaxis().tick_bottom()
        ax[i].get_yaxis().tick_left()

        for mf in membership_functions:
            membership_name, membership_range = mf
            membership = fuzz.trimf(x, membership_range)
            ax[i].plot(x, membership, label=membership_name)

        ax[i].legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Create environment and agent
    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)
    done = False
    batch_size = 32
    best_reward = 0
    # Plot fuzzy membership functions before training
    state = np.random.rand(state_size)
    plot_fuzzy_membership_functions(state)

    # Training loop
    for episode in range(10):
        state = env.reset()[0]
        state = np.reshape(state, (4,1))
        print(state.shape)
        state = np.reshape(state, (1, state_size))

        for time in range(500):
            env.render()
            action = agent.act(state)
            next_state, reward, done, _, _ = env.step(action)
            reward = reward if not done else -10
            next_state = np.reshape(next_state, [1, state_size])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done:
                print("episode: {}/{}, score: {}, e: {:.2}"
                      .format(episode, 1000, time, agent.epsilon))
                break
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)
        if time > best_reward:
            best_reward = time
            agent.save("cartpole-dqn_fs.h5")
        if e % 2 == 0:
            agent.save("cartpole-dqn.h5")

    env.close()
