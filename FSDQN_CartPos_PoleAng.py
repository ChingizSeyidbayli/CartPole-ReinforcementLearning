import gym
import numpy as np
import random
import skfuzzy as fuzz
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

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
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_size)

        # Fuzzify the states to get fuzzy membership values
        fuzzy_states = self.fuzzify_states(state)

        # Compute the degree of membership for each action based on fuzzy rules
        action_mf = self.get_rule_action(fuzzy_states)

        # Select the action with the highest degree of membership
        action_index = np.argmax(action_mf)
        return action_index

    def fuzzify_states(self, state):
        # Fuzzify each state element to get its membership values
        fuzzy_states = []

        # Fuzzify Cart Position
        cart_pos_mf = fuzz.interp_membership(self.cart_pos_range, self.cart_pos_mf, state[0])
        fuzzy_states.append(cart_pos_mf)

        # Fuzzify Cart Velocity
        cart_vel_mf = fuzz.interp_membership(self.cart_vel_range, self.cart_vel_mf, state[1])
        fuzzy_states.append(cart_vel_mf)

        # Fuzzify Pole Angle
        pole_angle_mf = fuzz.interp_membership(self.pole_angle_range, self.pole_angle_mf, state[2])
        fuzzy_states.append(pole_angle_mf)

        # Fuzzify Pole Angular Velocity
        pole_ang_vel_mf = fuzz.interp_membership(self.pole_ang_vel_range, self.pole_ang_vel_mf, state[3])
        fuzzy_states.append(pole_ang_vel_mf)

        return fuzzy_states

    def get_rule_action(self, fuzzy_states):
        # Define fuzzy rules
        fuzzy_rules = [
            # Rule 1: If Cart Position is Low AND Pole Angle is Low, then Action is Move Left
            (np.fmin(fuzzy_states[0], self.low), 0),
            # Rule 2: If Cart Position is Low AND Pole Angle is Medium, then Action is Move Left
            (np.fmin(fuzzy_states[0], self.medium), 0),
            # Rule 3: If Cart Position is Low AND Pole Angle is High, then Action is Move Right
            (np.fmin(fuzzy_states[0], self.high), 1),
            # Rule 4: If Cart Position is Medium AND Pole Angle is Low, then Action is Move Left
            (np.fmin(fuzzy_states[0], self.low), 0),
            # Rule 5: If Cart Position is Medium AND Pole Angle is Medium, then Action is No Movement
            (np.fmin(fuzzy_states[0], self.medium), 2),
            # Rule 6: If Cart Position is Medium AND Pole Angle is High, then Action is Move Right
            (np.fmin(fuzzy_states[0], self.high), 1),
            # Rule 7: If Cart Position is High AND Pole Angle is Low, then Action is Move Right
            (np.fmin(fuzzy_states[0], self.low), 1),
            # Rule 8: If Cart Position is High AND Pole Angle is Medium, then Action is Move Right
            (np.fmin(fuzzy_states[0], self.medium), 1),
            # Rule 9: If Cart Position is High AND Pole Angle is High, then Action is Move Right
            (np.fmin(fuzzy_states[0], self.high), 1)
        ]

        # Compute the degree of membership for each action based on the fuzzy rules
        action_mf = np.zeros(self.action_size)
        for rule in fuzzy_rules:
            action = rule[1]
            action_mf[action] = np.fmax(action_mf[action], rule[0])

        return action_mf

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


def train_agent(agent, episodes=1000, batch_size=32):
    best_reward = 0
    for e in range(episodes):
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        for time in range(500):
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            reward = reward if not done else -10
            next_state = np.reshape(next_state, [1, state_size])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done:
                print("Episode: {}/{}, Score: {}, Epsilon: {:.2f}".format(e, episodes, time, agent.epsilon))
                break
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)
        if time > best_reward:
            best_reward = time
            agent.save("cartpole-dqn.h5")

def test_agent(agent, episodes=100):
    for e in range(episodes):
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        for time in range(500):
            action = agent.act(state)
            next_state, _, done, _ = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])
            state = next_state
            env.render()
            if done:
                break

if __name__ == "__main__":
    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)

    train_agent(agent)
    test_agent(agent)

    env.close()
