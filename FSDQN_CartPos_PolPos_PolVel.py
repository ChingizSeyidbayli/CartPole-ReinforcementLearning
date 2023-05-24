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
        fuzzy_state = self.fuzzify_state(state)

        # Compute the fuzzy action values based on the fuzzy membership values
        fuzzy_actions = self.compute_fuzzy_actions(fuzzy_state)

        # Choose the action with the highest fuzzy action value
        action_index = np.argmax(fuzzy_actions)
        return action_index

    def fuzzify_state(self, state):
        # Fuzzify the cart position
        cart_position_mf = fuzz.trimf(state[0], [-2.4, 0, 2.4])

        # Fuzzify the pole position
        pole_position_mf = fuzz.trimf(state[1], [-0.4, 0, 0.4])

        # Fuzzify the pole angular velocity
        pole_velocity_mf = fuzz.trimf(state[2], [-0.5, 0, 0.5])

        return cart_position_mf, pole_position_mf, pole_velocity_mf

    def compute_fuzzy_actions(self, fuzzy_state):
        # Compute the fuzzy action values based on the fuzzy membership values of the states
        cart_position_mf, pole_position_mf, pole_velocity_mf = fuzzy_state

        # Define the fuzzy action rules based on the fuzzy membership values of the states
        action_values = []

        # Rule 1: If cart is at left, pole is left, and pole angular velocity is negative, move left with high intensity
        action_values.append(min(cart_position_mf[0], pole_position_mf[0], pole_velocity_mf[0]))

        # Rule 2: If cart is at left, pole is middle, and pole angular velocity is negative, move left with medium intensity
        action_values.append(min(cart_position_mf[0], pole_position_mf[1], pole_velocity_mf[0]))

        # Rule 3: If cart is at left, pole is right, and pole angular velocity is negative, move left with low intensity
        action_values.append(min(cart_position_mf[0], pole_position_mf[2], pole_velocity_mf[0]))

        # Rule 4: If cart is at left, pole is left, and pole angular velocity is zero, move left with medium intensity
        action_values.append(min(cart_position_mf[0], pole_position_mf[0], pole_velocity_mf[1]))

        # Rule 5: If cart is at left, pole is middle, and pole angular velocity is zero, move left with low intensity
        action_values.append(min(cart_position_mf[0], pole_position_mf[1], pole_velocity_mf[1]))

        # Rule 6: If cart is at left, pole is right, and pole angular velocity is zero, move left with low intensity
        action_values.append(min(cart_position_mf[0], pole_position_mf[2], pole_velocity_mf[1]))

        # Rule 7: If cart is at left, pole is left, and pole angular velocity is positive, move left with low intensity
        action_values.append(min(cart_position_mf[0], pole_position_mf[0], pole_velocity_mf[2]))

        # Rule 8: If cart is at left, pole is middle, and pole angular velocity is positive, move left with low intensity
        action_values.append(min(cart_position_mf[0], pole_position_mf[1], pole_velocity_mf[2]))

        # Rule 9: If cart is at left, pole is right, and pole angular velocity is positive, move left with low intensity
        action_values.append(min(cart_position_mf[0], pole_position_mf[2], pole_velocity_mf[2]))

        # Rule 10: If cart is in the middle, pole is left, and pole angular velocity is negative, move left with medium intensity
        action_values.append(min(cart_position_mf[1], pole_position_mf[0], pole_velocity_mf[0]))

        # Rule 11: If cart is in the middle, pole is middle, and pole angular velocity is negative, move left with low intensity
        action_values.append(min(cart_position_mf[1], pole_position_mf[1], pole_velocity_mf[0]))

        # Rule 12: If cart is in the middle, pole is right, and pole angular velocity is negative, move right with low intensity
        action_values.append(min(cart_position_mf[1], pole_position_mf[2], pole_velocity_mf[0]))

        # Rule 13: If cart is in the middle, pole is left, and pole angular velocity is zero, move left with low intensity
        action_values.append(min(cart_position_mf[1], pole_position_mf[0], pole_velocity_mf[1]))

        # Rule 14: If cart is in the middle, pole is middle, and pole angular velocity is zero, do not move
        action_values.append(min(cart_position_mf[1], pole_position_mf[1], pole_velocity_mf[1]))

        # Rule 15: If cart is in the middle, pole is right, and pole angular velocity is zero, move right with low intensity
        action_values.append(min(cart_position_mf[1], pole_position_mf[2], pole_velocity_mf[1]))

        # Rule 16: If cart is in the middle, pole is left, and pole angular velocity is positive, move right with low intensity
        action_values.append(min(cart_position_mf[1], pole_position_mf[0], pole_velocity_mf[2]))

        # Rule 17: If cart is in the middle, pole is middle, and pole angular velocity is positive, move right with low intensity
        action_values.append(min(cart_position_mf[1], pole_position_mf[1], pole_velocity_mf[2]))

        # Rule 18: If cart is in the middle, pole is right, and pole angular velocity is positive, move right with low intensity
        action_values.append(min(cart_position_mf[1], pole_position_mf[2], pole_velocity_mf[2]))

        # Rule 19: If cart is at right, pole is left, and pole angular velocity is negative, move right with low intensity
        action_values.append(min(cart_position_mf[2], pole_position_mf[0], pole_velocity_mf[0]))

        # Rule 20: If cart is at right, pole is middle, and pole angular velocity is negative, move right with low intensity
        action_values.append(min(cart_position_mf[2], pole_position_mf[1], pole_velocity_mf[0]))

        # Rule 21: If cart is at right, pole is right, and pole angular velocity is negative, move right with low intensity
        action_values.append(min(cart_position_mf[2], pole_position_mf[2], pole_velocity_mf[0]))

        # Rule 22: If cart is at right, pole is left, and pole angular velocity is zero, move right with low intensity
        action_values.append(min(cart_position_mf[2], pole_position_mf[0], pole_velocity_mf[1]))

        # Rule 23: If cart is at right, pole is middle, and pole angular velocity is zero, move right with low intensity
        action_values.append(min(cart_position_mf[2], pole_position_mf[1], pole_velocity_mf[1]))

        # Rule 24: If cart is at right, pole is right, and pole angular velocity is zero, move right with medium intensity
        action_values.append(min(cart_position_mf[2], pole_position_mf[2], pole_velocity_mf[1]))

        # Rule 25: If cart is at right, pole is left, and pole angular velocity is positive, move right with low intensity
        action_values.append(min(cart_position_mf[2], pole_position_mf[0], pole_velocity_mf[2]))

        # Rule 26: If cart is at right, pole is middle, and pole angular velocity is positive, move right with medium intensity
        action_values.append(min(cart_position_mf[2], pole_position_mf[1], pole_velocity_mf[2]))

        # Rule 27: If cart is at right, pole is right, and pole angular velocity is positive, move right with high intensity
        action_values.append(min(cart_position_mf[2], pole_position_mf[2], pole_velocity_mf[2]))

        return action_values

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


if __name__ == "__main__":
    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)

    done = False
    batch_size = 32
    best_reward = 0
    for e in range(1000):  # number of episodes to train
        state = env.reset()
        state = np.reshape(state, (1, state_size))
        for time in range(500):
            env.render()
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            reward = reward if not done else -10
            next_state = np.reshape(next_state, (1, state_size))
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done:
                print("episode: {}/{}, score: {}, epsilon: {:.2}"
                      .format(e, 1000, time, agent.epsilon))
                break
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)
            if time > best_reward:
                best_reward = time
                agent.save("best_model.h5")
