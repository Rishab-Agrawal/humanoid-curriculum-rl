import numpy as np
import math
import pickle

LEARNING_RATE = 0.1

DISCOUNT = 0.95
prior_reward = 0

num_of_actions = [9]
Observation = [3000, 21, 21] # assuming 150,000 to be the upper limit for the robot reward
np_array_win_size = np.array([50, 0.2, 0.2])

epsilon = 1

epsilon_decay_value = 0.99995

q_table = []

def train(yes_or_no):
    global q_table
    global epsilon

    if yes_or_no == "y":
        q_table = np.random.uniform(low=0, high=1, size=(Observation + num_of_actions))

        # disallowing actions which make x and y gravity go beyond limits of -2 to 2
        for i in range(3000):
            for j in range(21):
                q_table[i][j][0][1] = -100000
                q_table[i][j][0][5] = -100000
                q_table[i][j][0][7] = -100000
                q_table[i][j][20][2] = -100000
                q_table[i][j][20][6] = -100000
                q_table[i][j][20][8] = -100000

                q_table[i][0][j][3] = -100000
                q_table[i][0][j][5] = -100000
                q_table[i][0][j][6] = -100000
                q_table[i][20][j][4] = -100000
                q_table[i][20][j][7] = -100000
                q_table[i][20][j][8] = -100000
    else:
        f = open('env_q_table.pkl', 'rb')
        q_table = pickle.load(f)
        f.close()
        epsilon = 0.1

def get_discrete_state(state):
    discrete_state = state/np_array_win_size + [0, 10, 10]
    return tuple(discrete_state.astype(int))

discrete_state = tuple(np.array([0,0,0]).astype(int))
local_episode = 1

def act():
    if np.random.random() > epsilon:
        action = np.argmax(q_table[discrete_state]) #take cordinated action
    else:
        # taking a random action, but disallowing actions which make x and y gravity go beyond limits of -2 to 2
        action_choices = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        if(discrete_state[2] == 0):
            action_choices.remove(1)
            action_choices.remove(5)
            action_choices.remove(7)
        if(discrete_state[2] == 20):
            action_choices.remove(2)
            action_choices.remove(6)
            action_choices.remove(8)
        if(discrete_state[1] == 0):
            if 3 in action_choices:
                action_choices.remove(3)
            if 5 in action_choices:
                action_choices.remove(5)
            if 6 in action_choices:
                action_choices.remove(6)
        if(discrete_state[1] == 20):
            if 4 in action_choices:
                action_choices.remove(4)
            if 7 in action_choices:
                action_choices.remove(7)
            if 8 in action_choices:
                action_choices.remove(8)
        action = np.random.choice(action_choices)
    
    return action

def upd_q_table(action, new_state, reward):
    global discrete_state
    global epsilon
    global prior_reward
    global local_episode

    new_discrete_state = get_discrete_state(new_state)

    max_future_q = np.max(q_table[new_discrete_state])

    current_q = q_table[discrete_state + (action,)]

    new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)

    q_table[discrete_state + (action,)] = new_q

    discrete_state = new_discrete_state

    if epsilon > 0.05: #epsilon modification
        if reward > prior_reward and local_episode > 5000:
            epsilon = math.pow(epsilon_decay_value, local_episode - 5000)
    
    local_episode += 1
    prior_reward = reward

def get_state_for_no_training(new_state):
    global discrete_state
    global epsilon
    global local_episode

    discrete_state = get_discrete_state(new_state)

    if epsilon > 0.05: #epsilon modification
        if local_episode > 10000:
            epsilon = math.pow(epsilon_decay_value, local_episode - 10000)
    
    local_episode += 1

def save_table():
    f = open('env_q_table.pkl','wb')
    pickle.dump(q_table, f)
    f.close()