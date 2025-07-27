# to train: python filename Humanoid-v4 SAC -t
# to test: python filename Humanoid-v4 SAC -s .\models\SAC_num.zip

import gymnasium as gym
from stable_baselines3 import SAC, TD3, A2C
import os
import argparse
import stable_baselines3.common.callbacks as callbacks
from stable_baselines3.common.utils import safe_mean

from env_humanoid_v4_custom import CustomHumanoidEnv

import ql_env

# Create directories to hold models and logs
model_dir = "models"
log_dir = "logs"
os.makedirs(model_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)
my_ep_num = 0
env_reward_data = [[0, -1], [0, -1], [0, -1], [0, -1], [0, -1], [0, -1], [0, -1], [0, -1], [0, -1], [0, -1], [0, -1]]
ep_dist_weight = 0
cur_env_action = 0
prev_ep_rew = 0

class CustomCallback(callbacks.BaseCallback):

    def __init__(self, env, first_env_action, train_env, verbose=0):
        super(CustomCallback, self).__init__(verbose)
        self.env = env
        self.first_env_action = first_env_action
        self.train_env = train_env

    def _on_step(self) -> bool:
        global my_ep_num
        global env_reward_data
        global ep_dist_weight
        global cur_env_action
        global prev_ep_rew

        ep_dist_weight += 0.000005 # encouraging back-to-back execution on different gravity ranges over time

        if (len(self.model.ep_info_buffer) > 1 and self.model.ep_info_buffer[-1]["r"] != prev_ep_rew) or self.model.ep_info_buffer == 1: # ensuring that episode changed, since this function executes at every step.
            my_ep_num += 1
            prev_ep_rew = self.model.ep_info_buffer[-1]["r"]
            last_episode_reward = safe_mean([ep_info["r"] for ep_info in self.model.ep_info_buffer]) # reward to the robot, as used by the SAC algorithm

            x_grav = self.env.model.opt.gravity[0]
            y_grav = self.env.model.opt.gravity[1]

            new_state_for_env = [last_episode_reward, x_grav, y_grav] # defined the state of the environment for which it will check the q table for the action

            if(self.train_env == "y"):
                # Saving the latest reward and episode number of certain gravity states which will be used to calculate the reward for the q learning env.
                temp_data = [last_episode_reward, my_ep_num]
                
                if x_grav >= -0.2 and x_grav <= 0.2:
                    if y_grav <= -1.8: # [0, -2], i.e., x axis gravity is roughly 0, y axis gravity is roughly -2. (m/s^2)
                        env_reward_data[0] = temp_data
                    if y_grav <= -0.8 and y_grav >= -1.2: # [0, -1]
                        env_reward_data[1] = temp_data
                    if y_grav >= -0.2 and y_grav <= 0.2: # [0, 0]
                        env_reward_data[2] = temp_data
                    if y_grav >= 0.8 and y_grav <= 1.2: # [0, 1]
                        env_reward_data[3] = temp_data
                    if y_grav >= 1.8: # [0, 2]
                        env_reward_data[4] = temp_data
                
                elif y_grav >= -0.2 and y_grav <= 0.2:
                    if x_grav <= -1.8: # [-2, 0]
                        env_reward_data[5] = temp_data
                    if x_grav <= -0.8 and x_grav >= -1.2: # [-1, 0]
                        env_reward_data[6] = temp_data
                    if x_grav >= 0.8 and x_grav <= 1.2: # [1, 0]
                        env_reward_data[7] = temp_data
                    if x_grav >= 1.8: # [2, 0]
                        env_reward_data[8] = temp_data

                else:
                    if x_grav >= 0.8 and x_grav <= 1.2 and y_grav >= -1.2 and y_grav <= -0.8: # [1, -1]
                        env_reward_data[9] = temp_data
                    if y_grav >= 0.8 and y_grav <= 1.2 and x_grav >= -1.2 and x_grav <= -0.8: # [-1, 1]
                        env_reward_data[10] = temp_data

                reward_for_env = 0
                for saved_state in env_reward_data: # adding latest rewards of the specific states and subtracting how long ago they were experienced.
                    if saved_state[1] == -1:
                        continue
                    reward_for_env += saved_state[0]
                    reward_for_env -= ep_dist_weight*(my_ep_num - saved_state[1]) # final reward for the environment q learning algorithm
                
                if my_ep_num == 1:
                    ql_env.upd_q_table(self.first_env_action, new_state_for_env, reward_for_env) # first action taken before this function was called first
                else:
                    ql_env.upd_q_table(cur_env_action, new_state_for_env, reward_for_env) # cur action taken and saved in the last call to this function
                
            else:
                ql_env.get_state_for_no_training(new_state_for_env) # if using a saved q table for the env, just send current state to env, reward not required
            
            cur_env_action = ql_env.act() # get appropriate action for state given to the environment
            grav_upd = action_to_grav_upd(cur_env_action) #action converted to gravity changes using the function defined below

            # update gravity by adding changes (as directed by env action) and rounding off
            self.env.model.opt.gravity[0] += grav_upd[0]
            self.env.model.opt.gravity[1] += grav_upd[1]
            self.env.model.opt.gravity[0] = round(self.env.model.opt.gravity[0], 1)
            self.env.model.opt.gravity[1] = round(self.env.model.opt.gravity[1], 1)

            print()
            print("**********************************************************************************************************")
            print("Training env:", self.train_env)
            print("My episode number:", my_ep_num)
            print()
            print("Reward to robot:", last_episode_reward)
            print("State for env (Reward to robot, gravity x, gravity y):", new_state_for_env)
            if self.train_env == "y":
                print("Temporary, env reward data:", env_reward_data)
                print("Reward for env:", reward_for_env)
            print("Action from env (gravity x update, gravity y update):", grav_upd)
            print("New gravity values (x, y):", self.env.model.opt.gravity[0], self.env.model.opt.gravity[1])
            print("**********************************************************************************************************")
            print()

        return True  # Continue training

def action_to_grav_upd(action): # mapping of action integer to corresponding changes in x and y gravity
    if action == 0:
        gravity_upd = [0, 0]
    elif action == 1:
        gravity_upd = [0, -0.2]
    elif action == 2:
        gravity_upd = [0, 0.2]
    elif action == 3:
        gravity_upd = [-0.2, 0]
    elif action == 4:
        gravity_upd = [0.2, 0]
    elif action == 5:
        gravity_upd = [-0.2, -0.2]
    elif action == 6:
        gravity_upd = [-0.2, 0.2]
    elif action == 7:
        gravity_upd = [0.2, -0.2]
    else:
        gravity_upd = [0.2, 0.2]
    
    return gravity_upd

def train(env, sb3_algo, train_env):
    match sb3_algo:
        case 'SAC':
            model = SAC('MlpPolicy', env, verbose=1, device='cuda', tensorboard_log=log_dir)
        case 'TD3':
            model = TD3('MlpPolicy', env, verbose=1, device='cuda', tensorboard_log=log_dir)
        case 'A2C':
            model = A2C('MlpPolicy', env, verbose=1, device='cuda', tensorboard_log=log_dir)
        case _:
            print('Algorithm not found')
            return

    TIMESTEPS = 25000
    iters = 0

    while True:
        iters += 1

        if (iters == 1):
            ql_env.train(train_env) # initialization of q table based on whether training mode on or off

            # first action to be performed by the environment, without any rewards
            first_env_action = ql_env.act()
            grav_upd = action_to_grav_upd(first_env_action)
            env.model.opt.gravity[0] += grav_upd[0]
            env.model.opt.gravity[1] += grav_upd[1]
            env.model.opt.gravity[0] = round(env.model.opt.gravity[0], 1)
            env.model.opt.gravity[1] = round(env.model.opt.gravity[1], 1)

        model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, callback=CustomCallback(env, first_env_action, train_env))
        model.save(f"{model_dir}/{sb3_algo}_{TIMESTEPS*iters}")
        ql_env.save_table() #saving env q table every 25000 timesteps, along with the SAC algo stuff for the humanoid

def test(env, sb3_algo, path_to_model):

    match sb3_algo:
        case 'SAC':
            model = SAC.load(path_to_model, env=env)
        case 'TD3':
            model = TD3.load(path_to_model, env=env)
        case 'A2C':
            model = A2C.load(path_to_model, env=env)
        case _:
            print('Algorithm not found')
            return

    obs = env.reset()[0]
    done = False
    extra_steps = 500
    cur_steps = 0
    env.model.opt.gravity[:] = [-1, 1, -9.81] # for a constant specific gravity, uncomment this line
    while True:
        action, _ = model.predict(obs)
        obs, _, done, _, _ = env.step(action)

        # for simulating pushes, uncomment these lines and comment the if done block
        # cur_steps += 1

        # if cur_steps > 200:
        #     env.model.opt.gravity[:] = [0, 0, -9.81] # last tests: backward push, -1.25, forward push 1.25, left push 1.5, right push -1.5

        # if cur_steps > 300:
        #     env.model.opt.gravity[:] = [0, 0, -9.81]
        #     cur_steps = 0

        if done:
            extra_steps -= 1

            if extra_steps < 0:
                break


if __name__ == '__main__':

    # Parse command line inputs
    parser = argparse.ArgumentParser(description='Train or test model.')
    parser.add_argument('gymenv', help='Gymnasium environment i.e. Humanoid-v4')
    parser.add_argument('sb3_algo', help='StableBaseline3 RL algorithm i.e. SAC, TD3')
    parser.add_argument('-t', '--train', action='store_true')
    parser.add_argument('-s', '--test', metavar='path_to_model')
    args = parser.parse_args()


    if args.train:
        print("Train environment?")
        train_env = input("(y/n): ")
        gymenv = CustomHumanoidEnv(render_mode=None)
        # gymenv = gym.make(args.gymenv, render_mode=None)
        train(gymenv, args.sb3_algo, train_env)

    if(args.test):
        if os.path.isfile(args.test):
            gymenv = CustomHumanoidEnv(render_mode='human')
            # gymenv = gym.make(args.gymenv, render_mode='human')
            test(gymenv, args.sb3_algo, path_to_model=args.test)
        else:
            print(f'{args.test} not found.')
