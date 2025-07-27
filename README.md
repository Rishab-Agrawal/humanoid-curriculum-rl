# Adaptive Gravity Curriculum Learning for MuJoCo Humanoid

This project implements **Automated Curriculum Learning** for a MuJoCo Humanoid robot using a custom **Q-Learning-based meta-environment**. The idea is to perturb the robot's gravity conditions (in the x and y directions) during training in a way that optimally challenges and aids the agent’s learning process—an example of learning *how to teach* a reinforcement learning agent.

The Q-learning environment observes the robot’s training performance and learns when to modify the gravitational conditions to improve the agent’s ability to learn walking under diverse perturbations.

📄 **Publication**: [Mitigating the Trade-Off Between Robustness, Optimality and Sample Efficiency in Simulated Legged Locomotion - IEEE Xplore](https://ieeexplore.ieee.org/document/10654466)  
▶️ **Results Video**: [Watch Comparison (MP4)](https://drive.google.com/file/d/12GmX-yZOsXB_hhqO02-xF9PYSJqFJOHF/view?usp=sharing)

---

## Demo

The [Results Video](https://drive.google.com/file/d/12GmX-yZOsXB_hhqO02-xF9PYSJqFJOHF/view?usp=sharing) showcases a side-by-side comparison of humanoid agents trained with and without curriculum learning under varying gravity conditions.


---

## Features

- Custom meta-environment that applies perturbations to gravity.
- Q-Learning for environment-level curriculum scheduling.
- Trains a MuJoCo humanoid agent using **Stable Baselines3** (`SAC`, `TD3`, or `A2C`).
- Logs and saves training checkpoints and Q-tables.

---

## Repository Structure

```
.
├── main_humanoid_custom.py      # Main script for training/testing the RL agent
├── ql_env.py                    # Q-learning logic for the meta-environment (curriculum learning)
├── env_humanoid_v4_custom.py   # Custom wrapper for the Humanoid-v4 environment
├── requirements.txt             # Python dependencies for setting up the environment
├── README.md                    # Project overview and instructions
├── .gitignore                   # Specifies untracked files to ignore in version control
├── LICENSE                      # MIT License for usage and distribution
├── assets/
│   └── xml_humanoid_custom.xml # Custom MuJoCo XML file defining the humanoid model
│
├── models/                      # (Generated) Directory where trained RL models are saved
├── logs/                        # (Generated) Directory for TensorBoard logs
├── env_q_table.pkl              # (Generated) Saved Q-table for curriculum policy

```

---

## How It Works

1. **Base Environment:** The Humanoid-v4 MuJoCo environment.
2. **Meta-Environment:** A Q-table that learns when to apply gravity changes during training.
3. **States to Q-Learning:** `(robot_reward, gravity_x, gravity_y)`
4. **Actions:** Changes in x and y gravity — chosen from 9 discrete pairs.
5. **Reward to Meta-Env:** Based on recent rewards + time-decay to encourage progressive challenges.

---

## Installation & Setup

### 1. Clone the repository

```bash
git clone https://github.com/Rishab-Agrawal/humanoid-curriculum-rl.git
cd humanoid-curriculum-rl
```

### 2. Create and activate a virtual environment (recommended)

```bash
conda create -n humanoid-curriculum python=3.10
conda activate humanoid-curriculum
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

If `gymnasium[mujoco]` fails, try installing manually:

```bash
pip install gymnasium
pip install mujoco
```

---

## Training and Testing

### Train a Humanoid agent with curriculum learning

```bash
python main_humanoid_custom.py Humanoid-v4 SAC -t
```

The program will prompt:

```
Train environment? (y/n):
```

- Type `y` to allow the Q-learning environment to learn the curriculum.

### Test a trained model

```bash
python main_humanoid_custom.py Humanoid-v4 SAC -s ./models/SAC_50000.zip
```

---

## Visualize Training with TensorBoard

```bash
tensorboard --logdir logs/
```

Then open [http://localhost:6006](http://localhost:6006) in your browser.

---

## Notes

- The perturbations are constrained to `[-2, 2]` in both x and y axes to maintain physical realism.
- Gravity is updated every episode based on the Q-table’s selected action.
- `env_q_table.pkl` stores the meta-environment's learned curriculum knowledge.
- Rewards to the Q-environment are based on robot performance and how recently that performance occurred.

---

## Future Extensions

- Replace Q-table with a deep RL model (e.g., DQN or PPO) for more flexible curriculum shaping.
- Add curriculum control over other physics parameters like friction, motor strength, or joint limits.

---

## Author

Rishab Agrawal

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for more details.
