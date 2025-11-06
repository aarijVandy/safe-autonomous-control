# Safe Autonomous Vehicle Control

A research project focused on developing and evaluating neural controllers for autonomous vehicles with robustness under adversarial driving conditions.

## 🎯 Project Outline

1. **Train a neural controller** with good accuracy in normal driving scenarios
2. **Compare performance** against a baseline PID controller
3. **Provide security guarantees** by demonstrating robustness under adversarial driving conditions

## 🎥 Demo

https://github.com/aarijVandy/safe-autonomous-control/assets/YOUR_USER_ID/pid_normal_kp0.5_ki0.1_kd0.2-episode-0.mp4

The PID controller successfully:
- Maintains safe following distance
- Avoids collisions

> **Note**: To generate your own demo videos, use: `python tests/test_pid_controller.py --scenario normal --episodes 1 --record-video`


## 🚀 Quick Start

### 1. Installation

First, create a virtual environment and install dependencies:

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Test the Environment

Run the environment test to verify the setup:

```bash
# Run all unit tests
python tests/test_environment.py --test-only

# Test with normal traffic scenario
python tests/test_environment.py --scenario normal --episodes 3

# Test with adversarial traffic
python tests/test_environment.py --scenario severe --episodes 3 --render
```

### 3. Test PID Controller

```bash
# Test PID controller with visualization
python tests/test_pid_controller.py --scenario normal --episodes 3 --render

# Test without visualization (faster)
python tests/test_pid_controller.py --scenario normal --episodes 5

# Record video of simulation (saved to media/ folder)
python tests/test_pid_controller.py --scenario normal --episodes 1 --record-video

# Test with custom PID gains
python tests/test_pid_controller.py --kp 0.8 --ki 0.15 --kd 0.3 --episodes 5
```

### 3a. Visualize PID Performance (Recommended!)

```bash
# Generate comprehensive performance plots (6 plots: speed, distance, TTC, action, jerk, reward)
python scripts/visualize_pid.py --scenario normal --kp 0.5 --ki 0.1 --kd 0.2

# With live rendering + plots
python scripts/visualize_pid.py --scenario normal --kp 0.5 --ki 0.1 --kd 0.2 --render

# Save plots to file
python scripts/visualize_pid.py --scenario normal --kp 0.8 --ki 0.15 --kd 0.3 --save pid_performance.png

# Compare multiple PID configurations
python scripts/visualize_pid.py --scenario normal --compare
```

See **[VISUALIZATION_QUICK_REF.md](VISUALIZATION_QUICK_REF.md)** for quick reference or **[docs/VISUALIZATION_GUIDE.md](docs/VISUALIZATION_GUIDE.md)** for complete guide.

### 4. Optimize PID Parameters

```bash
# Optimize PID controller for normal traffic (grid search)
python src/training/optimize_pid.py --scenario normal --method grid_search --episodes 5

# Quick optimization with random search
python src/training/optimize_pid.py --scenario normal --method random_search --iterations 30 --episodes 5

# Optimize for adversarial scenarios
python src/training/optimize_pid.py --scenario moderate --method grid_search --episodes 5
```

See **[docs/PID_TRAINING_GUIDE.md](docs/PID_TRAINING_GUIDE.md)** for complete training guide.

### 5. Basic Usage (Environment)

```python
from src.environments.nade_wrapper import create_nade_env
from src.controllers.pid_controller import PIDController

# Create environment with normal traffic
env = create_nade_env(adversarial_mode=False)

# Create PID controller
controller = PIDController(kp=0.5, ki=0.1, kd=0.2)

# Run episode
obs, info = env.reset()
controller.reset()
done = truncated = False

while not (done or truncated):
    action = controller.compute_action(obs)  # PID control
    obs, reward, done, truncated, info = env.step(action)

env.close()
```

## 🏗️ Environment Features

### NADE-Compatible Highway Environment

The environment is based on `highway-env` and configured to match NADE specifications

### Scenario Configurations

Four pre-configured scenarios are available:

| Scenario | Adversarial Mode | Intensity | Description |
|----------|------------------|-----------|-------------|
| **normal** | No | 0.0 | Normal highway traffic with cooperative vehicles |
| **mild** | Yes | 0.3 | Occasional aggressive behaviors |
| **moderate** | Yes | 0.5 | Frequent aggressive behaviors |
| **severe** | Yes | 0.8 | Very frequent aggressive behaviors |

## 📊 Current Status

✅ **Completed:**
- [x] Project structure setup
- [x] NADE-compatible environment wrapper
- [x] Environment configuration system
- [x] Basic testing framework
- [x] **PID controller implementation**
- [x] **PID parameter optimization (grid search & random search)**
- [x] **Performance metrics (response time, collision avoidance, jerk minimization)**
- [x] Documentation

🚧 **In Progress:**
- [ ] Neural controller (PPO) training
- [ ] Adversarial training pipeline
- [ ] Evaluation metrics and comparison

## 🔧 Development

### Running Tests

```bash
# Run environment tests only
python tests/test_environment.py --test-only

# Run with visualization
python tests/test_environment.py --scenario normal --render

# Test different scenarios
python tests/test_environment.py --scenario mild --episodes 5
python tests/test_environment.py --scenario moderate --episodes 5
python tests/test_environment.py --scenario severe --episodes 5
```


### Core Libraries

- **highway-env**: Highway driving simulation
- **gymnasium**: RL environment interface
- **numpy**: Numerical computations
- **pytorch**: Deep learning (for neural controller)
- **stable-baselines3**: RL algorithms (PPO)
- **scipy**: Control algorithms (PID)

## 🔬 Research Context

This project is based on NADE (Naturalistic and Adversarial Driving Environment) from Michigan Traffic Lab:

- **Paper**: ["Intelligent driving intelligence test for autonomous vehicles with naturalistic and adversarial environment"](https://www.nature.com/articles/s41467-021-21007-8) (Nature Communications, 2021)
- **Original Repository**: [michigan-traffic-lab/NADE](https://github.com/michigan-traffic-lab/Naturalistic-and-Adversarial-Driving-Environment)
- **Base Environment**: [highway-env](https://github.com/Farama-Foundation/HighwayEnv)
