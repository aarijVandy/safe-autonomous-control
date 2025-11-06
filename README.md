# Safe Autonomous Vehicle Control

A research project focused on developing and evaluating neural controllers for autonomous vehicles with robustness under adversarial driving conditions.

## 🎯 Project Outline

1. **Train a neural controller** with good accuracy in normal driving scenarios
2. **Compare performance** against a baseline PID controller
3. **Provide security guarantees** by demonstrating robustness under adversarial driving conditions


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

### 3. Basic Usage

```python
from src.environments.nade_wrapper import create_nade_env

# Create environment with normal traffic
env = create_nade_env(adversarial_mode=False)

# Or create with adversarial traffic
env = create_nade_env(
    adversarial_mode=True,
    adversarial_intensity=0.5  # 0.0 to 1.0
)

# Run episode
obs, info = env.reset()
done = truncated = False

while not (done or truncated):
    action = env.action_space.sample()  # Your controller here
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
