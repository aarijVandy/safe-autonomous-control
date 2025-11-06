# Safe Autonomous Vehicle Control

A research project focused on developing and evaluating neural controllers for autonomous vehicles with robustness under adversarial driving conditions.

## 🎯 Project Objectives

1. **Train a neural controller** with good accuracy in normal driving scenarios
2. **Compare performance** against a baseline PID controller
3. **Provide security guarantees** by demonstrating robustness under adversarial driving conditions

## 📁 Project Structure

```
safe-autonomous-control/
├── src/
│   ├── environments/          # NADE-compatible environment wrappers
│   │   ├── __init__.py
│   │   └── nade_wrapper.py   # Highway-env wrapper with NADE features
│   ├── controllers/           # (Coming soon) PID and neural controllers
│   ├── training/             # (Coming soon) RL training infrastructure
│   ├── evaluation/           # (Coming soon) Metrics and comparison
│   └── utils/                # (Coming soon) Helper functions
├── config/
│   └── environment_config.py  # Environment scenario configurations
├── tests/
│   └── test_environment.py    # Environment validation tests
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

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

The environment is based on `highway-env` and configured to match NADE specifications:

- **3-lane highway** with realistic traffic
- **Adversarial behaviors**: cut-ins, hard brakes, slowdowns
- **Longitudinal control focus**: speed/acceleration control (no lateral steering)
- **Gymnasium-compatible interface** for RL training

### Observation Space

5-dimensional continuous observation:

1. **Relative distance** to lead vehicle (m)
2. **Relative velocity** (m/s)
3. **Time-to-collision** (TTC) in seconds
4. **Ego vehicle velocity** (m/s)
5. **Adjacent lane occupancy** (binary: 0 or 1)

### Action Space

1-dimensional continuous action:

- **Acceleration/deceleration** in [-3, 2] m/s²
  - -3 m/s²: Maximum braking
  - 2 m/s²: Maximum acceleration

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
- [x] Documentation

🚧 **In Progress:**
- [ ] PID controller implementation
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

### Code Style

The project follows PEP 8 guidelines with:
- **black** formatter (line length: 100 characters)
- **Type hints** for function signatures
- **Google-style docstrings**

## 📚 Dependencies

### Core Libraries

- **highway-env**: Highway driving simulation
- **gymnasium**: RL environment interface
- **numpy**: Numerical computations
- **pytorch**: Deep learning (for neural controller)
- **stable-baselines3**: RL algorithms (PPO)
- **scipy**: Control algorithms (PID)

### Development Tools

- **pytest**: Testing framework
- **black**: Code formatting
- **pylint/flake8**: Linting

## 🔬 Research Context

This project is based on NADE (Naturalistic and Adversarial Driving Environment) from Michigan Traffic Lab:

- **Paper**: ["Intelligent driving intelligence test for autonomous vehicles with naturalistic and adversarial environment"](https://www.nature.com/articles/s41467-021-21007-8) (Nature Communications, 2021)
- **Original Repository**: [michigan-traffic-lab/NADE](https://github.com/michigan-traffic-lab/Naturalistic-and-Adversarial-Driving-Environment)
- **Base Environment**: [highway-env](https://github.com/Farama-Foundation/HighwayEnv)

## 📄 License

This project is developed for research and educational purposes.

## 🤝 Contributing

This is a research project. For questions or suggestions, please open an issue.

## 📧 Contact

For more information, refer to the project documentation in the `openspec/` directory.

---

**Note**: This is the first phase of the project focusing on environment setup. Subsequent phases will implement controllers, training, and evaluation systems.
