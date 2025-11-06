# Project Context

## Purpose
Develop and evaluate a neural controller for autonomous vehicles and demonstrate its robustness under adversarial driving conditions. The project focuses on three core goals:

**Key Objectives:**
1. **Train a neural controller** with good accuracy in normal driving scenarios
2. **Compare performance** against a baseline PID controller
3. **Provide security guarantees** by training and testing against adversarial driving conditions (sudden brakes, lane changes, slowdowns)

**Scope:**
- Longitudinal vehicle control (speed/acceleration control)
- NADE simulation environment as primary testbed
- Focus on practical implementation over theoretical verification

## Tech Stack

### Core Simulation & Environment
- **NADE (Naturalistic-and-Adversarial Driving Environment)** - Primary and only testbed
- **Python 3.8+** - Primary programming language

### Machine Learning & RL
- **PyTorch** or **TensorFlow** - Deep learning framework
- **Stable-Baselines3** or **Ray RLlib** - RL algorithm implementations (PPO, DQN)
- **Gymnasium (OpenAI Gym)** - RL environment interfaces
- **NumPy** - Numerical computations
- **Pandas** - Data analysis and metrics tracking

### Control & Planning
- **scipy** - For baseline PID controller implementation

### Safety & Visualization
- **matplotlib/seaborn** - Visualization of safety metrics and trajectories
- **tensorboard** - Training monitoring and visualization

### Development & Testing
- **pytest** - Unit and integration testing
- **black** - Code formatting
- **pylint/flake8** - Linting
- **jupyter** - Exploratory analysis and visualization
- **git** - Version control

## Project Conventions

### Code Style
- Follow **PEP 8** style guidelines
- Use **black** formatter with line length of 100 characters
- Use type hints for function signatures
- Docstrings: Google style format
- Naming conventions:
  - Classes: `PascalCase` (e.g., `SafetyShield`, `NeuralController`)
  - Functions/methods: `snake_case` (e.g., `calculate_ttc`, `train_policy`)
  - Constants: `UPPER_SNAKE_CASE` (e.g., `MIN_DISTANCE`, `MAX_ACCELERATION`)
  - Private methods: prefix with `_` (e.g., `_clip_action`)

### Architecture Patterns

**Simplified Design:**
```
src/
├── controllers/        # PID baseline and neural controller
├── environments/       # NADE wrapper
├── training/          # RL training with adversarial scenarios
├── evaluation/        # Metrics and comparison analysis
└── utils/             # Helper functions, config management
```

**Key Patterns:**
- **Strategy Pattern** - Interchangeable controllers (PID vs Neural)
- **Observer Pattern** - Logging and metrics collection

### Testing Strategy

**Testing Levels:**
1. **Unit Tests** - Core controller logic and dynamics
2. **Integration Tests** - Controller + environment interactions
3. **Evaluation Tests** - Performance comparison between controllers

**Key Test Scenarios:**
- PID controller stability and tuning validation
- Neural controller training convergence
- Adversarial scenario handling (cut-ins, hard brakes, slowdowns)
- Comparative performance metrics

### Git Workflow
- **Main Branch:** `main` - stable, tested code
- **Development Branch:** `dev` - integration branch
- **Feature Branches:** `feature/<description>` (e.g., `feature/safety-shield`)
- **Experiment Branches:** `exp/<description>` (e.g., `exp/ppo-adversarial`)

**Commit Conventions:**
- Use conventional commits format: `<type>(<scope>): <description>`
- Types: `feat`, `fix`, `docs`, `test`, `refactor`, `perf`, `chore`
- Examples:
  - `feat(controller): add neural policy with PPO`
  - `fix(safety): correct TTC calculation for zero velocity`
  - `test(eval): add ablation study for adversary intensity`

## Domain Context

### Autonomous Vehicle Control
- **Longitudinal Control:** Focus on speed/acceleration control (not lateral steering)
- **Ego Vehicle:** The autonomous vehicle being controlled
- **Adversarial Agents:** Background traffic with aggressive/stochastic behaviors

### Safety Metrics
- **TTC (Time-to-Collision):** Time until collision if current velocities maintained
- **Minimum Distance:** Safe following distance
- **Collision Rate:** Frequency of accidents per episode
- **Success Rate:** Percentage of episodes completed without collision

### Control Concepts
- **PID Controller:** Proportional-Integral-Derivative feedback control (baseline)
- **Neural Controller:** Deep RL policy (PPO-based)
- **Adversarial Training:** Training with challenging traffic scenarios

### Observation Space
- Relative distance to lead vehicle
- Relative velocity (speed difference)
- Time-to-collision
- Ego vehicle velocity
- (Optional) Adjacent lane occupancy for lane-change adversaries

### Action Space
- Continuous: acceleration/deceleration (with actuator limits, e.g., [-3, 2] m/s²)
- Discrete alternative: {hard_brake, soft_brake, maintain, accelerate}

### Training Philosophy
- **Reward Design:** Balance progress, safety (avoid collisions), and smooth control
- **Adversarial Training:** Train directly on mix of normal and adversarial scenarios
- **Security Focus:** Demonstrate robustness guarantees under adversarial conditions

## Important Constraints

### Technical Constraints
- **Actuator Limits:** Acceleration bounded to realistic values (e.g., -5 to +3 m/s²)
- **Observation Latency:** Real-world sensor delays (may model 50-100ms lag)
- **Computational Budget:** Neural policy must run at 10-20 Hz for real-time control
- **Simulation Timestep:** Typically 0.05-0.1 seconds

### Safety Constraints
- **C1:** Minimum safe distance from lead vehicle
- **C2:** No rear-end collisions
- **C3:** Smooth acceleration (avoid excessive jerk)

### Project Timeline Constraints
- Phased development: baseline PID → neural training → adversarial testing
- Focus on practical implementation and comparison
- Single simulation platform (NADE only)

### Performance Requirements
- **Safety:** High success rate (>95%) in normal scenarios, demonstrable robustness in adversarial
- **Accuracy:** Neural controller should match or exceed PID baseline
- **Security:** Maintain acceptable performance under adversarial conditions

## External Dependencies

### Simulation Platforms
- **NADE Toolbox** - Primary and only simulation environment
  - Provides highway environment with adversarial traffic generation
  - Built-in intelligent background agents
  - Open-source, Python-based

### Libraries & Frameworks
- **Stable-Baselines3** - https://stable-baselines3.readthedocs.io/
  - PPO implementation for neural controller
  - Well-documented, actively maintained

### Hardware Requirements
- GPU recommended for neural network training (any modern GPU)
- Standard laptop/desktop CPU sufficient
- ~20GB storage for logs and model checkpoints
