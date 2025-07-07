# 🎮 PlayfulRL

## Introduction

**PlayfulRL** is a personal reinforcement learning playground where I build JAX-compatible environments for simple games like 2048 and Wuziqi (Five in a Row). The goal is to learn RL and JAX by implementing game logic from scratch and building environments suitable for training agents.


## Game Environments

### 1. 2048 
A grid-based sliding puzzle game with pure JAX logic and JAX-based step/reset API.

**Features:**
- Pure JAX implementation for fast computation
- Jumanji RL environment interface compatibility
- Automatic action masking to prevent invalid moves
- Built-in score tracking based on tile merges
- Visualization support (static images and animated GIFs)
- Functional design for easy parallelization

**Game Rules:**
- Slide tiles in four directions (Up, Right, Down, Left)
- Tiles with the same value merge when they collide
- New tiles (2 or 4) appear randomly in empty cells after each move
- Game ends when no valid moves are possible
- Score increases with each successful merge

## Project Structure

```
PlayfulRL/
├── src/
│   ├── envs/                   # Game environments
│   │   ├── game_2048/          # 2048 implementation
│   │   │   ├── env.py          # Main environment class
│   │   │   ├── types.py        # Type definitions
│   │   │   ├── utils.py        # Game logic utilities
│   │   │   ├── viewer.py       # Visualization components
│   │   │   ├── demo_2048.py    # Interactive demo
│   │   │   ├── run_env.py      # Environment runner
│   │   │   └── test.py         # Unit tests
│   │   └── wuziqi/             
│   └── utils/                  # Shared utilities and types
├── notebooks/                  
├── requirements.txt            # Python dependencies
├── .gitignore
└── README.md
```

## Quick Start

### Prerequisites
```bash
# Install from requirements.txt
pip install -r requirements.txt

# Or install manually
pip install jax jaxlib jumanji pygame pillow numpy
```

### Basic Usage for 2048

#### 1. Programmatic Usage
```python
import jax
from src.envs.game_2048.env import Game2048

# Initialize and run environment
env = Game2048(board_size=4)
key = jax.random.PRNGKey(0)
state, timestep = env.reset(key)

while not timestep.last():
    valid_actions = jax.numpy.where(state.action_mask)[0]
    action = valid_actions[0]  # Take first valid action
    state, timestep = env.step(state, action)
    print(f"Score: {state.score}")
```


#### 2. Demo with random agent
```bash
cd src/envs/game_2048
python run_env.py
```

#### 3. Interactive Game Demo
```bash
cd src/envs/game_2048
python demo_2048.py
```
**Controls:** `w` (up), `a` (left), `s` (down), `d` (right), `q` (quit)
