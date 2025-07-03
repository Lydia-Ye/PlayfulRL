# 🎮 PlayfulRL

## Introcution

**PlayfulRL** is a personal reinforcement learning playground where I build JAX-compatible environments for simple games like 2048 and Wuziqi (Five in a Row). The goal is to learn RL and JAX by implementing game logic from scratch and building environments suitable for training agents.

### Game Environments
- `2048 (TODO)`: Grid-based sliding puzzle game (with pure logic and JAX-based step/reset API)
- `Wuziqi (TODO)`: Five-in-a-row board game (WIP)

### JAX-Compatible
All environments are written in a functional style using JAX to enable fast simulation, JIT compilation, and vectorization.


## Project Structure

```

PlayfulRL/
├── src/
│   ├── envs/         # Game-specific environments
│   └── core/         # Shared utilities and types
├── notebooks/        # Step-by-step walkthroughs
├── scripts/          # Simple agent runners
├── tests/            # Unit tests
├── requirements.txt
└── README.md

````

## Setup

```bash
# Clone the repo
git clone https://github.com/your-username/PlayfulRL.git
cd PlayfulRL

# (Optional) Create a virtual environment
python -m venv venv
source venv/bin/activate  # or .\venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt
````