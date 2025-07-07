import jax
import jax.numpy as jnp
from .env import Game2048


def main():
    # Create a random key
    key = jax.random.PRNGKey(0)
    # Create the environment
    env = Game2048()
    # Reset the environment
    state, timestep = env.reset(key)
    print("Initial State:", state)
    print("Initial TimeStep:", timestep)

    # Take a random action (0=up, 1=right, 2=down, 3=left)
    action = int(jax.random.randint(key, (), 0, 4))
    print("Taking action:", action)
    next_state, next_timestep = env.step(state, action)
    print("Next State:", next_state)
    print("Next TimeStep:", next_timestep)

if __name__ == "__main__":
    main() 