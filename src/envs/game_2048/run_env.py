import jax
import jax.numpy as jnp
from .env import Game2048


def main():
    # Initialize the environment
    game2048 = Game2048(board_size=4)

    # Generate a random key for the environment
    key = jax.random.PRNGKey(0)

    # Reset the environment to get the initial state and timestep
    state, timestep = game2048.reset(key)

    # Print the initial state and timestep
    print("Initial State:", state)
    print("Initial Timestep:", timestep)

    # Initialize a list to store the states for later visualization
    states = [state]

    # Variable to accumulate the total reward
    total_reward = jnp.float32(0)

    # Interaction loop until the game ends
    while not timestep.last():
        # Split the key for randomness in action selection
        action_key, key = jax.random.split(key)

        # Randomly select an action based on the action mask
        action = jax.random.choice(action_key, jnp.arange(4), p=state.action_mask.flatten())

        # Take the action and update the state and timestep
        state, timestep = game2048.step(state, action)

        # Accumulate the reward
        total_reward += timestep.reward

        # Store the state for visualization
        states.append(state)

    # Print the total reward after the game ends
    print("Total Reward:", total_reward)

    game2048.animate(states=states, interval=400)

if __name__ == "__main__":
    main() 