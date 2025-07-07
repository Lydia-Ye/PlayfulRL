import jax
import jax.numpy as jnp
from .utils import add_random_cell

def test_add_random_cell(add_random_cell_fn):
    try:
        # Sample board before adding a random cell
        board = jnp.array(
            [
                [2,  2,  0,  0],
                [4,  8, 16,  0],
                [16, 0, 32, 0],
                [32, 64, 0, 64]
            ]
        )
        # Random key for stochastic operations
        key = jax.random.PRNGKey(0)

        # Jitted method for adding a random cell
        add_random_cell_jit = jax.jit(add_random_cell_fn)

        # Adding a random cell to the board
        new_board = add_random_cell_jit(board, key)

        # Check if the function ran without errors
        assert new_board is not None, "The function did not return a valid board. Please check your implementation."

        # Check if exactly one zero has been replaced by 2 or 4
        num_zeros_before = jnp.sum(board == 0)
        num_zeros_after = jnp.sum(new_board == 0)
        num_new_tiles = num_zeros_before - num_zeros_after
        unique_values = jnp.unique(new_board)

        assert num_new_tiles == 1, "It looks like your code needs a bit of work. There should be exactly one new tile."
        assert any(tile in unique_values for tile in [2, 4]), "The new tile should be either 2 or 4."
        print("Nice! Your answer looks correct.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    test_add_random_cell(add_random_cell)