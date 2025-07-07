import jax
import jax.numpy as jnp
from .utils import add_random_cell, generate_board, merge_col, move_up, shift_up

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

def test_generate_board(generate_board_fn):
    # Test generate_board
    board_size = 4
    key = jax.random.PRNGKey(0)
    initial_board = generate_board(board_size, key)
    print(initial_board)

def test_shift_up(shift_up_fn):
    board_col = jnp.array([0, 2, 0, 2])
    shift_up_jit = jax.jit(shift_up)
    updated_col = shift_up_jit(board_col)
    print(updated_col)


def test_merge_col(merge_col_fn):
    board_col = jnp.array([0, 2, 0, 2])
    print("Initial column: ",board_col)

    shift_up_jit = jax.jit(shift_up)
    updated_col = shift_up_jit(board_col)
    print("After shifting the elements: ",updated_col)

    updated_col = merge_col(updated_col)
    print("After merging equal elements: ",updated_col[0], "with a reward equals: ",updated_col[1])

def test_move_up(move_up_fn):
    board = jnp.array([
        [4, 2, 2, 0],
        [4, 8, 2, 4],
        [4, 8, 2, 4],
        [4, 8, 0, 0]
    ])
    board, reward = move_up(board)
    print("After moving up: \n",board,"\n Reward: ",reward )

if __name__ == "__main__":
    test_add_random_cell(add_random_cell)
    test_generate_board(generate_board)
    test_shift_up(shift_up)
    test_merge_col(merge_col)
    test_move_up(move_up)