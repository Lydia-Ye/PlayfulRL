import jax
import jax.numpy as jnp
from .types import Board, PRNGKey

'''
How to add a new cell to the board?
    1. We start by generating a new key for randomness using jax.random.split, ensuring each step is unpredictable.
    2. Using jnp.argwhere, we find all the empty spots on the board – our blank canvases.
    3. We then use jax.random.choice to randomly select one of these empty spots.
    4. Another jax.random.choice determines whether the new cell will be a 2 or a 4, with probabilities of 90% and 10% respectively.
    5. Finally, we place the new cell on the board with board.at[index].set(value).
'''
def add_random_cell(board: Board, key: PRNGKey) -> Board:
    # Generate a new key
    key, subkey = jax.random.split(key)

    # Select position of the new random cell
    empty_flatten_board = jnp.ravel(board == 0)
    tile_idx = jax.random.choice(
        key, jnp.arange(len(empty_flatten_board)), p=empty_flatten_board
    )
    # Convert the selected tile's location in the flattened array to its position on the board.
    board_size=board.shape[0]
    position = jnp.divmod(tile_idx, board_size)

    # Choose the value of the new cell: 2 with probability 90% or 4 with probability of 10%
    cell_value = jax.random.choice(
        subkey, jnp.array([2, 4]), p=jnp.array([0.9, 0.1])
    )
    board = board.at[position].set(cell_value)

    return board

    # from .utils import add_random_cell