from typing import Tuple
import jax
import jax.numpy as jnp
from .types import Board, PRNGKey, Array

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


'''
How to generate a new board?
    1. We start by creating an empty board of the specified size.
    2. We then call add_random_cell to place a random cell with a value of 2 or 4.
    3. Finally, we return the newly generated board.
'''
def generate_board(board_size: int, key: PRNGKey) -> Board:
    # Create empty board
    board = jnp.zeros((board_size, board_size), dtype=jnp.int32)

    # Fill one random cell with a value of 2 or 4
    board = add_random_cell(board, key)

    return board

def shift_nonzero_element(carry: Tuple) -> Tuple[Array, int]:
    """True function definition"""
    # Unpack the tuple to get the column, current insertion index, and the index of the element to check
    col, j, i = carry

    # Set the value at the current insertion index to the value of the element at index i
    col = col.at[j].set(col[i])

    # Increment the insertion index for the next potential non-zero element
    j += 1

    return col, j

def shift_column_elements_up(carry: Tuple, i: int) -> Tuple[Array, None]:
    # Unpack the tuple to get the current column and the insertion index
    col, j = carry

    # Conditionally shift the non-zero element at index i to the insertion index
    # If the element at index i is zero, just pass the current state through
    col, j = jax.lax.cond(
        col[i] != 0, # condition
        shift_nonzero_element, # true_fn
        lambda col_j_i: col_j_i[:2], # false_fn
        (col, j, i) # vars
    )

    return (col, j), None

def fill_with_zero(carry: Tuple[Array, int]) -> Tuple[Array, int]:
    """Define bofy_fn of the while loop"""
    # Unpack the tuple to get the current column and the insertion index
    col, j = carry

    # Set the current position j in the column to zero
    col = col.at[j].set(0)

    # Increment the insertion index to move to the next position
    j += 1

    return col, j

def shift_up(col: Array) -> Array:
    # Initialize the insertion index to zero
    j = 0

    # Scan through the column, shifting non-zero elements upwards
    (col, j), _ = jax.lax.scan(
        f=shift_column_elements_up,
        init=(col, j),
        xs=jnp.arange(len(col))
    )

    # After all non-zero elements are shifted up, fill the remaining positions with zeros
    # This while loop continues until all positions from j to the end of the column are set to zero
    col, j = jax.lax.while_loop(
        lambda col_j: col_j[1] < len(col_j[0]), # condition function
        fill_with_zero, # body_fn (true_fn)
        (col, j) #init_vars
    )

    return col

# Handle the merging of two equal adjacent elements. 
# This function will double the value of the first element and set the second element to zero while updating the reward.
def merge_elements(carry: Tuple[jnp.ndarray, float, int]) -> Tuple[jnp.ndarray, float]:
    col, reward, i = carry  # Unpack the input tuple into the column, current reward, and index
    new_col_i = 2 * col[i]  # Double the value of the element at index i
    col = col.at[i].set(new_col_i)  # Set the doubled value at index i
    col = col.at[i + 1].set(0)  # Set the value at index i + 1 to zero
    reward += new_col_i  # Add the value of the merged element to the reward
    return col, reward

# This function will scan through the column and merge adjacent equal elements.
def merge_equal_elements(carry: Tuple[jnp.ndarray, float], i: int) -> Tuple[Tuple[jnp.ndarray, float], None]:
    col, reward = carry
    col, reward = jax.lax.cond(
        (col[i] != 0) & (col[i] == col[i + 1]),  # Condition: elements at index i and i+1 are equal and not zero
        merge_elements,  # Function to call if the condition is true
        lambda col_reward_i: col_reward_i[:2],  # Function to call if the condition is false (pass through)
        (col, reward, i),
    )
    return (col, reward), None

# This function will scan through the column and merge adjacent equal elements.
@jax.jit
def merge_col(col: jnp.ndarray) -> Tuple[jnp.ndarray, float]:
    reward = 0.0
    elements_indices = jnp.arange(len(col) - 1)
    (col, reward), _ = jax.lax.scan(
        f=merge_equal_elements, init=(col, reward), xs=elements_indices
    )
    return col, reward


# This function will move the column up.
def move_up_col(carry: Tuple[jnp.ndarray, float], c: int) -> Tuple[Tuple[jnp.ndarray, float], None]:
    board, additional_reward = carry
    col = board[:, c]
    col = shift_up(col)  # In example: [4, 4, 2, 2] -> [4, 4, 2, 2]
    col, reward = merge_col(col)  # In example: [4, 4, 2, 2] -> [8, 0, 4, 0]
    col = shift_up(col)  # In example: [8, 0, 4, 0] -> [8, 4, 0, 0]
    additional_reward += reward
    return (board.at[:, c].set(col), additional_reward), None

# This function will move the board up.
@jax.jit
def move_up(board: Board) -> Tuple[Board, float]:
    """Move up."""
    additional_reward = 0.0 # Initialize the reward to zero
    col_indices = jnp.arange(board.shape[0])  # Board of size 4 -> [0, 1, 2, 3]
    (board, additional_reward), _ = jax.lax.scan(
        f=move_up_col,
        init=(board, additional_reward),
        xs=col_indices,
    )
    return board, additional_reward

# This function will move the board down.
def move_down(board: jnp.ndarray) -> Tuple[jnp.ndarray, float]:
    """Move down."""
    board, additional_reward = move_up(
        board=jnp.flip(board, 0)
    )
    return jnp.flip(board, 0), additional_reward

# This function will move the board left.
def move_left(board: jnp.ndarray) -> Tuple[jnp.ndarray, float]:
    """Move left."""
    board, additional_reward = move_up(
        board=jnp.rot90(board, k=-1)
    )
    return jnp.rot90(board, k=1), additional_reward

# This function will move the board right.
def move_right(board: jnp.ndarray) -> Tuple[jnp.ndarray, float]:
    """Move right."""
    board, additional_reward = move_up(
        board=jnp.rot90(board, k=1)
    )
    return jnp.rot90(board, k=-1), additional_reward