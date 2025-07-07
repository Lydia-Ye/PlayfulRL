from typing import Optional, Tuple, Sequence
import jax
import jax.numpy as jnp
from jumanji.env import Environment
from jumanji import specs
from jumanji.types import TimeStep, restart, termination, transition
from .types import State, PRNGKey, Observation, Board, Array
from .viewer import Game2048Viewer
from .utils import move_up, move_down, move_left, move_right

class Game2048(Environment[State, specs.DiscreteArray, Observation]):

    def __init__(self, board_size: int = 4, env_viewer: Optional[Game2048Viewer] = None) -> None:
        """ Initializes the game environment with a specified board size and an optional viewer for rendering."""
        self.board_size = board_size
        # Create viewer used for rendering
        self._env_viewer = env_viewer or Game2048Viewer("2048", board_size)

    def __repr__(self) -> str:
        """String representation of the environment."""
        return f"2048 Game(board_size={self.board_size})"

    def reset(self, key: PRNGKey) -> Tuple[State, TimeStep[Observation]]:
        """Resets the environment."""

        key, board_key = jax.random.split(key)
        board = self._generate_board(board_key)
        action_mask = self._get_action_mask(board)

        obs = Observation(board=board, action_mask=action_mask)
        timestep = restart(observation=obs)

        state = State(
            board=board,
            step_count=jnp.int32(0),
            action_mask=action_mask,
            key=key,
            score=jnp.array(0, float),
        )
        return state, timestep

    def step(self, state: State, action: int) -> Tuple[State, TimeStep[Observation]]:
        """Updates the environment state after the agent takes an action."""

        # Take the action in the environment: Up, Right, Down, Left.
        updated_board, additional_reward = jax.lax.switch(
            action,
            [move_up, move_right, move_down, move_left],
            state.board,
        )

        # Generate action mask to keep in the state for the next step and
        # to provide to the agent in the observation.
        action_mask = self._get_action_mask(board=updated_board)

        # Check if the episode terminates (i.e. there are no legal actions).
        done = ~jnp.any(action_mask)

        # Generate new key.
        random_cell_key, new_state_key = jax.random.split(state.key)

        # Update the state of the board by adding a new random cell.
        updated_board = jax.lax.cond(
            done,
            lambda board, pkey: board,
            self._add_random_cell,
            updated_board,
            random_cell_key,
        )
        # Build the state.
        state = State(
            board=updated_board,
            action_mask=action_mask,
            step_count=state.step_count + 1,
            key=new_state_key,
            score=state.score + additional_reward.astype(float),
        )
        # Generate the observation from the environment state.
        observation = Observation( board=updated_board, action_mask=action_mask)

        # Return either a MID or a LAST timestep depending on done.
        timestep = jax.lax.cond(
            done,
            termination,
            transition,
            additional_reward,
            observation,
        )
        return state, timestep

    def _generate_board(self, key: PRNGKey) -> Board:
        """Generates an initial board for the environment."""

        # Create empty board
        board = jnp.zeros((self.board_size, self.board_size), dtype=jnp.int32)

        # Fill one random cell with a value of 2 or 4
        board = self._add_random_cell(board, key)

        return board

    def _add_random_cell(self, board: Board, key: PRNGKey) -> Board:
        """Adds a new random cell to the board."""
        key, subkey = jax.random.split(key)

        # Select position of the new random cell
        empty_flatten_board = jnp.ravel(board == 0)
        tile_idx = jax.random.choice(
            key, jnp.arange(len(empty_flatten_board)), p=empty_flatten_board
        )
        # Convert the selected tile's location in the flattened array to its position on the board.
        position = jnp.divmod(tile_idx, self.board_size)

        # Choose the value of the new cell: 2 with probability 90% or 4 with probability of 10%
        cell_value = jax.random.choice(
            subkey, jnp.array([2, 4]), p=jnp.array([0.9, 0.1])
        )
        board = board.at[position].set(cell_value)

        return board

    def _get_action_mask(self, board: Board) -> Array:
        """Generates a binary mask indicating which actions are valid. """
        action_mask = jnp.array(
            [
                jnp.any(move_up(board)[0] != board),
                jnp.any(move_right(board)[0] != board),
                jnp.any(move_down(board)[0] != board),
                jnp.any(move_left(board)[0] != board),
            ],
        )
        return action_mask

    def observation_spec(self) -> specs.Spec:
      """Specifications of the observation of the `Game2048` environment."""
      return specs.Spec(
          Observation,
          "ObservationSpec",
          board=specs.Array(
              shape=(self.board_size, self.board_size),
              dtype=jnp.int32,
              name="board",
          ),
          action_mask=specs.BoundedArray(
              shape=(4,),
              dtype=bool,
              minimum=False,
              maximum=True,
              name="action_mask",
          ),
      )

    def action_spec(self) -> specs.DiscreteArray:
        """Returns the action spec."""
        return specs.DiscreteArray(4, name="action")

    def render(self, state: State, save: bool = True, path: str = "./2048.png") -> None:
        """Renders the current state of the game board."""
        self._env_viewer.render(state=state, save=save, path=path)

    def animate(
        self,
        states: Sequence[State],
        interval: int = 200,
        save_path: Optional[str] = None,
    ):
        """Creates an animated gif of the 2048 game board based on the sequence of game states."""
        return self._env_viewer.animate(
            states=states, interval=interval, save_path=save_path
        )