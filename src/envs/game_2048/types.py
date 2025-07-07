from typing import NamedTuple, TypeAlias
import jax.numpy as jnp

Array: TypeAlias = jnp.ndarray
Numeric: TypeAlias = float  # or use jnp.float32 if you want to be more specific
PRNGKey: TypeAlias = Array  # or use jax.random.PRNGKeyArray if available

Board: TypeAlias = Array

class State(NamedTuple):
    board: Board
    step_count: Numeric
    action_mask: Array
    key: PRNGKey
    score: Numeric

class Observation(NamedTuple):
    board: Board
    action_mask: Array