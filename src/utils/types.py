from dataclasses import dataclass, field
from typing import Generic, Dict, TypeVar
import jax.numpy as jnp
from jumanji.types import StepType

Array = jnp.ndarray
Observation = TypeVar('Observation')

@dataclass
class TimeStep(Generic[Observation]):
    step_type: StepType  # Type of the step (restart, transition, or termination)
    reward: Array  # Reward received at this timestep
    discount: Array  # Discount factor for future rewards
    observation: Observation  # Observation made at this timestep
    extras: Dict = field(default_factory=dict)  # Any additional information 