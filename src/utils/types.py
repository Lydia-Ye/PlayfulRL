from dataclasses import dataclass, field
from typing import Generic, Dict, TypeVar
import jax.numpy as jnp

Array = jnp.ndarray
Observation = TypeVar('Observation')

# Placeholder for StepType, this should be defined according to the RL framework
StepType = int  # or use an Enum for better clarity

@dataclass
class TimeStep(Generic[Observation]):
    step_type: StepType  # Type of the step (restart, transition, or termination)
    reward: Array  # Reward received at this timestep
    discount: Array  # Discount factor for future rewards
    observation: Observation  # Observation made at this timestep
    extras: Dict = field(default_factory=dict)  # Any additional information 