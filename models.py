"""Data models for the Cloud Resource Management Environment."""

from pydantic import Field

try:
    from openenv.core.env_server.types import Action, Observation
except ImportError:
    from openenv.core.env_server.types import Action, Observation


class CloudAction(Action):
    """Action for cloud resource environment - scaling decisions for servers."""
    decisions: str = Field(
        ...,
        description='JSON string mapping server_id to action. Example: {"server_0": "scale_up", "server_1": "maintain"}. Valid actions: scale_up, scale_down, maintain',
    )


class CloudObservation(Observation):
    """Observation from the cloud resource environment."""
    cluster_state: str = Field(default="", description="JSON string with current cluster server metrics")
    task_name: str = Field(default="", description="Current task name")
    timestep: int = Field(default=0, ge=0, description="Current simulation timestep")
    max_timesteps: int = Field(default=0, ge=0, description="Maximum timesteps for task")
    feedback: str = Field(default="", description="Feedback from last action")
    score: float = Field(default=0.0, description="Current cumulative normalized score in [0,1]")
