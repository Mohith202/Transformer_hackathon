"""Data models for the Cloud GPU+CPU Resource Management Environment."""

from pydantic import Field

try:
    from openenv.core.env_server.types import Action, Observation
except ImportError:
    from openenv.core.env_server.types import Action, Observation


class CloudAction(Action):
    """Action for cloud GPU+CPU resource environment.

    Decisions are task-specific JSON strings mapping node_id to action:

    Task 1 (gpu_cpu_allocation):
        Actions: allocate_high, allocate_low, maintain, migrate
        Example: {"node_0": "allocate_high", "node_1": "maintain"}

    Task 2 (thermal_management):
        Actions: increase_cooling, decrease_cooling, migrate_load, maintain
        Example: {"node_0": "increase_cooling", "node_1": "migrate_load"}

    Task 3 (heuristic_fragmentation):
        Actions: best_fit, first_fit, compact, split_workload
        Example: {"node_0": "best_fit", "node_1": "compact"}
    """

    decisions: str = Field(
        ...,
        description=(
            'JSON string mapping node_id to action. '
            'Task-specific valid actions — see task_info for the current task.'
        ),
    )


class CloudObservation(Observation):
    """Observation from the cloud GPU+CPU resource environment."""

    cluster_state: str = Field(
        default="",
        description="JSON string with current cluster node metrics (GPU, CPU, thermal, fragmentation)",
    )
    task_name: str = Field(default="", description="Current task name")
    timestep: int = Field(default=0, ge=0, description="Current simulation timestep")
    max_timesteps: int = Field(default=0, ge=0, description="Maximum timesteps for task")
    feedback: str = Field(default="", description="Feedback from last action")
    score: float = Field(default=0.0, description="Current cumulative normalised score in [0,1]")
