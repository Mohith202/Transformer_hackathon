"""
Cloud Resource Management Environment – an OpenEnv environment for
cloud GPU/CPU resource scaling powered by real-world data patterns.

Example:
    >>> from cloud_resource_env import CloudResourceClient
    >>>
    >>> with CloudResourceClient(base_url="http://localhost:8000").sync() as env:
    ...     env.reset(task="single_server_scaling")
    ...     state = env.call_tool("get_cluster_state")
    ...     result = env.call_tool("take_action", decisions='{"server_0": "maintain"}')
"""

from openenv.core.env_server.mcp_types import CallToolAction, ListToolsAction

from .client import CloudResourceClient
from .models import CloudAction, CloudObservation

__all__ = [
    "CloudResourceClient",
    "CloudAction",
    "CloudObservation",
    "CallToolAction",
    "ListToolsAction",
]
