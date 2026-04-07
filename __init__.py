"""
Cloud GPU+CPU Resource Management Environment – an OpenEnv environment for
cloud GPU and CPU resource scaling, thermal management, and heuristic
fragmentation powered by simulated cloud workload patterns.

Example:
    >>> from cloud_resource_env import CloudResourceClient
    >>>
    >>> with CloudResourceClient(base_url="http://localhost:8000").sync() as env:
    ...     env.reset(task="gpu_cpu_allocation")
    ...     state = env.call_tool("get_cluster_state")
    ...     result = env.call_tool("take_action", decisions='{"node_0": "maintain"}')
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
