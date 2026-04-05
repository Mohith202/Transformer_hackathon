"""Cloud Resource Management Environment Client."""

from openenv.core.mcp_client import MCPToolClient


class CloudResourceClient(MCPToolClient):
    """
    Client for the Cloud Resource Management Environment.

    Inherits all functionality from MCPToolClient:
    - list_tools(): Discover available tools
    - call_tool(name, **kwargs): Call a tool by name
    - reset(**kwargs): Reset the environment
    - step(action): Execute an action

    Example (async):
        >>> async with CloudResourceClient(base_url="http://localhost:8000") as env:
        ...     await env.reset(task="single_server_scaling")
        ...     state = await env.call_tool("get_cluster_state")
        ...     result = await env.call_tool("take_action", decisions='{"server_0": "scale_up"}')

    Example (sync):
        >>> with CloudResourceClient(base_url="http://localhost:8000").sync() as env:
        ...     env.reset(task="single_server_scaling")
        ...     state = env.call_tool("get_cluster_state")
    """

    pass
