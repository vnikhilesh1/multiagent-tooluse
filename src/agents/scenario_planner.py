"""Scenario Planner Agent for generating conversation scenarios.

This agent takes a tool chain and creates a realistic scenario
describing a user goal that would require those tools.
"""

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from src.agents.base import BaseAgent
from src.models.scenario import Scenario

if TYPE_CHECKING:
    from src.llm import LLMClient
    from src.models.context import ConversationContext
    from src.models.registry import Endpoint, ToolRegistry
    from src.graph.client import GraphClient


@dataclass
class EndpointData:
    """Simple dataclass to hold endpoint data for tool conversion.

    Used when we have sampled chain data but no full Endpoint objects.
    """
    id: str
    name: str
    description: str = ""
    tool_id: str = ""
    domain: str = ""
    parameters: List[Dict[str, Any]] = field(default_factory=list)


class ScenarioPlannerAgent(BaseAgent):
    """Agent that plans conversation scenarios from tool chains.

    This agent analyzes a set of endpoints/tools and generates a realistic
    scenario describing what a user might want to accomplish using those tools.

    Attributes:
        llm: LLMClient for generating scenarios
        name: Agent identifier
        registry: Optional ToolRegistry for endpoint lookup
        graph_client: Optional GraphClient for fetching endpoint details

    Example:
        >>> agent = ScenarioPlannerAgent(llm=client, name="planner")
        >>> context = ConversationContext(tool_chain=["weather_get", "book_flight"])
        >>> context = agent.generate(context)
        >>> print(context.scenario_description)
    """

    def __init__(
        self,
        llm: "LLMClient",
        name: str = "scenario_planner",
        registry: "ToolRegistry" = None,
        graph_client: "GraphClient" = None,
    ) -> None:
        """Initialize the scenario planner agent.

        Args:
            llm: LLMClient instance for LLM calls
            name: Agent identifier
            registry: Optional ToolRegistry for endpoint lookup
            graph_client: Optional GraphClient for fetching endpoint details from graph
        """
        super().__init__(llm=llm, name=name)
        self.registry = registry
        self.graph_client = graph_client

    def generate(self, context: "ConversationContext") -> "ConversationContext":
        """Generate a scenario from the tool chain in context.

        Takes the tool_chain from context, looks up endpoint details,
        and generates a realistic scenario using the LLM.

        Args:
            context: ConversationContext with tool_chain set

        Returns:
            Updated context with scenario_description set
        """
        # Get endpoints - prefer sampled chain data if available
        sampled_chain_data = context.grounding_values.get("sampled_chain_data", [])
        if sampled_chain_data:
            # Use the full endpoint data from sampler
            endpoints = self._get_endpoints_from_chain_data(sampled_chain_data)
        else:
            # Fallback to registry lookup
            endpoints = self._get_endpoints(context.tool_chain)

        # Convert to Anthropic tool format
        tools = self._convert_endpoints_to_tools(endpoints)

        # Build prompt for scenario generation
        prompt = self._build_prompt(endpoints, tools)

        # Generate scenario using LLM
        scenario = self.llm.complete_structured(
            prompt=prompt,
            response_model=Scenario,
            temperature=0.7,
        )

        # Update context with scenario
        context.scenario_description = scenario.description

        # Store scenario details in grounding_values for later use
        context.grounding_values["scenario"] = {
            "user_goal": scenario.user_goal,
            "expected_flow": scenario.expected_flow,
            "disambiguation_points": scenario.disambiguation_points,
            "available_tools": tools,
        }

        return context

    def _get_endpoints(self, tool_chain: List[str]) -> List["Endpoint"]:
        """Look up endpoints from tool chain IDs.

        Args:
            tool_chain: List of endpoint IDs

        Returns:
            List of Endpoint objects (empty list if registry not set)
        """
        if not self.registry:
            return []

        endpoints = []
        for endpoint_id in tool_chain:
            endpoint = self.registry.get_endpoint(endpoint_id)
            if endpoint:
                endpoints.append(endpoint)
        return endpoints

    def _get_endpoints_from_chain_data(
        self, chain_data: List[Dict[str, Any]]
    ) -> List[EndpointData]:
        """Convert sampled chain data to EndpointData objects.

        Uses the graph_client to fetch full endpoint details including
        parameters, which are needed for tool definitions.

        Args:
            chain_data: List of dicts from DFSSampler with endpoint_id, tool_id,
                       domain, completeness_score, name

        Returns:
            List of EndpointData objects with parameters from graph
        """
        endpoints = []

        for item in chain_data:
            endpoint_id = item.get("endpoint_id", "")
            name = item.get("name", endpoint_id)
            tool_id = item.get("tool_id", "")
            domain = item.get("domain", "")
            description = ""
            parameters = []

            # Try to get full details from graph_client
            if self.graph_client:
                ep_data = self.graph_client.get_endpoint_by_id(endpoint_id)
                if ep_data:
                    description = ep_data.get("description", "")
                    name = ep_data.get("name", name)
                    # Parse parameters from JSON if available
                    params_json = ep_data.get("parameters_json", "[]")
                    if params_json:
                        try:
                            parameters = json.loads(params_json)
                        except (json.JSONDecodeError, TypeError):
                            parameters = []

            endpoint = EndpointData(
                id=endpoint_id,
                name=name,
                description=description,
                tool_id=tool_id,
                domain=domain,
                parameters=parameters,
            )
            endpoints.append(endpoint)

        return endpoints

    def _convert_endpoints_to_tools(
        self, endpoints: List[Any]
    ) -> List[Dict[str, Any]]:
        """Convert endpoints to Anthropic tool_use format.

        Args:
            endpoints: List of Endpoint or EndpointData objects

        Returns:
            List of tool definitions in Anthropic format:
            [{"name": "...", "description": "...", "input_schema": {...}}]
        """
        tools = []
        for endpoint in endpoints:
            tool = {
                "name": endpoint.id,
                "description": endpoint.description or f"Call {endpoint.name}",
                "input_schema": self._build_input_schema(endpoint),
            }
            tools.append(tool)
        return tools

    def _build_input_schema(self, endpoint: Any) -> Dict[str, Any]:
        """Build JSON Schema for endpoint parameters.

        Handles both Endpoint objects (with Parameter objects) and
        EndpointData objects (with dict parameters).

        Args:
            endpoint: Endpoint or EndpointData with parameters

        Returns:
            JSON Schema object for input_schema
        """
        properties = {}
        required = []

        # Type mapping for JSON Schema
        type_mapping = {
            "string": "string",
            "integer": "integer",
            "number": "number",
            "boolean": "boolean",
            "array": "array",
            "object": "object",
            "unknown": "string",
        }

        for param in endpoint.parameters:
            # Handle both Parameter objects and dicts
            if isinstance(param, dict):
                # EndpointData: parameters are dicts
                param_name = param.get("name", "")
                param_type_raw = param.get("type", "string")
                # Handle type being a string or having a .value
                if isinstance(param_type_raw, str):
                    param_type = type_mapping.get(param_type_raw.lower(), "string")
                else:
                    param_type = type_mapping.get(getattr(param_type_raw, "value", "string"), "string")
                param_description = param.get("description", f"Parameter {param_name}")
                param_required = param.get("required", False)
                param_enum = param.get("enum")
                param_default = param.get("default")
            else:
                # Endpoint: parameters are Parameter objects
                param_name = param.name
                param_type = type_mapping.get(param.type.value, "string")
                param_description = param.description or f"Parameter {param_name}"
                param_required = param.required
                param_enum = param.enum
                param_default = param.default

            if not param_name:
                continue

            properties[param_name] = {
                "type": param_type,
                "description": param_description,
            }

            if param_enum:
                properties[param_name]["enum"] = param_enum

            if param_default is not None:
                properties[param_name]["default"] = param_default

            if param_required:
                required.append(param_name)

        schema = {
            "type": "object",
            "properties": properties,
        }

        if required:
            schema["required"] = required

        return schema

    def _build_prompt(
        self, endpoints: List[Any], tools: List[Dict[str, Any]]
    ) -> str:
        """Build prompt for scenario generation.

        Args:
            endpoints: List of Endpoint or EndpointData objects
            tools: Tools in Anthropic format

        Returns:
            Prompt string for LLM
        """
        # Format tool information
        tool_descriptions = []
        for endpoint in endpoints:
            # Build parameter string based on type
            params_list = []
            for param in endpoint.parameters:
                if isinstance(param, dict):
                    name = param.get("name", "")
                    ptype = param.get("type", "string")
                    if isinstance(ptype, str):
                        ptype_str = ptype
                    else:
                        ptype_str = getattr(ptype, "value", "string")
                    req = " (required)" if param.get("required", False) else ""
                    params_list.append(f"{name}: {ptype_str}{req}")
                else:
                    req = " (required)" if param.required else ""
                    params_list.append(f"{param.name}: {param.type.value}{req}")

            params_str = ", ".join(params_list)
            tool_descriptions.append(
                f"- {endpoint.id}: {endpoint.description or endpoint.name}\n"
                f"  Parameters: {params_str or 'none'}"
            )

        tools_text = "\n".join(tool_descriptions) if tool_descriptions else "No tools specified"

        prompt = f"""Generate a realistic conversation scenario that would use these tools in order:

Tools available:
{tools_text}

Create a scenario with:
1. A clear description of the situation
2. A specific user goal that naturally requires these tools
3. The expected flow of tool calls (endpoint IDs in order)
4. Any disambiguation points (turn indices where the assistant might need to ask for clarification)

The scenario should feel natural - like a real user request, not a test case.
Focus on realistic use cases that would genuinely benefit from these tools.

Respond with a Scenario object containing:
- description: Brief description of the scenario
- user_goal: What the user wants to accomplish
- expected_flow: List of endpoint IDs in execution order
- disambiguation_points: List of turn indices (0-based) where clarification needed
- available_tools: Leave empty (will be populated separately)
"""
        return prompt

    def __repr__(self) -> str:
        """Return string representation."""
        return f"ScenarioPlannerAgent(name={self.name!r})"
