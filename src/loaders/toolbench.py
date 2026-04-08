"""ToolBench data loader.

Parses ToolBench JSON files into Tool, Endpoint, and Parameter models.
Handles multiple data formats and normalizes them to a consistent structure.
"""

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from tqdm import tqdm

from src.logging_config import get_logger
from src.models import (
    Endpoint,
    Parameter,
    Tool,
    ToolRegistry,
    infer_domain,
    normalize_type_string,
    sanitize_id,
    truncate_description,
)

logger = get_logger(__name__)


class ToolBenchLoader:
    """Loader for ToolBench JSON files.

    Parses tool definitions from ToolBench dataset files and converts
    them to Tool, Endpoint, and Parameter model instances.

    Handles multiple JSON formats:
    - api_list: Array of API definitions with required/optional parameters
    - endpoints: Array of endpoint definitions with parameters dict
    - apis: Array of API definitions with different field names

    Example:
        >>> loader = ToolBenchLoader()
        >>> registry = loader.load_directory(Path("data/toolbench"))
        >>> print(f"Loaded {registry.tool_count} tools")

        >>> # Load single file
        >>> tool = loader.load_file(Path("data/toolbench/weather.json"))

        >>> # Load with limit for testing
        >>> registry = loader.load_directory(Path("data"), limit=10)
    """

    def __init__(
        self,
        max_description_length: int = 500,
        default_method: str = "GET",
        skip_malformed: bool = True,
    ):
        """Initialize the loader.

        Args:
            max_description_length: Maximum description length before truncation
            default_method: Default HTTP method when not specified
            skip_malformed: Whether to skip malformed entries (True) or raise (False)
        """
        self._max_description_length = max_description_length
        self._default_method = default_method
        self._skip_malformed = skip_malformed

    def load_directory(
        self,
        directory: Path,
        limit: Optional[int] = None,
        show_progress: bool = True,
        pattern: str = "*.json",
        recursive: bool = True,
    ) -> ToolRegistry:
        """Load all JSON files from a directory.

        Args:
            directory: Path to directory containing JSON files
            limit: Maximum number of tools to load (None for all)
            show_progress: Whether to show progress bar
            pattern: Glob pattern for JSON files
            recursive: Whether to search subdirectories (default True)

        Returns:
            ToolRegistry containing all loaded tools

        Raises:
            FileNotFoundError: If directory doesn't exist
        """
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")

        registry = ToolRegistry()

        # Use rglob for recursive search, glob for non-recursive
        if recursive:
            files = list(directory.rglob(pattern))
        else:
            files = list(directory.glob(pattern))

        if limit:
            files = files[:limit]

        iterator = tqdm(files, desc="Loading tools", disable=not show_progress)

        for file_path in iterator:
            try:
                tool = self.load_file(file_path)
                if tool:
                    registry.add_tool(tool)
            except Exception as e:
                if self._skip_malformed:
                    logger.warning(f"Skipping {file_path}: {e}")
                else:
                    raise

        return registry

    def load_file(self, file_path: Path) -> Optional[Tool]:
        """Load a single JSON file as a Tool.

        Args:
            file_path: Path to JSON file

        Returns:
            Tool instance, or None if file couldn't be parsed

        Raises:
            FileNotFoundError: If file doesn't exist
        """
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Extract category from parent directory name if not in data
            parent_dir = file_path.parent.name
            if parent_dir and parent_dir not in ("tools", "toolbench", "data"):
                # Use directory name as category hint
                data["_directory_category"] = parent_dir.replace("_", " ")

            return self._parse_tool(data, source_file=str(file_path))
        except json.JSONDecodeError as e:
            if self._skip_malformed:
                logger.warning(f"Invalid JSON in {file_path}: {e}")
                return None
            raise
        except Exception as e:
            if self._skip_malformed:
                logger.warning(f"Error parsing {file_path}: {e}")
                return None
            raise

    def _parse_tool(
        self, data: Dict[str, Any], source_file: Optional[str] = None
    ) -> Optional[Tool]:
        """Parse a tool definition from JSON data.

        Detects the format (api_list, endpoints, or apis) and delegates
        to the appropriate parser.

        Args:
            data: Parsed JSON data
            source_file: Source filename for ID generation

        Returns:
            Tool instance, or None if parsing failed
        """
        try:
            # Detect format and delegate
            if "api_list" in data:
                return self._parse_api_list_format(data, source_file)
            elif "endpoints" in data:
                return self._parse_endpoints_format(data, source_file)
            elif "apis" in data:
                return self._parse_apis_format(data, source_file)
            else:
                # Try to parse as api_list format with empty list
                return self._parse_api_list_format(data, source_file)
        except Exception as e:
            if self._skip_malformed:
                logger.warning(f"Error parsing tool data: {e}")
                return None
            raise

    def _parse_api_list_format(
        self, data: Dict[str, Any], source_file: Optional[str] = None
    ) -> Tool:
        """Parse tool with api_list array format.

        Expected format:
        {
            "tool_name": "...",
            "category_name": "...",
            "api_list": [...]
        }
        """
        tool_id = self._extract_tool_id(data, source_file)
        tool_name = data.get("tool_name") or data.get("name") or tool_id
        # Priority: category_name > category > directory category > Uncategorized
        category = (
            data.get("category_name")
            or data.get("category")
            or data.get("_directory_category")
            or "Uncategorized"
        )
        description = data.get("description") or data.get("tool_description") or ""
        api_host = data.get("api_host") or data.get("host") or data.get("url") or ""

        endpoints = []
        seen_endpoint_ids = set()
        for api_data in data.get("api_list", []):
            endpoint = self._parse_endpoint_from_api_list(api_data, tool_id, category)
            if endpoint and endpoint.id not in seen_endpoint_ids:
                endpoints.append(endpoint)
                seen_endpoint_ids.add(endpoint.id)

        return Tool(
            id=tool_id,
            name=str(tool_name),
            category=str(category),
            description=truncate_description(
                str(description), self._max_description_length
            ),
            api_host=str(api_host),
            endpoints=endpoints,
            raw_schema=data,
        )

    def _parse_endpoints_format(
        self, data: Dict[str, Any], source_file: Optional[str] = None
    ) -> Tool:
        """Parse tool with endpoints array format.

        Expected format:
        {
            "id": "...",
            "name": "...",
            "category": "...",
            "endpoints": [...]
        }
        """
        tool_id = self._extract_tool_id(data, source_file)
        tool_name = data.get("name") or data.get("tool_name") or tool_id
        category = (
            data.get("category")
            or data.get("category_name")
            or data.get("_directory_category")
            or "Uncategorized"
        )
        description = data.get("description") or ""
        api_host = data.get("api_host") or data.get("host") or ""

        endpoints = []
        seen_endpoint_ids = set()
        for endpoint_data in data.get("endpoints", []):
            endpoint = self._parse_endpoint_from_endpoints(
                endpoint_data, tool_id, category
            )
            if endpoint and endpoint.id not in seen_endpoint_ids:
                endpoints.append(endpoint)
                seen_endpoint_ids.add(endpoint.id)

        return Tool(
            id=tool_id,
            name=str(tool_name),
            category=str(category),
            description=truncate_description(
                str(description), self._max_description_length
            ),
            api_host=str(api_host),
            endpoints=endpoints,
            raw_schema=data,
        )

    def _parse_apis_format(
        self, data: Dict[str, Any], source_file: Optional[str] = None
    ) -> Tool:
        """Parse tool with apis array format.

        Expected format:
        {
            "tool_name": "...",
            "apis": [...]
        }
        """
        tool_id = self._extract_tool_id(data, source_file)
        tool_name = data.get("tool_name") or data.get("name") or tool_id
        category = (
            data.get("category")
            or data.get("category_name")
            or data.get("_directory_category")
            or "Uncategorized"
        )
        description = data.get("description") or data.get("tool_description") or ""
        api_host = data.get("api_host") or data.get("host") or ""

        endpoints = []
        seen_endpoint_ids = set()
        for api_data in data.get("apis", []):
            endpoint = self._parse_endpoint_from_apis(api_data, tool_id, category)
            if endpoint and endpoint.id not in seen_endpoint_ids:
                endpoints.append(endpoint)
                seen_endpoint_ids.add(endpoint.id)

        return Tool(
            id=tool_id,
            name=str(tool_name),
            category=str(category),
            description=truncate_description(
                str(description), self._max_description_length
            ),
            api_host=str(api_host),
            endpoints=endpoints,
            raw_schema=data,
        )

    def _parse_endpoint_from_api_list(
        self,
        api_data: Dict[str, Any],
        tool_id: str,
        tool_category: str,
    ) -> Optional[Endpoint]:
        """Parse an endpoint from api_list format.

        Handles required_parameters and optional_parameters arrays.
        """
        name = api_data.get("name") or api_data.get("api_name")
        if not name:
            return None

        endpoint_id = self._generate_endpoint_id(tool_id, str(name))
        method = self._normalize_method(api_data.get("method"))
        path = api_data.get("url") or api_data.get("path") or ""
        description = api_data.get("description") or api_data.get("api_description") or ""

        # Parse parameters from required_parameters and optional_parameters
        parameters = []
        required_params = api_data.get("required_parameters", [])
        optional_params = api_data.get("optional_parameters", [])

        if required_params:
            parameters.extend(
                self._parse_parameters_from_list(required_params, required=True)
            )
        if optional_params:
            parameters.extend(
                self._parse_parameters_from_list(optional_params, required=False)
            )

        # Also check for generic 'parameters' field
        if "parameters" in api_data:
            params = api_data["parameters"]
            if isinstance(params, list):
                parameters.extend(self._parse_parameters_from_list(params))
            elif isinstance(params, dict):
                parameters.extend(self._parse_parameters_from_dict(params))

        # Infer domain from category or path
        domain = infer_domain(
            category=tool_category,
            path=str(path),
            name=str(name),
            description=str(description),
        )

        return Endpoint(
            id=endpoint_id,
            tool_id=tool_id,
            name=str(name),
            method=method,
            path=str(path),
            description=truncate_description(
                str(description), self._max_description_length
            ),
            parameters=parameters,
            domain=domain,
            raw_schema=api_data,
        )

    def _parse_endpoint_from_endpoints(
        self,
        endpoint_data: Dict[str, Any],
        tool_id: str,
        tool_category: str,
    ) -> Optional[Endpoint]:
        """Parse an endpoint from endpoints format.

        Handles parameters as a dict with name as key.
        """
        name = endpoint_data.get("name") or endpoint_data.get("endpoint_name")
        if not name:
            return None

        endpoint_id = self._generate_endpoint_id(tool_id, str(name))
        method = self._normalize_method(endpoint_data.get("method"))
        path = endpoint_data.get("path") or endpoint_data.get("url") or ""
        description = endpoint_data.get("description") or ""

        # Parse parameters from dict format
        parameters = []
        if "parameters" in endpoint_data:
            params = endpoint_data["parameters"]
            if isinstance(params, dict):
                parameters = self._parse_parameters_from_dict(params)
            elif isinstance(params, list):
                parameters = self._parse_parameters_from_list(params)

        # Infer domain from category or path
        domain = infer_domain(
            category=tool_category,
            path=str(path),
            name=str(name),
            description=str(description),
        )

        return Endpoint(
            id=endpoint_id,
            tool_id=tool_id,
            name=str(name),
            method=method,
            path=str(path),
            description=truncate_description(
                str(description), self._max_description_length
            ),
            parameters=parameters,
            domain=domain,
            raw_schema=endpoint_data,
        )

    def _parse_endpoint_from_apis(
        self,
        api_data: Dict[str, Any],
        tool_id: str,
        tool_category: str,
    ) -> Optional[Endpoint]:
        """Parse an endpoint from apis format.

        Handles api_name, api_description, parameter_name, etc.
        """
        name = api_data.get("api_name") or api_data.get("name")
        if not name:
            return None

        endpoint_id = self._generate_endpoint_id(tool_id, str(name))
        method = self._normalize_method(api_data.get("method"))
        path = api_data.get("path") or api_data.get("url") or ""
        description = api_data.get("api_description") or api_data.get("description") or ""

        # Parse parameters
        parameters = []
        if "parameters" in api_data:
            params = api_data["parameters"]
            if isinstance(params, list):
                parameters = self._parse_parameters_from_list(params)
            elif isinstance(params, dict):
                parameters = self._parse_parameters_from_dict(params)

        # Infer domain from category or path
        domain = infer_domain(
            category=tool_category,
            path=str(path),
            name=str(name),
            description=str(description),
        )

        return Endpoint(
            id=endpoint_id,
            tool_id=tool_id,
            name=str(name),
            method=method,
            path=str(path),
            description=truncate_description(
                str(description), self._max_description_length
            ),
            parameters=parameters,
            domain=domain,
            raw_schema=api_data,
        )

    def _parse_parameters_from_list(
        self,
        params: List[Dict[str, Any]],
        required: bool = False,
    ) -> List[Parameter]:
        """Parse parameters from a list of parameter dicts.

        Handles formats:
        - {"name": "...", "type": "...", "description": "..."}
        - {"parameter_name": "...", "parameter_type": "..."}
        """
        result = []
        for p in params:
            # Handle different field names
            name = p.get("name") or p.get("parameter_name") or p.get("param_name")
            if not name:
                continue

            type_str = p.get("type") or p.get("parameter_type") or p.get("param_type")
            description = (
                p.get("description") or p.get("parameter_description") or ""
            )

            # Check if required is specified in the param itself
            param_required = p.get("required", required)
            if isinstance(param_required, str):
                param_required = param_required.lower() == "true"

            result.append(
                Parameter(
                    name=str(name),
                    type=normalize_type_string(type_str),
                    description=truncate_description(
                        str(description), self._max_description_length
                    ),
                    required=bool(param_required),
                    default=p.get("default"),
                    enum=p.get("enum"),
                )
            )

        return result

    def _parse_parameters_from_dict(
        self,
        params: Dict[str, Dict[str, Any]],
    ) -> List[Parameter]:
        """Parse parameters from a dict with name as key.

        Handles format:
        {"param_name": {"type": "...", "required": true, "default": ...}}
        """
        result = []
        for name, param_data in params.items():
            if not isinstance(param_data, dict):
                continue

            type_str = param_data.get("type") or param_data.get("parameter_type")
            description = param_data.get("description") or ""

            param_required = param_data.get("required", False)
            if isinstance(param_required, str):
                param_required = param_required.lower() == "true"

            result.append(
                Parameter(
                    name=str(name),
                    type=normalize_type_string(type_str),
                    description=truncate_description(
                        str(description), self._max_description_length
                    ),
                    required=bool(param_required),
                    default=param_data.get("default"),
                    enum=param_data.get("enum"),
                )
            )

        return result

    def _generate_endpoint_id(self, tool_id: str, endpoint_name: str) -> str:
        """Generate a unique endpoint ID.

        Uses tool_id and endpoint_name to create a deterministic ID.
        """
        try:
            sanitized_name = sanitize_id(endpoint_name)
        except ValueError:
            # If endpoint name can't be sanitized, use a hash
            sanitized_name = hashlib.md5(endpoint_name.encode()).hexdigest()[:12]

        return f"{tool_id}_{sanitized_name}"

    def _extract_tool_id(
        self,
        data: Dict[str, Any],
        source_file: Optional[str] = None,
    ) -> str:
        """Extract or generate a tool ID.

        Priority:
        1. 'id' field
        2. 'tool_id' field
        3. Sanitized 'tool_name' or 'name'
        4. Sanitized source filename
        5. Hash of the data
        """
        # Try explicit ID fields
        if data.get("id"):
            return sanitize_id(str(data["id"]))
        if data.get("tool_id"):
            return sanitize_id(str(data["tool_id"]))

        # Try name fields
        if data.get("tool_name"):
            return sanitize_id(str(data["tool_name"]))
        if data.get("name"):
            return sanitize_id(str(data["name"]))

        # Try source filename
        if source_file:
            name = Path(source_file).stem
            try:
                return sanitize_id(name)
            except ValueError:
                pass

        # Last resort: hash the data
        data_str = json.dumps(data, sort_keys=True)
        return hashlib.md5(data_str.encode()).hexdigest()[:16]

    def _normalize_method(self, method: Optional[str]) -> str:
        """Normalize HTTP method to uppercase.

        Returns default_method if method is None or empty.
        """
        if not method or not str(method).strip():
            return self._default_method
        return str(method).upper().strip()
