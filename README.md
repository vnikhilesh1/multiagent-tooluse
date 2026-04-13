# toolgen

> Generate synthetic multi-agent tool-use conversations for LLM training

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)]()

toolgen is a CLI tool that creates high-quality synthetic conversation data featuring multi-turn, multi-tool interactions between users and AI agents. It uses a knowledge graph to model tool relationships and employs multiple specialized agents to generate realistic, diverse training data.

## Overview

### Key Features

- **Multi-agent generation pipeline** — Specialized agents for scenario planning, user simulation, tool mocking, quality judging, and repair
- **Knowledge graph-based tool selection** — NetworkX graph models tool relationships, enabling intelligent tool combination sampling
- **Cross-conversation diversity steering** — Tracks tool usage across generations to ensure diverse coverage
- **Within-conversation grounding** — Maintains consistent entity values (IDs, names) across tool calls
- **Automated quality evaluation** — Judge agent scores conversations on naturalness, coherence, and correctness
- **Self-repair capability** — Automatically fixes conversations that fail validation

### Use Cases

- Generate training data for tool-use LLMs
- Create benchmarks for evaluating agent capabilities
- Produce synthetic datasets for fine-tuning
- Test tool integration scenarios

## Installation

### From Source

```bash
git clone https://github.com/your-org/toolgen.git
cd toolgen
pip install -e .
```

### With Development Dependencies

```bash
pip install -e ".[dev]"
```

### Requirements

- Python 3.9+
- Key dependencies:
  - `networkx` — Knowledge graph management
  - `anthropic` — Claude API client
  - `pyyaml` — Configuration parsing
  - `click` — CLI framework
  - `jsonlines` — JSONL file handling

## Graph Setup

toolgen uses a NetworkX knowledge graph to model relationships between tools, entities, and domains.

### Graph Structure

```
Nodes:
  - Tools: API endpoints with input/output schemas
  - Entities: Data types (user, order, product, etc.)
  - Domains: Categories (e-commerce, travel, finance)

Edges:
  - requires: Tool A needs output from Tool B
  - produces: Tool produces entity type
  - related_to: Tools commonly used together
  - belongs_to: Tool belongs to domain
```

### Building the Graph

```bash
# Build from tool definition files
toolgen build --tools ./tools/ --output ./data/graph.json

# Build with custom schema
toolgen build --tools ./tools/ --schema ./schema.yaml
```

### Persistence Options

**In-memory (default):**
```python
from toolgen.graph import ToolGraph
graph = ToolGraph()  # Ephemeral, lost on exit
```

**JSON export/import:**
```bash
# Export
toolgen build --tools ./tools/ --output ./data/graph.json

# Import
toolgen generate --graph ./data/graph.json --output data.jsonl
```

**Pickle serialization:**
```python
import pickle
from toolgen.graph import ToolGraph

graph = ToolGraph.from_tools("./tools/")
with open("graph.pkl", "wb") as f:
    pickle.dump(graph, f)
```

## Configuration

### config.yaml

```yaml
llm:
  provider: "anthropic"        # or "openai"
  model: "claude-3-sonnet"     # Model to use
  temperature: 0.7             # Generation temperature
  max_tokens: 4096             # Max tokens per response

generation:
  num_conversations: 100       # Number to generate
  min_turns: 3                 # Minimum conversation turns
  max_turns: 10                # Maximum conversation turns
  tools_per_conversation: 2-5  # Tool count range

diversity:
  steering_enabled: true       # Cross-conversation steering
  smoothing: 1.0               # Laplace smoothing for weights
  domain_balance: true         # Balance across domains

validation:
  min_quality_score: 0.7       # Minimum overall score
  max_repair_attempts: 3       # Repair attempts before discard

graph:
  persistence: "json"          # "memory", "json", or "pickle"
  path: "./data/graph.json"    # Path for persistence

output:
  format: "jsonl"              # Output format
  path: "./output/conversations.jsonl"
```

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `ANTHROPIC_API_KEY` | API key for Claude | Required |
| `OPENAI_API_KEY` | API key for OpenAI (if using) | Optional |
| `TOOLGEN_CONFIG` | Path to config file | `./config.yaml` |
| `TOOLGEN_LOG_LEVEL` | Logging verbosity | `INFO` |
| `TOOLGEN_CACHE_DIR` | Cache directory | `~/.cache/toolgen` |

## CLI Usage

### Build Command

Build the knowledge graph from tool definitions:

```bash
# Basic build
toolgen build --tools ./tools/ --output ./data/graph.json

# With custom schema validation
toolgen build --tools ./tools/ --schema ./schema.yaml --output ./data/graph.json

# Verbose output
toolgen build --tools ./tools/ --output ./data/graph.json --verbose
```

### Generate Command

Generate synthetic conversations:

```bash
# Using config file
toolgen generate --config config.yaml --output conversations.jsonl

# With CLI overrides
toolgen generate --config config.yaml \
    --num 100 \
    --min-turns 3 \
    --max-turns 8 \
    --output data.jsonl

# Dry run (no API calls, shows what would be generated)
toolgen generate --dry-run --num 10

# Disable diversity steering
toolgen generate --config config.yaml \
    --no-cross-conversation-steering \
    --output baseline.jsonl

# Set random seed for reproducibility
toolgen generate --config config.yaml --seed 42 --output data.jsonl
```

### Evaluate Command

Evaluate generated conversations:

```bash
# Basic evaluation
toolgen evaluate --input conversations.jsonl

# Save report to file
toolgen evaluate --input conversations.jsonl --output report.txt

# JSON output for programmatic use
toolgen evaluate --input conversations.jsonl --format json
```

## Quick Start

Get started in 5 minutes:

```bash
# 1. Clone and install
git clone https://github.com/your-org/toolgen.git
cd toolgen
pip install -e .

# 2. Set up API key
export ANTHROPIC_API_KEY="your-key-here"

# 3. Create minimal config
cat > config.yaml << 'EOF'
llm:
  provider: anthropic
  model: claude-3-sonnet
  temperature: 0.7

generation:
  num_conversations: 10
  min_turns: 3
  max_turns: 6

output:
  format: jsonl
  path: ./output/conversations.jsonl
EOF

# 4. Build the graph (using included sample tools)
toolgen build --tools ./sample_tools/ --output ./data/graph.json

# 5. Generate conversations
toolgen generate --config config.yaml --output my_data.jsonl

# 6. Evaluate the results
toolgen evaluate --input my_data.jsonl
```

## Output Format

Generated conversations are saved as JSONL, one conversation per line:

```json
{
  "id": "conv_a1b2c3d4",
  "messages": [
    {"role": "user", "content": "I need to check my recent orders"},
    {"role": "assistant", "content": "I'll look that up for you.", "tool_calls": [
      {"id": "call_1", "name": "get_user", "arguments": {"email": "user@example.com"}}
    ]},
    {"role": "tool", "tool_call_id": "call_1", "content": "{\"user_id\": \"usr_123\", \"name\": \"John\"}"},
    {"role": "assistant", "content": "Found your account. Let me get your orders.", "tool_calls": [
      {"id": "call_2", "name": "get_orders", "arguments": {"user_id": "usr_123"}}
    ]},
    {"role": "tool", "tool_call_id": "call_2", "content": "{\"orders\": [{\"id\": \"ord_456\", \"status\": \"shipped\"}]}"},
    {"role": "assistant", "content": "You have one recent order (ord_456) which has been shipped."}
  ],
  "tools": ["get_user", "get_orders"],
  "domain": "e-commerce",
  "quality_score": 0.87,
  "validation": {
    "passed": true,
    "scores": {
      "naturalness": 0.85,
      "tool_appropriateness": 0.90,
      "coherence": 0.88,
      "completeness": 0.85
    }
  },
  "metadata": {
    "scenario": "User checking order status",
    "generated_at": "2024-01-15T10:30:00Z",
    "seed": 42
  }
}
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests (`pytest`)
5. Commit (`git commit -m 'Add amazing feature'`)
6. Push (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## License

MIT License — see [LICENSE](LICENSE) for details.
