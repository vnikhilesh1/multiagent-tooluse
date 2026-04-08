# Multi-Agent Tool-Use Conversation Generator
# Implementation Plan

---

## Executive Summary

This document outlines the implementation plan for building an offline synthetic data generation system that produces multi-turn conversations with multi-step/multi-tool traces, grounded in ToolBench schemas.

**Total Estimated Time: 28-35 hours**
**Budget Target: 15-20 hours (see Prioritization section)**

---

## LLM API Configuration

This project uses **Claude** via **Hyperspace AI Local LLM Proxy**.

| Setting | Value |
|---------|-------|
| Base URL | `http://localhost:6655/anthropic` |
| API Key | `31d1207b-312d-4faf-85b2-ca6d750ed60b` (or `ANTHROPIC_API_KEY` env var) |
| Default Model | `claude-sonnet-4-20250514` |
| SDK | `anthropic` Python package |

**Client initialization:**
```python
import anthropic

client = anthropic.Anthropic(
    base_url="http://localhost:6655/anthropic",
    api_key="31d1207b-312d-4faf-85b2-ca6d750ed60b"
)
```

**Note:** The Hyperspace proxy must be running on port 6655 before executing any LLM calls.

---

## Architecture Decisions Summary

| Component | Decision |
|-----------|----------|
| Data Model | Pydantic + LLM Inference + Preserve Original |
| Graph Database | Neo4j |
| Node Types | Tool + Endpoint (parameters as attributes) |
| Edge Types | Same-Domain + Semantic Similarity |
| Embeddings | Claude via Hyperspace proxy (text generation for similarity) |
| Sampling Algorithm | DFS with Backtracking |
| Constraints | Pydantic model (SamplingConstraints) |
| Patterns | Full (Sequential, Parallel, Branching, Iterative) |
| Orchestration | Centralized Controller |
| Communication | Direct Function Calls + Shared State |
| Agents | All 7 roles |
| Structured Output | Anthropic Tool Use (function calling) |
| Mock Execution | LLM-Generated responses |
| Context | Explicit object + Lazy ID generation |
| Value Extraction | LLM-based |
| Judge Dimensions | 4 (tool_correctness, argument_grounding, task_completion, naturalness) |
| Scoring Scale | 1-5 Likert |
| Repair Strategy | LLM "Fix This" prompt |
| Retry Limits | 3 attempts with escalation |
| Grounding | Full History + Explicit Available Values |
| Cross-conv Steering | Sampling Weights + Prompt Injection |
| Diversity Tracking | Simple Counters + Tool-Pair Tracking |
| Diversity Metrics | Tool Entropy + Unique Pair Ratio |
| LLM Model | Claude Sonnet via Hyperspace proxy |
| Caching | Deterministic Prompt Caching |
| CLI Framework | Typer |
| Configuration | YAML + CLI Override + Env Vars |
| Reproducibility | Single Seed + Config Serialization |
| Testing | Fake LLM (unit) + VCR (integration) + Real LLM (E2E) |

---

## Project Structure

```
toolbench-conversation-generator/
├── pyproject.toml
├── README.md
├── DESIGN.md
├── config.yaml
├── docker-compose.yml
├── .env.example
├── .gitignore
│
├── src/
│   ├── __init__.py
│   ├── cli.py                          # Typer CLI
│   ├── config.py                       # Configuration loading
│   │
│   ├── registry/                       # Tool Registry
│   │   ├── __init__.py
│   │   ├── models.py                   # Pydantic models
│   │   ├── loader.py                   # ToolBench parser
│   │   ├── inference.py                # LLM schema inference
│   │   └── validators.py               # Custom validators
│   │
│   ├── graph/                          # Neo4j Tool Graph
│   │   ├── __init__.py
│   │   ├── client.py                   # Neo4j connection
│   │   ├── schema.py                   # Constraints, indexes
│   │   ├── builder.py                  # Graph construction
│   │   ├── embeddings.py               # Embeddings/similarity
│   │   └── queries.py                  # Cypher queries
│   │
│   ├── sampling/                       # Tool Chain Sampler
│   │   ├── __init__.py
│   │   ├── constraints.py              # SamplingConstraints
│   │   ├── dfs_sampler.py              # DFS with backtracking
│   │   ├── patterns.py                 # Chain patterns
│   │   └── diversity_weights.py        # Inverse frequency
│   │
│   ├── agents/                         # Multi-Agent System
│   │   ├── __init__.py
│   │   ├── base.py                     # Base agent class
│   │   ├── context.py                  # ConversationContext
│   │   ├── orchestrator.py             # Centralized controller
│   │   ├── scenario_planner.py         # Scenario Planner
│   │   ├── user_simulator.py           # User Simulator
│   │   ├── assistant.py                # Assistant (function calling)
│   │   ├── tool_executor.py            # Tool Executor (LLM mocks)
│   │   ├── judge.py                    # Critic/Judge
│   │   ├── repair.py                   # Repair Agent
│   │   └── diversity_steering.py       # Diversity Steering
│   │
│   ├── extraction/                     # Value Extraction
│   │   ├── __init__.py
│   │   └── llm_extractor.py            # LLM-based extraction
│   │
│   ├── evaluation/                     # Evaluation Pipeline
│   │   ├── __init__.py
│   │   ├── validator.py                # Structural validation
│   │   ├── metrics.py                  # Diversity metrics
│   │   └── aggregator.py               # Score aggregation
│   │
│   ├── output/                         # Output Handling
│   │   ├── __init__.py
│   │   ├── serializer.py               # JSONL serialization
│   │   └── metadata.py                 # Metadata schema
│   │
│   └── llm/                            # LLM Client
│       ├── __init__.py
│       ├── client.py                   # Anthropic/Hyperspace wrapper
│       └── cache.py                    # Prompt caching
│
├── tests/
│   ├── __init__.py
│   ├── conftest.py                     # Fixtures, FakeLLM
│   ├── unit/
│   │   ├── test_registry.py
│   │   ├── test_sampling.py
│   │   ├── test_patterns.py
│   │   ├── test_context.py
│   │   └── test_validators.py
│   ├── integration/
│   │   ├── test_neo4j.py
│   │   ├── test_orchestrator.py
│   │   └── test_repair_loop.py         # Required by spec
│   └── e2e/
│       └── test_full_pipeline.py       # Required by spec
│
├── scripts/
│   ├── setup_neo4j.py
│   ├── download_toolbench.py
│   └── run_diversity_experiment.py
│
├── data/
│   ├── toolbench/                      # Raw ToolBench data
│   └── generated/                      # Output datasets
│
└── .cache/
    ├── embeddings/
    ├── inferences/
    └── llm_responses/
```

---

## Implementation Phases

---

### Phase 1: Project Foundation
**Time Estimate: 3-4 hours**

#### 1.1 Project Setup (0.5 hr)
- Create project directory structure
- Set up pyproject.toml with dependencies:
  - pydantic>=2.0
  - typer>=0.9
  - pyyaml>=6.0
  - neo4j>=5.0
  - anthropic>=0.30.0
  - python-dotenv>=1.0
  - tqdm>=4.0
  - joblib>=1.3
  - numpy>=1.24
  - pytest>=7.0 (dev)
  - vcrpy>=5.0 (dev)
- Create .gitignore (Python, .env, .cache/, data/)
- Create .env.example with required variables

#### 1.2 Configuration System (1.0 hr)
- Create Config Pydantic model with nested configs:
  - ModelsConfig (primary model name)
  - Neo4jConfig (uri, username, password)
  - SamplingConfig (min_steps, max_steps, semantic_threshold)
  - QualityConfig (min_score, max_retries)
  - GenerationConfig (default_count, parallel_workers)
  - CacheConfig (enabled, directory)
- Implement YAML loading with environment variable expansion
- Support ${VAR:-default} syntax
- CLI args override YAML values

#### 1.3 LLM Client Wrapper (1.0 hr)
- Create LLMClient class wrapping Anthropic SDK via Hyperspace proxy
- Configure client with:
  - base_url: "http://localhost:6655/anthropic"
  - api_key: from env or default "31d1207b-312d-4faf-85b2-ca6d750ed60b"
  - model: "claude-sonnet-4-20250514" (default)
- Implement methods:
  - `complete(prompt, system)` → str
  - `complete_json(prompt, system)` → dict
  - `complete_structured(prompt, response_model)` → Pydantic model
  - `complete_with_tools(messages, tools, tool_choice)` → dict with tool_calls
  - `chat(messages, system)` → str
- Handle API errors gracefully
- Support model override per call

#### 1.4 Prompt Caching System (0.5 hr)
- Create LLMCache class
- Hash-based caching (MD5 of prompt)
- Two-tier: memory dict + disk JSON files
- Methods: `get(prompt)`, `set(prompt, value)`
- Cache directory configurable

#### 1.5 CLI Skeleton (0.5 hr)
- Create Typer app with three commands:
  - `build` (stub)
  - `generate` (stub)
  - `evaluate` (stub)
- Add common options (--config, --verbose)
- Set up logging

**Deliverables:**
- [ ] Working project structure
- [ ] config.yaml with all settings
- [ ] LLM client with function calling support
- [ ] Prompt caching system
- [ ] CLI that accepts commands

---

### Phase 2: Tool Registry & Data Model
**Time Estimate: 4-5 hours**

#### 2.1 Pydantic Models (1.5 hr)
- **ParameterType** enum: string, integer, number, boolean, array, object, unknown
- **Parameter** model:
  - name: str
  - type: ParameterType (default: string)
  - description: str (default: "")
  - required: bool (default: False)
  - default: Optional[Any]
  - enum: Optional[List[Any]]
- **Endpoint** model:
  - id: str
  - tool_id: str
  - name: str
  - method: str (default: "GET")
  - path: str
  - description: str
  - parameters: List[Parameter]
  - response_schema: Optional[Dict]
  - domain: str (default: "unknown")
  - raw_schema: Optional[Dict] (excluded from serialization)
  - completeness_score: float (0-1)
  - inferred_fields: List[str]
- **Tool** model:
  - id: str
  - name: str
  - category: str
  - description: str
  - api_host: str
  - endpoints: List[Endpoint]
  - raw_schema: Optional[Dict]
  - completeness_score: float
- **ToolRegistry** model:
  - tools: Dict[str, Tool]
  - endpoints: Dict[str, Endpoint]
  - domains: List[str]
  - Methods: add_tool(), get_endpoint(), get_endpoints_by_domain()

#### 2.2 Custom Validators (0.5 hr)
- Type normalization validator (str→string, int→integer, etc.)
- Description truncation (max 500 chars)
- ID sanitization (replace spaces, special chars)
- Domain inference from category/path

#### 2.3 ToolBench Loader (1.5 hr)
- Handle multiple ToolBench JSON formats:
  - api_list vs endpoints vs apis
  - tool_name vs name
  - tool_description vs description
  - parameters as dict vs list
  - required_parameters + optional_parameters
- Extract tool_id from various sources (field, filename)
- Normalize method names (uppercase, default GET)
- Infer domain from category or path
- Graceful handling of malformed entries
- Progress bar with tqdm
- Optional limit parameter for testing

#### 2.4 LLM Schema Inference (1.0 hr)
- SchemaInferenceEngine class
- Identify tools needing inference (completeness < threshold)
- Prompt LLM to infer:
  - Missing descriptions
  - Parameter types marked as "unknown"
  - Response schemas
- Cache all inferences to disk
- Track which fields were inferred
- Configurable: --use-llm-inference flag

#### 2.5 Completeness Scoring (0.5 hr)
- Score based on:
  - Has description? (+0.2)
  - All parameters have types? (+0.3)
  - Has response schema? (+0.2)
  - Has meaningful name? (+0.1)
  - Has valid path? (+0.2)
- Apply to both Tool and Endpoint
- Use for filtering in sampling

**Deliverables:**
- [ ] Pydantic models with all fields
- [ ] Loader that handles any ToolBench format
- [ ] raw_schema preserved on each entity
- [ ] LLM inference with caching
- [ ] Completeness scores

---

### Phase 3: Neo4j Graph Setup
**Time Estimate: 4-5 hours**

#### 3.1 Neo4j Setup (1.0 hr)
- Create docker-compose.yml with Neo4j 5.18
- Configure plugins (APOC, GDS)
- Set memory limits
- Create Neo4jClient class:
  - Connection management (driver, session)
  - `verify_connection()` → bool
  - `query(cypher, **params)` → List[Dict]
  - `write(cypher, **params)` → None
  - `write_batch(cypher, batch, batch_size)` → None
  - Context manager for sessions
- Support both Docker and Aura (environment-based)
- Document both setup options in README

#### 3.2 Schema Creation (0.5 hr)
- Create constraints:
  - Tool.id unique
  - Endpoint.id unique
  - Domain.name unique
- Create indexes:
  - Endpoint.domain
  - Endpoint.name
  - Tool.category
- Handle "already exists" gracefully

#### 3.3 Graph Builder - Nodes (1.0 hr)
- ToolGraphBuilder class
- Batch create Tool nodes:
  - id, name, category, description (truncated), completeness
- Batch create Endpoint nodes:
  - id, tool_id, name, method, path, description, domain, completeness
- Create HAS_ENDPOINT edges (Tool → Endpoint)
- Use UNWIND for batch performance
- Progress reporting

#### 3.4 Domain Nodes and Edges (0.5 hr)
- Extract unique domains from endpoints
- Create Domain nodes
- Create IN_DOMAIN edges (Endpoint → Domain)
- Batch operations

#### 3.5 Text Embeddings via Claude (1.0 hr)
- EmbeddingGenerator class using Claude for similarity
- Generate text for comparison: "{name} {description} {domain}"
- Use Claude to generate semantic similarity scores between endpoint pairs
- Alternative: use sentence-transformers locally for embeddings
- Cache embeddings/similarity scores to JSON file
- Only compute for endpoints not already cached
- Return Dict[endpoint_id, List[float]] or similarity matrix

#### 3.6 Same-Domain Edges (0.5 hr)
- Single Cypher query:
  ```
  MATCH (e1:Endpoint)-[:IN_DOMAIN]->(d)<-[:IN_DOMAIN]-(e2:Endpoint)
  WHERE e1.id < e2.id
  MERGE (e1)-[:SAME_DOMAIN]->(e2)
  ```
- Run after all endpoints created

#### 3.7 Semantic Similarity Edges (1.0 hr)
- Compute pairwise cosine similarity
- Filter by threshold (default 0.7)
- Create SEMANTICALLY_SIMILAR edges with score attribute
- Batch insert edges
- Progress bar (this is slow for 16k endpoints)
- Consider: only compute for endpoints without SAME_DOMAIN edge

**Deliverables:**
- [ ] Neo4j running (Docker or Aura)
- [ ] Client wrapper with all methods
- [ ] Graph with Tool, Endpoint, Domain nodes
- [ ] HAS_ENDPOINT, IN_DOMAIN edges
- [ ] SAME_DOMAIN edges
- [ ] SEMANTICALLY_SIMILAR edges with scores
- [ ] Cached embeddings

---

### Phase 4: Tool Chain Sampler
**Time Estimate: 3-4 hours**

#### 4.1 Sampling Constraints Model (0.5 hr)
- SamplingConstraints Pydantic model:
  - min_steps: int = 2
  - max_steps: int = 5
  - required_domains: Optional[List[str]]
  - excluded_domains: Optional[List[str]]
  - required_tools: Optional[List[str]]
  - excluded_tools: Optional[List[str]]
  - pattern: Literal["sequential", "parallel", "branching", "iterative"]
  - min_completeness: float = 0.5
  - require_multi_tool: bool = False
- `validate_chain(chain, endpoints)` method

#### 4.2 DFS Sampler (1.5 hr)
- DFSSampler class with Neo4j client
- DFSState dataclass: current, path, visited, depth
- `sample(constraints)` → Optional[ChainPattern]
- Algorithm:
  1. Get start candidates from Neo4j (filtered by constraints)
  2. Apply diversity weights to prioritize underrepresented tools
  3. Pick random start from top candidates
  4. DFS traversal:
     - Push to stack: (node, path, visited, depth)
     - Pop and check if target depth reached
     - If valid chain, return it
     - Otherwise get neighbors and push to stack
  5. Backtrack when stuck
  6. Try up to 10 different starting points
- `_get_neighbors(node, visited, constraints)`:
  - Cypher query for SAME_DOMAIN and SEMANTICALLY_SIMILAR edges
  - Filter by completeness and excluded domains
  - Apply diversity weights
  - Return top 10 candidates
- `_is_valid_chain(chain, constraints)`:
  - Check multi-tool requirement
  - Check required domains
  - Check length bounds

#### 4.3 Pattern Classes (1.0 hr)
- ChainPattern abstract base class:
  - `to_execution_plan()` → List[List[ToolStep]]
  - `get_endpoints()` → List[str]
  - `pattern_type` property
- ToolStep dataclass: endpoint_id, depends_on, is_parallel
- **SequentialChain**: A → B → C
  - Steps in order, each depends on previous
- **ParallelChain**: [A, B] → C
  - First N steps parallel, rest sequential
  - parallel_steps + then_steps
- **BranchingChain**: A → [B if cond else C] → D
  - start, branches dict, merge
  - `select_branch(condition)` method
- **IterativeChain**: A → B → B → B → C
  - start, loop_step, loop_count, end

#### 4.4 Pattern-Aware Sampling (0.5 hr)
- `_build_pattern(chain, pattern_type)` method
- For parallel: split first 2 steps as parallel
- For branching: first step, middle as branch, last as merge
- For iterative: first, repeated middle, last
- Fallback to sequential if chain too short

#### 4.5 Diversity Weight Integration (0.5 hr)
- Accept diversity_weights dict from DiversitySteeringAgent
- Apply to start candidate selection
- Apply to neighbor prioritization
- Higher weight = more likely to be selected

**Deliverables:**
- [ ] SamplingConstraints model with validation
- [ ] DFS sampler with backtracking
- [ ] All four pattern classes
- [ ] Pattern-aware chain building
- [ ] Diversity weight support

---

### Phase 5: Conversation Context & Base Agent
**Time Estimate: 2 hours**

#### 5.1 ConversationContext Dataclass (0.5 hr)
- Fields:
  - conversation_id: str (auto-generated)
  - messages: List[Message]
  - tool_outputs: List[ToolOutput]
  - generated_ids: Dict[str, str]
  - grounding_values: Dict[str, Any]
  - tool_chain: List[str]
  - current_step: int
  - target_steps: int
  - scenario_description: str
  - seed: Optional[int]
  - start_time: datetime

#### 5.2 Message Model (0.25 hr)
- role: str (user, assistant, tool)
- content: Optional[str]
- tool_calls: Optional[List[Dict]]
- tool_call_id: Optional[str]
- timestamp: datetime

#### 5.3 ToolOutput Model (0.25 hr)
- endpoint: str
- arguments: Dict[str, Any]
- result: Dict[str, Any]
- call_id: str

#### 5.4 Helper Methods (0.5 hr)
- `add_message(message)`: append to messages
- `add_tool_output(output)`: append and increment step
- `get_history_for_prompt()`: format as "User: ...\nAssistant: ..."
- `get_available_values()`: format grounding values for injection
- `is_complete` property: current_step >= target_steps
- `to_conversation()`: convert to final output dict

#### 5.5 Base Agent Class (0.25 hr)
- Abstract class with:
  - llm: LLMClient
  - name: str
  - `generate(context, **kwargs)` abstract method

#### 5.6 Lazy ID Generation (0.25 hr)
- `generate_id(entity_type)` method on context
- Check if entity_type in generated_ids
- If not, create: f"{entity_type}_{uuid4().hex[:8]}"
- Return existing or new ID

**Deliverables:**
- [ ] ConversationContext with all fields
- [ ] Message and ToolOutput models
- [ ] All helper methods
- [ ] BaseAgent abstract class
- [ ] Lazy ID generation

---

### Phase 6: Agent Implementation
**Time Estimate: 8-10 hours**

#### 6.1 Scenario Planner Agent (1.0 hr)
- Input: context, tool_chain (ChainPattern)
- Output: Scenario model
- Scenario model:
  - description: str
  - user_goal: str
  - expected_flow: List[str]
  - disambiguation_points: List[int]
  - available_tools: List[Dict] (Anthropic tool format)
- Prompt LLM with tool chain details
- Ask for realistic scenario that would use these tools
- Convert endpoints to Anthropic tool format (tool_use)

#### 6.2 User Simulator Agent (1.0 hr)
- Input: context, scenario
- Output: Message (role=user)
- First message:
  - Based on scenario.user_goal
  - May be vague (requiring clarification)
  - Natural, conversational tone
- Follow-up messages:
  - Respond to assistant's last message
  - Provide clarification if at disambiguation point
  - Continue toward goal

#### 6.3 Assistant Agent with Function Calling (1.5 hr)
- Input: context, scenario
- Output: Message (role=assistant, possibly with tool_calls)
- Decision logic:
  - If at disambiguation point → ask clarifying question
  - If missing required info → ask for it
  - If ready → use appropriate tool
  - If all tools used → provide final summary
- Build messages for Claude:
  - System prompt with user goal and grounding values
  - Conversation history
  - Available tools (not yet used)
- Use `complete_with_tools()` from LLM client (Anthropic tool_use format)
- Parse tool_calls from response

#### 6.4 Tool Executor Agent - LLM Mocks (1.5 hr)
- Input: context, tool_call
- Output: ToolOutput
- Build prompt with:
  - Endpoint name, description, method
  - Arguments provided
  - Grounding values from context
  - Request for realistic mock response
- Requirements in prompt:
  - Match expected API structure
  - Generate realistic IDs (prefixed)
  - Reference prior IDs if applicable
  - Include relevant fields (status, names, prices)
- Fallback to basic mock if LLM fails:
  - {status: "success", id: generated_id, data: arguments}

#### 6.5 LLM Value Extraction (0.5 hr)
- Called after each tool execution
- Input: tool output JSON
- Output: Dict of referenceable values
- Prompt LLM to identify:
  - IDs (anything with "id" in name)
  - Reference numbers
  - Names that might be needed
  - Status values
- Update context.grounding_values
- Fallback: regex-based extraction for *_id fields

#### 6.6 Judge Agent (1.0 hr)
- Input: completed context
- Output: JudgeScores model
- JudgeScores:
  - tool_correctness: int (1-5)
  - argument_grounding: int (1-5)
  - task_completion: int (1-5)
  - naturalness: int (1-5)
  - reasoning: str
  - `average` property
- Single prompt with:
  - Full conversation formatted
  - Scoring criteria for each dimension
  - Request for JSON output
- Use `complete_structured()` with JudgeScores model

#### 6.7 Repair Agent (1.0 hr)
- Input: context, scores
- Output: repaired context
- "Fix This" approach:
  - Serialize conversation to JSON
  - Include scores and feedback
  - Ask LLM to fix issues
  - Parse fixed conversation
- Focus areas based on low scores:
  - Low argument_grounding → fix hallucinated references
  - Low tool_correctness → fix tool selection
  - Low task_completion → complete the task
  - Low naturalness → improve dialogue
- Return original if repair fails

#### 6.8 Diversity Steering Agent (0.5 hr)
- DiversityTracker dataclass:
  - tool_counts: Counter
  - domain_counts: Counter
  - tool_pair_counts: Counter
  - pattern_hashes: Set[str]
- Methods:
  - `suggest_constraints()` → SamplingConstraints
    - Find underrepresented domain
    - Set as required_domain
  - `get_diversity_weights()` → Dict[str, float]
    - Inverse frequency: max_count / (count + 1)
  - `record(context)`:
    - Update all counters
    - Add pattern hash
  - `compute_metrics()` → Dict:
    - tool_entropy
    - unique_pair_ratio
    - unique_tools
    - unique_patterns

#### 6.9 Centralized Orchestrator (1.5 hr)
- ConversationOrchestrator class
- Initialize all 7 agents
- Initialize DFS sampler
- `generate_dataset(count, seed, cross_conversation_steering)`:
  - Loop count times
  - Set seed for each conversation
  - Call `generate_single()`
  - Collect results
- `generate_single(seed, use_steering)`:
  - Retry loop (max 3 attempts):
    1. Get constraints (with/without steering)
    2. Sample tool chain
    3. Generate scenario
    4. Generate conversation turns
    5. Judge quality
    6. Repair if needed
    7. Record and return if passing
  - Escalation on retry:
    a. Regenerate with feedback
    b. Resample tool chain
    c. Discard and log
- `_generate_conversation(context, scenario)`:
  - While not complete:
    - User simulator generates message
    - Assistant generates response
    - If tool calls: execute each, extract values
  - Return completed context

**Deliverables:**
- [ ] All 7 agents implemented
- [ ] Scenario planning with Anthropic tool format
- [ ] User simulation with disambiguation
- [ ] Assistant with function calling
- [ ] LLM-generated mock responses
- [ ] LLM value extraction
- [ ] 4-dimension judge scoring
- [ ] LLM-based repair
- [ ] Diversity tracking and steering
- [ ] Orchestrator with retry/repair loop

---

### Phase 7: Evaluation & Metrics
**Time Estimate: 2-3 hours**

#### 7.1 Structural Validator (0.5 hr)
- `validate_structure(conversation)` → (bool, List[str])
- Check:
  - Has conversation_id
  - Has messages list
  - Messages have required fields (role, content or tool_calls)
  - Tool calls have endpoint and arguments
  - Tool outputs are valid JSON
- Return success and list of errors

#### 7.2 Tool Entropy Metric (0.5 hr)
- `compute_entropy(tool_counts)` → float
- Formula: H = -Σ(p_i * log(p_i))
- Higher = more even distribution
- Handle edge cases (single tool, empty counts)

#### 7.3 Unique Tool-Pair Ratio (0.5 hr)
- `compute_pair_ratio(pair_counts, total_tools)` → float
- Formula: |unique_pairs| / |possible_pairs|
- possible_pairs = n * (n-1) / 2
- Higher = more diverse combinations

#### 7.4 Aggregation Functions (0.5 hr)
- `aggregate_scores(results)` → Dict:
  - mean_tool_correctness
  - mean_argument_grounding
  - mean_task_completion
  - mean_naturalness
  - mean_overall
  - pass_rate (% above threshold)
  - repair_rate (% needing repair)
  - multi_step_rate (% with ≥3 tool calls)
  - multi_tool_rate (% with ≥2 distinct tools)

#### 7.5 JSONL Serializer (0.5 hr)
- `serialize_conversation(result)` → str (JSON line)
- Include:
  - conversation_id
  - messages (role, content, tool_calls)
  - judge_scores (all 4 + reasoning)
  - metadata:
    - seed
    - tools_used
    - num_turns
    - pattern_type
    - generated_at
    - config snapshot
- `write_dataset(results, path)`:
  - Write JSONL file
  - One conversation per line

**Deliverables:**
- [ ] Structural validator
- [ ] Entropy metric
- [ ] Pair ratio metric
- [ ] Score aggregation
- [ ] JSONL serializer with metadata

---

### Phase 8: CLI Implementation
**Time Estimate: 2 hours**

#### 8.1 Build Command (0.5 hr)
```
toolgen build --toolbench-path PATH [--use-llm-inference] [--limit N]
```
- Load ToolBench data
- Optionally run LLM inference
- Build tool registry
- Set up Neo4j schema
- Create graph (nodes, edges)
- Generate embeddings
- Create semantic edges
- Print statistics

#### 8.2 Generate Command (1.0 hr)
```
toolgen generate --output PATH --count N --seed S [--no-cross-conversation-steering]
```
- Load registry from Neo4j
- Initialize orchestrator
- Generate conversations
- Write JSONL output
- Print summary statistics
- Support --no-cross-conversation-steering flag

#### 8.3 Evaluate Command (0.5 hr)
```
toolgen evaluate --input PATH
```
- Load JSONL file
- Compute aggregate statistics
- Compute diversity metrics
- Print formatted report:
  - Quality scores (mean, std)
  - Pass/fail rates
  - Multi-step/multi-tool rates
  - Diversity metrics

**Deliverables:**
- [ ] `build` command working end-to-end
- [ ] `generate` command with steering toggle
- [ ] `evaluate` command with full report

---

### Phase 9: Testing
**Time Estimate: 3-4 hours**

#### 9.1 FakeLLM Fixture (0.5 hr)
- FakeLLM class implementing LLMClient interface
- Pattern-based response mapping
- Default responses for each method
- Configurable per-test overrides
- Pytest fixture in conftest.py

#### 9.2 Unit Tests: Registry (0.5 hr)
- Test Pydantic model validation
- Test graceful defaults
- Test loader with sample JSON
- Test completeness scoring

#### 9.3 Unit Tests: Sampling (0.25 hr)
- Test SamplingConstraints validation
- Test constraint checking on chains

#### 9.4 Unit Tests: Patterns (0.25 hr)
- Test each pattern class
- Test to_execution_plan()
- Test get_endpoints()

#### 9.5 Unit Tests: Context (0.25 hr)
- Test message adding
- Test grounding value formatting
- Test ID generation
- Test completion detection

#### 9.6 Unit Tests: Validator (0.25 hr)
- Test structural validation
- Test error messages

#### 9.7 Integration Test: Retry/Repair Loop (1.0 hr)
- **Required by spec**
- Test orchestrator with FakeLLM
- Simulate low-quality first attempt
- Verify repair is triggered
- Verify re-evaluation
- Test all three retry escalations

#### 9.8 E2E Test: Full Pipeline (1.0 hr)
- **Required by spec**
- Use subset of 50-100 real tools
- Generate 100 conversations (real LLM)
- Assert:
  - Mean judge score > 3.5
  - 50% have ≥3 tool calls
  - 50% have ≥2 distinct tools
  - No structural validation failures
- Mark as slow, skip in CI
- Target: <10 minutes runtime

**Deliverables:**
- [ ] FakeLLM fixture
- [ ] Unit tests for all modules
- [ ] Integration test for retry/repair (required)
- [ ] E2E test for 100 samples (required)

---

### Phase 10: Documentation
**Time Estimate: 2-3 hours**

#### 10.1 README.md (0.5 hr)
- Project overview
- Installation instructions
- Neo4j setup (Docker and Aura)
- Configuration
- CLI usage examples
- Quick start guide

#### 10.2 DESIGN.md: Architecture & Decisions (0.5 hr)
- System overview diagram
- Component descriptions
- Agent roles and responsibilities
- Communication protocol (direct calls + shared state)
- Key design decisions with rationale

#### 10.3 DESIGN.md: Context Management (0.5 hr)
- Within-conversation grounding:
  - Explicit available values injection
  - LLM extraction approach
  - Lazy ID generation
- Cross-conversation steering:
  - Diversity tracking
  - Sampling weight adjustment
  - Prompt injection
- Tradeoffs and limitations
- What would change at scale

#### 10.4 DESIGN.md: Prompt Design (0.5 hr)
- Key prompts documented:
  - Scenario planning prompt
  - User simulation prompt
  - Tool mock generation prompt
  - Judge scoring prompt
  - Repair prompt
- Rationale for each structure
- **At least one failed iteration and lessons learned**

#### 10.5 DESIGN.md: Diversity & Quality Analysis (0.5 hr)
- Metrics chosen:
  - Tool usage entropy (justification)
  - Unique tool-pair ratio (justification)
- Quality metrics (judge scores)
- Placeholder for experiment results

#### 10.6 Run Diversity Experiment (0.5 hr)
- Run A: --no-cross-conversation-steering
- Run B: with steering enabled
- Same seed for both
- Record:
  - Diversity metrics (A vs B)
  - Quality metrics (A vs B)
- Add results to DESIGN.md
- Analyze diversity-quality tradeoff

**Deliverables:**
- [ ] README.md with full instructions
- [ ] DESIGN.md with all required sections
- [ ] Diversity experiment results
- [ ] Analysis of tradeoffs

---

## Time Summary

| Phase | Description | Min Hours | Max Hours |
|-------|-------------|-----------|-----------|
| 1 | Project Foundation | 3 | 4 |
| 2 | Tool Registry | 4 | 5 |
| 3 | Neo4j Graph | 4 | 5 |
| 4 | Sampler | 3 | 4 |
| 5 | Context & Base | 2 | 2 |
| 6 | Agents | 8 | 10 |
| 7 | Evaluation | 2 | 3 |
| 8 | CLI | 2 | 2 |
| 9 | Testing | 3 | 4 |
| 10 | Documentation | 2 | 3 |
| **TOTAL** | | **33** | **42** |

---

## Prioritization: Fit in 15-20 Hours

### P0: Must Have (Core) — 15-18 hours

| Component | Time | Justification |
|-----------|------|---------------|
| Project setup + config + LLM client | 2 hr | Foundation |
| Pydantic models + basic loader | 2 hr | Data model |
| Neo4j setup + graph (skip semantic edges) | 2.5 hr | Graph requirement |
| Basic DFS sampler (sequential only) | 1.5 hr | Sampler requirement |
| Context + 4 core agents | 5 hr | Core functionality |
| Basic orchestrator (no repair) | 1 hr | Generation loop |
| CLI (build, generate) | 1 hr | Required commands |
| E2E test + README | 1.5 hr | Required deliverables |
| DESIGN.md (document gaps) | 1.5 hr | Required documentation |

### P1: Should Have (Quality) — add 5-8 hours

| Component | Time | Justification |
|-----------|------|---------------|
| Semantic similarity edges | 1 hr | Better sampling |
| Repair agent + retry loop | 1.5 hr | Required integration test |
| Diversity steering | 1 hr | Required experiment |
| Full patterns | 1.5 hr | Expressiveness |
| Diversity experiment | 1 hr | Required in DESIGN.md |
| Integration tests | 1 hr | Quality assurance |

### P2: Nice to Have (Polish) — add 5-7 hours

| Component | Time | Justification |
|-----------|------|---------------|
| LLM schema inference | 1.5 hr | Better data quality |
| Scenario Planner agent | 1 hr | More realistic scenarios |
| LLM value extraction | 1 hr | Better grounding |
| Prompt caching | 0.5 hr | Cost savings |
| evaluate CLI command | 0.5 hr | Convenience |
| Unit tests | 1 hr | Code quality |
| DESIGN.md polish | 1 hr | Documentation quality |

---

## Recommended Implementation Order

### Day 1 (4-5 hours)
- Phase 1: Project foundation
- Phase 2.1-2.3: Models and loader

### Day 2 (4-5 hours)
- Phase 2.4-2.5: LLM inference and scoring (optional)
- Phase 3: Neo4j graph setup

### Day 3 (4-5 hours)
- Phase 4: DFS sampler and patterns
- Phase 5: Context and base agent

### Day 4 (4-5 hours)
- Phase 6.1-6.4: Core agents (Planner, User, Assistant, Executor)

### Day 5 (4-5 hours)
- Phase 6.5-6.9: Remaining agents and orchestrator

### Day 6 (4-5 hours)
- Phase 7: Evaluation and metrics
- Phase 8: CLI commands

### Day 7 (4-5 hours)
- Phase 9: Testing
- Phase 10: Documentation

---

## Risk Mitigation

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| Neo4j setup fails | High | Medium | Document Aura as fallback |
| LLM costs exceed budget | Medium | Medium | Aggressive caching; limit inference |
| DFS too slow | Medium | Low | Add timeout; reduce depth; cache |
| Full patterns too complex | High | High | Ship sequential only; document rest |
| Judge scores inconsistent | Medium | Medium | Add reasoning; few-shot examples |
| Repair creates new errors | Medium | Medium | Validate output; limit attempts |
| Time overrun | High | High | Prioritize P0; document what's missing |

---

## Cost Estimate

| Component | Cost | Notes |
|-----------|------|-------|
| Build: LLM inference (1000 tools) | ~$10 | One-time, cached |
| Build: Embeddings (16k tools) | ~$2 | One-time, cached |
| Generate: All agents (100 convs) | ~$15 | Claude Sonnet via Hyperspace |
| **TOTAL** | **~$27** | For build + 100 conversations |

---

## Definition of Done

### Minimum Viable Submission
- [ ] `build` command creates Neo4j graph from ToolBench
- [ ] `generate` command produces JSONL with conversations
- [ ] Conversations include multi-step tool calls
- [ ] LLM-as-judge scores included in output
- [ ] E2E test passes (100 samples, score > 3.5)
- [ ] README.md with setup instructions
- [ ] DESIGN.md with architecture and decisions

### Full Submission
- [ ] All above plus:
- [ ] `evaluate` command with metrics
- [ ] Retry/repair loop working
- [ ] Diversity steering with toggle
- [ ] Diversity experiment results in DESIGN.md
- [ ] All four pattern types
- [ ] Integration test for repair loop
- [ ] Unit tests for major modules

---

## Notes

- DESIGN.md carries 25% weight — invest time in clear documentation
- Working-but-incomplete with strong DESIGN.md > complete-but-shallow
- Document what you would do next if time runs out
- Honest reasoning > polished claims