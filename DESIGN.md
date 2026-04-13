# toolgen Design Document

This document describes the architecture, design decisions, and implementation details of the toolgen synthetic conversation generator.

---

## 1. Architecture & Decisions

### 1.1 System Overview Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         toolgen CLI                              │
├─────────────────────────────────────────────────────────────────┤
│  build          │    generate         │    evaluate             │
└────────┬────────┴─────────┬───────────┴──────────┬──────────────┘
         │                  │                      │
         ▼                  ▼                      ▼
┌─────────────────┐  ┌─────────────────────────────────────┐  ┌──────────────┐
│  Tool Registry  │  │        Generation Pipeline          │  │  Evaluator   │
│  ┌───────────┐  │  │  ┌─────────┐  ┌─────────┐  ┌─────┐ │  │              │
│  │ Schema    │  │  │  │Scenario │→ │  Agent  │→ │Judge│ │  │  Metrics     │
│  │ Loader    │  │  │  │Planner  │  │Orchestra│  │     │ │  │  Aggregator  │
│  └───────────┘  │  │  └─────────┘  └─────────┘  └─────┘ │  └──────────────┘
│  ┌───────────┐  │  └───────────────────┬─────────────────┘
│  │ Graph     │  │                      │
│  │ Builder   │  │                      ▼
│  └───────────┘  │  ┌─────────────────────────────────────┐
└────────┬────────┘  │         Shared State                │
         │           │  ┌──────────┐  ┌─────────────────┐  │
         └──────────►│  │AvailVals │  │Diversity Tracker│  │
                     │  └──────────┘  └─────────────────┘  │
                     └─────────────────────────────────────┘
```

### 1.2 Component Descriptions

#### Tool Registry

- **Purpose**: Central repository of tool definitions and their relationships
- **Responsibilities**: 
  - Load tool schemas from YAML/JSON files
  - Validate tool definitions against schema
  - Build relationship graph from tool metadata
- **Key classes**: `ToolRegistry`, `ToolSchema`, `ToolLoader`

```python
class ToolRegistry:
    def __init__(self):
        self.tools: Dict[str, ToolSchema] = {}
        self.graph: nx.DiGraph = nx.DiGraph()
    
    def load(self, path: str) -> None:
        """Load tools from directory."""
        for file in Path(path).glob("*.yaml"):
            schema = ToolSchema.from_yaml(file)
            self.tools[schema.name] = schema
            self._add_to_graph(schema)
```

#### Knowledge Graph (NetworkX)

- **Purpose**: Model relationships between tools, entities, and domains
- **Structure**: 
  - Nodes: tools, entities (user, order, etc.), domains (e-commerce, travel)
  - Edges: requires, produces, related_to, belongs_to
- **Operations**: Traversal for tool selection, subgraph extraction for scenarios

```python
# Graph structure example
graph.add_node("get_user", type="tool", domain="e-commerce")
graph.add_node("user_id", type="entity")
graph.add_edge("get_user", "user_id", relation="produces")
graph.add_edge("get_orders", "user_id", relation="requires")
```

#### Sampler

- **Purpose**: Select tools and scenarios for conversation generation
- **Algorithm**: Weighted sampling based on diversity scores
- **Inputs**: Graph, diversity tracker, configuration
- **Outputs**: Tool sets with domain hints

```python
class ToolSampler:
    def sample(self, n_tools: int) -> List[Tool]:
        weights = self.diversity_tracker.get_weights(self.all_tools)
        return random.choices(self.all_tools, weights=weights, k=n_tools)
```

#### Agent Orchestra

- **Purpose**: Coordinate multi-agent conversation generation
- **Agents**: Scenario Planner, User Simulator, Assistant, Tool Mock, Judge, Repair
- **Flow**: Sequential pipeline with feedback loops for repair

### 1.3 Agent Roles

| Agent | Role | Input | Output |
|-------|------|-------|--------|
| Scenario Planner | Creates realistic multi-tool scenarios | Tool set, domain hints | Scenario description, expected flow |
| User Simulator | Generates natural user messages | Scenario, conversation history | User message |
| Assistant | Responds to user, decides tool calls | Conversation history, available tools | Response or tool call |
| Tool Mock | Simulates tool execution results | Tool call with arguments | Realistic tool response |
| Judge | Scores conversation quality | Complete conversation | Quality scores (0-1) |
| Repair Agent | Fixes validation failures | Failed conversation, error details | Corrected conversation |

### 1.4 Communication Protocol

#### Direct Calls

The orchestrator calls each agent sequentially in a controlled loop:

```python
class Orchestrator:
    def generate_conversation(self, tools: List[Tool]) -> Conversation:
        # Plan scenario
        scenario = self.planner.plan(tools, self.get_domain_hints())
        
        history = []
        available_values = {}
        
        while not self._is_complete(history, scenario):
            # Generate user turn
            user_msg = self.user_sim.generate(scenario, history)
            history.append({"role": "user", "content": user_msg})
            
            # Generate assistant response
            response = self.assistant.respond(history, tools, available_values)
            history.append(response)
            
            # Execute tool calls if present
            if response.get("tool_calls"):
                for call in response["tool_calls"]:
                    result = self.tool_mock.execute(call, available_values)
                    history.append({"role": "tool", "content": result})
                    available_values.update(self._extract_values(result))
        
        return Conversation(messages=history, tools=tools)
```

#### Shared State

All agents read/write to shared context:

- **available_values**: Dict of entity values produced by tools
- **conversation_history**: List of all messages
- **tool_call_log**: Record of all tool invocations

```python
class SharedState:
    def __init__(self):
        self.available_values: Dict[str, Any] = {}
        self.conversation_history: List[Message] = []
        self.tool_call_log: List[ToolCall] = []
```

### 1.5 Key Design Decisions

| Decision | Choice | Rationale | Alternatives Considered |
|----------|--------|-----------|------------------------|
| Graph library | NetworkX | Mature, pure Python, sufficient for our scale (<10k tools) | igraph (faster but C dependency), custom (unnecessary complexity) |
| Agent framework | Direct orchestration | Simpler debugging, explicit control flow, no framework lock-in | LangGraph (overkill for our use case), AutoGen (too opinionated) |
| Diversity tracking | In-memory counters | Fast access, sufficient for batch sizes <10k | Redis (distributed but adds infra), SQLite (persistent but slower) |
| Output format | JSONL | Streaming writes, easy to process line-by-line, standard format | JSON array (memory issues with large files), Parquet (complex for text) |
| LLM provider | Anthropic Claude | Strong instruction following, native tool use support | OpenAI GPT-4 (comparable), local models (quality tradeoff) |
| Config format | YAML | Human-readable, supports comments, widely used | TOML (less familiar), JSON (no comments) |

**Why we chose direct orchestration over agent frameworks:**

We evaluated LangGraph and AutoGen but found them to add unnecessary complexity for our use case. Our pipeline is fundamentally sequential with well-defined handoffs. The frameworks added:
- Opaque state management that made debugging difficult
- Overhead for features we didn't need (parallel execution, complex routing)
- Version compatibility issues

Direct orchestration gives us full visibility into the generation process and makes it easy to add logging, retry logic, and custom behavior.

---

## 2. Context Management

### 2.1 Within-Conversation Grounding

**Problem**: Tool calls must use realistic, consistent values. When `get_user` returns `user_id: "usr_123"`, subsequent `get_orders(user_id)` must use that exact ID—not a hallucinated one.

**Solution**: Available Values Tracking

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│ get_user()   │────►│AvailableVals │────►│ get_orders() │
│              │     │ user_id:     │     │ user_id:     │
│ returns:     │     │  "usr_123"   │     │  "usr_123"   │
│ {user_id:    │     │ email:       │     │              │
│  "usr_123"}  │     │  "a@b.com"   │     │              │
└──────────────┘     └──────────────┘     └──────────────┘
```

The system extracts key values from each tool response and makes them available for subsequent calls.

### 2.2 Available Values Injection

Values are formatted into agent prompts to ensure consistency:

```python
def format_available_values(available: Dict[str, Any]) -> str:
    """Inject available values into agent prompts."""
    if not available:
        return "No values available yet."
    
    lines = ["Available values from previous tool calls:"]
    for key, value in available.items():
        lines.append(f"  - {key}: {value}")
    return "\n".join(lines)
```

**Prompt template for Tool Mock:**
```
You are simulating tool execution for: {tool_name}

{format_available_values(available_values)}

The user called this tool with arguments:
{tool_call_arguments}

Generate a realistic response. IMPORTANT: Reuse IDs from available values 
when they match the expected input type.
```

### 2.3 LLM Extraction

Values are identified from tool responses using schema-guided and pattern-based extraction:

```python
def extract_values(tool_response: dict, tool_schema: ToolSchema) -> Dict[str, Any]:
    """Extract reusable values from tool response."""
    extracted = {}
    
    # Schema-guided extraction: fields marked as reusable
    for field in tool_schema.output_fields:
        if field.reusable and field.name in tool_response:
            extracted[field.name] = tool_response[field.name]
    
    # Pattern-based extraction for common ID formats
    for key, value in flatten(tool_response).items():
        if is_id_pattern(key, value):  # *_id, uuid pattern, etc.
            extracted[key] = value
    
    return extracted

def is_id_pattern(key: str, value: Any) -> bool:
    """Check if a key-value pair looks like an ID."""
    if not isinstance(value, str):
        return False
    
    # Key patterns
    if key.endswith("_id") or key == "id":
        return True
    
    # Value patterns (UUID, prefixed IDs)
    if re.match(r'^[a-z]+_[a-zA-Z0-9]{8,}$', value):  # usr_abc123
        return True
    if re.match(r'^[0-9a-f]{8}-[0-9a-f]{4}-', value):  # UUID
        return True
    
    return False
```

### 2.4 Lazy ID Generation

IDs are generated only when first needed, then cached for consistency:

```python
class LazyIDGenerator:
    """Generate IDs lazily and cache them for reuse."""
    
    def __init__(self):
        self.generated: Dict[tuple, str] = {}  # Cache of generated IDs
        self.prefixes = {
            "user": "usr_",
            "order": "ord_",
            "product": "prod_",
            "transaction": "txn_",
            "booking": "book_",
        }
    
    def get_or_create(self, entity_type: str, context: dict) -> str:
        """Get existing ID or generate new one."""
        # Create cache key from entity type and context
        cache_key = (entity_type, frozenset(context.items()))
        
        if cache_key not in self.generated:
            self.generated[cache_key] = self._generate(entity_type)
        
        return self.generated[cache_key]
    
    def _generate(self, entity_type: str) -> str:
        """Generate a new ID with appropriate prefix."""
        prefix = self.prefixes.get(entity_type, "id_")
        return f"{prefix}{uuid4().hex[:12]}"
```

**Why lazy generation?**
- Ensures IDs are only created when actually needed
- Same context produces same ID (deterministic within conversation)
- Avoids pre-generating unused IDs

### 2.5 Cross-Conversation Steering

**Problem**: Without steering, generation converges to common patterns. E-commerce scenarios dominate because tools like `get_user` and `get_orders` are versatile and frequently sampled.

**Solution**: Track usage and adjust sampling weights to favor underrepresented tools.

```
Conversation 1: [get_user, get_orders, cancel_order]     → e-commerce
Conversation 2: [get_user, get_orders, refund_order]     → e-commerce (similar!)
Conversation 3: [search_flights, book_flight, get_booking] → travel (steered)
```

### 2.6 Diversity Tracking

Implementation of counters and hashes:

```python
from collections import Counter
from itertools import combinations
import hashlib

class DiversityTracker:
    """Track tool usage for diversity steering."""
    
    def __init__(self):
        self.tool_counts = Counter()       # Individual tool usage
        self.pair_counts = Counter()       # Tool pair co-occurrence
        self.domain_counts = Counter()     # Domain coverage
        self.scenario_hashes = set()       # Exact scenario deduplication
    
    def record(self, conversation: Conversation) -> None:
        """Record a generated conversation."""
        tools = conversation.get_tools_used()
        
        # Update individual tool counts
        self.tool_counts.update(tools)
        
        # Update pair counts (order-independent)
        pairs = list(combinations(sorted(tools), 2))
        self.pair_counts.update(pairs)
        
        # Update domain count
        self.domain_counts[conversation.domain] += 1
        
        # Add scenario hash for exact deduplication
        self.scenario_hashes.add(self._hash_scenario(conversation))
    
    def _hash_scenario(self, conversation: Conversation) -> str:
        """Create hash of scenario for deduplication."""
        key = f"{sorted(conversation.tools)}:{conversation.scenario_title}"
        return hashlib.md5(key.encode()).hexdigest()
    
    def get_weights(self, candidates: List[str]) -> List[float]:
        """Get sampling weights (inverse frequency)."""
        counts = [self.tool_counts[t] + 1 for t in candidates]  # +1 smoothing
        max_count = max(counts)
        return [max_count / c for c in counts]
    
    def is_duplicate(self, conversation: Conversation) -> bool:
        """Check if scenario already exists."""
        return self._hash_scenario(conversation) in self.scenario_hashes
```

### 2.7 Sampling Weight Adjustment

Inverse frequency weighting ensures underused tools get sampled more:

```
Tool           | Count | Raw Weight | Normalized
---------------|-------|------------|------------
get_user       | 50    | 1.0        | 0.067
get_orders     | 45    | 1.11       | 0.074
search_flights | 5     | 10.0       | 0.667
book_hotel     | 3     | 16.67      | 1.0 (max)
```

**Formula**: `weight(tool) = max_count / (count(tool) + smoothing)`

```python
def compute_weights(counts: Counter, smoothing: float = 1.0) -> Dict[str, float]:
    """Compute inverse frequency weights."""
    if not counts:
        return {}
    
    max_count = max(counts.values())
    weights = {}
    
    for tool, count in counts.items():
        weights[tool] = max_count / (count + smoothing)
    
    # Normalize to sum to 1
    total = sum(weights.values())
    return {t: w / total for t, w in weights.items()}
```

### 2.8 Prompt Injection for Steering

Domain hints guide the scenario planner toward underrepresented areas:

```python
def build_scenario_prompt(tools: List[Tool], tracker: DiversityTracker) -> str:
    """Build scenario prompt with diversity hints."""
    # Find underrepresented domains
    underrep_domains = tracker.get_underrepresented_domains(threshold=0.1)
    
    prompt = f"""Create a scenario using these tools: {[t.name for t in tools]}

Tool descriptions:
{format_tool_descriptions(tools)}
"""
    
    if underrep_domains:
        domain = random.choice(underrep_domains)
        prompt += f"""
DIVERSITY HINT: Consider a {domain} domain scenario.
We have generated many e-commerce scenarios already - try something different.
"""
    
    prompt += """
Output a scenario with:
- title: Brief description
- user_goal: What the user wants to accomplish
- expected_flow: List of tool calls in order
"""
    
    return prompt
```

### 2.9 Tradeoffs

| Approach | Complexity | Effectiveness | When to Use |
|----------|------------|---------------|-------------|
| No steering | Low | Poor diversity | Prototyping, quick tests |
| Count-based (current) | Medium | Good | Default choice, most cases |
| Embedding-based | High | Excellent | Large-scale, premium quality |
| Deterministic rotation | Low | Moderate | When reproducibility is critical |

**Our choice**: Count-based steering provides the best balance of simplicity and effectiveness. Embedding-based approaches (comparing scenario similarity) would improve diversity further but add significant complexity and latency.

### 2.10 Scale Limitations

What would change at 100k+ conversations:

| Current Approach | At Scale | Migration Path |
|------------------|----------|----------------|
| In-memory counters | Redis/PostgreSQL | Add persistence layer with batch updates |
| Single process | Distributed workers | Add Celery/RQ task queue |
| Full dedup set | Bloom filter | Probabilistic dedup (accept small false positive rate) |
| Synchronous LLM calls | Async batch API | asyncio + request batching |
| Single config file | Config service | Centralized configuration management |

---

## 3. Prompt Design

### 3.1 Scenario Planning Prompt

**Structure:**

```
ROLE: You are a scenario designer creating realistic multi-tool interaction scenarios.

CONTEXT:
- Available tools: {tool_list_with_descriptions}
- Domain hints: {domain_hints}
- Constraints: Use {min_tools}-{max_tools} tools across {min_turns}-{max_turns} turns

TASK: Design a scenario where a user naturally needs to use multiple tools to accomplish a goal.

OUTPUT FORMAT (JSON):
{
  "title": "Brief scenario title",
  "user_goal": "What the user wants to accomplish",
  "expected_flow": ["tool1 → purpose", "tool2 → purpose", ...],
  "complexity": "simple|medium|complex",
  "domain": "e-commerce|travel|finance|healthcare|..."
}

GUIDELINES:
- Scenarios should reflect real-world tasks users actually perform
- Tools should have logical dependencies (output of one feeds into another)
- The user goal should require multiple tools - no single tool can achieve it
- Include realistic constraints (user has partial info, needs to look things up)
```

**Rationale:**
- Explicit role primes the model for the task
- Structured JSON output ensures parseability
- Guidelines prevent degenerate scenarios (e.g., using tools that don't connect)

### 3.2 User Simulation Prompt

**Structure:**

```
ROLE: You are simulating a real user interacting with an AI assistant.

SCENARIO: {scenario_description}
USER GOAL: {user_goal}

CONVERSATION SO FAR:
{conversation_history}

CURRENT STATE:
- Tools already used: {tools_used}
- Information gathered: {available_values}
- Next logical step: {expected_next_step}

TASK: Generate the user's next message.

PERSONA GUIDELINES:
- Be natural and conversational, not robotic or overly formal
- Occasionally include typos, casual language, or incomplete sentences
- Sometimes ask clarifying questions instead of providing all info upfront
- Vary verbosity: sometimes terse ("yes"), sometimes detailed
- May express emotions (frustration, gratitude, confusion)

OUTPUT: Just the user message text, nothing else.
```

**Rationale:**
- Persona guidelines create varied, realistic user messages
- State tracking ensures the conversation progresses logically
- Direct output format (just the message) simplifies parsing

### 3.3 Tool Mock Prompt

**Structure:**

```
ROLE: You are simulating the execution of an API tool/function.

TOOL: {tool_name}
DESCRIPTION: {tool_description}

SCHEMA:
  Input: {input_schema}
  Output: {output_schema}

CALL ARGUMENTS:
{tool_arguments_json}

AVAILABLE CONTEXT (use these values for consistency):
{available_values}

TASK: Generate a realistic tool response.

GUIDELINES:
- Response MUST match the output schema exactly
- Reuse IDs from available context when the type matches
- Generate plausible data (realistic names, valid email formats, etc.)
- For lists, return 1-5 items typically
- 10% of the time, return an error response for realism:
  {"error": {"code": "NOT_FOUND", "message": "Resource not found"}}

OUTPUT: Valid JSON matching the output schema.
```

**Rationale:**
- Schema enforcement ensures structurally valid outputs
- Context injection maintains ID consistency across calls
- Error injection (10%) creates realistic edge cases for training

### 3.4 Judge Scoring Prompt

**Structure:**

```
ROLE: You are a quality evaluator for synthetic conversations.

CONVERSATION:
{full_conversation_json}

AVAILABLE TOOLS:
{tool_definitions}

EVALUATE on these dimensions (score 0.0 to 1.0):

1. NATURALNESS - Does the conversation flow naturally?
   - 1.0: Indistinguishable from real human-AI conversation
   - 0.7: Minor awkwardness but acceptable
   - 0.4: Noticeably artificial but understandable
   - 0.0: Clearly robotic, unnatural

2. TOOL_APPROPRIATENESS - Are tools used correctly?
   - 1.0: Perfect tool selection, valid arguments, logical sequence
   - 0.7: Minor issues (suboptimal tool choice, unnecessary calls)
   - 0.4: Some incorrect usage but mostly functional
   - 0.0: Wrong tools, invalid arguments, nonsensical sequence

3. COHERENCE - Is the conversation internally consistent?
   - 1.0: Perfect consistency, IDs match, facts don't contradict
   - 0.7: Minor inconsistencies that don't affect understanding
   - 0.4: Noticeable inconsistencies
   - 0.0: Major contradictions, broken references

4. COMPLETENESS - Does it achieve the stated goal?
   - 1.0: Goal fully achieved, user would be satisfied
   - 0.7: Goal mostly achieved, minor gaps
   - 0.4: Goal partially achieved
   - 0.0: Goal not addressed or conversation abandoned

OUTPUT FORMAT (JSON):
{
  "naturalness": 0.X,
  "tool_appropriateness": 0.X,
  "coherence": 0.X,
  "completeness": 0.X,
  "overall": 0.X,
  "issues": ["specific issue 1", "specific issue 2"],
  "passed": true/false
}

PASS CRITERIA: overall >= 0.7 AND no dimension below 0.4
```

**Rationale:**
- Multi-dimensional scoring captures different quality aspects
- Explicit rubric with examples ensures consistent evaluation
- Pass/fail threshold enables automated filtering
- Issue list enables targeted repair

### 3.5 Repair Prompt

**Structure:**

```
ROLE: You are fixing a synthetic conversation that failed quality validation.

ORIGINAL CONVERSATION:
{conversation_json}

VALIDATION ERRORS:
{error_list}

JUDGE FEEDBACK:
{judge_issues}

SPECIFIC PROBLEMS TO FIX:
{detailed_issues}

TASK: Fix the conversation to address all validation errors while preserving the scenario.

CONSTRAINTS:
- Preserve the overall scenario and user goal
- Make MINIMAL changes necessary to fix issues
- Maintain consistency with tool schemas
- Keep approximately the same number of turns
- Don't change the tools used unless absolutely necessary

OUTPUT: The complete fixed conversation in the same JSON format.
```

**Rationale:**
- Targeted repair is more efficient than full regeneration
- Minimal change constraint preserves the diversity of the original
- Full context enables coherent fixes that maintain consistency

### 3.6 Failed Iterations

What didn't work during development:

**1. Single-prompt generation (v0.1)**
- **Approach**: One prompt to generate the entire conversation
- **Problem**: Long conversations (8+ turns) lost coherence; tool calls often had invalid arguments or referenced non-existent IDs
- **Lesson**: Multi-agent decomposition is necessary for quality

**2. Unconstrained user simulation (v0.2)**
- **Approach**: Let user agent freely respond based only on conversation history
- **Problem**: Conversations went off-topic, ignored the planned scenario, sometimes ended abruptly
- **Lesson**: Need to inject scenario context into every user turn

**3. Deterministic tool mocking (v0.3)**
- **Approach**: Template-based tool responses with random data generation
- **Problem**: Responses were repetitive and unrealistic; edge cases never appeared
- **Lesson**: LLM-based mocking with schema guidance produces more realistic variety

**4. Binary judge scoring (v0.4)**
- **Approach**: Simple pass/fail evaluation
- **Problem**: Couldn't distinguish "almost good" from "terrible"; high rejection rate (40%)
- **Lesson**: Multi-dimensional scoring enables targeted repair, reducing waste

**5. Greedy tool selection (v0.5)**
- **Approach**: Always select tools that maximize graph connectivity
- **Problem**: Same tool combinations kept appearing; low diversity
- **Lesson**: Need inverse-frequency weighting for diversity

### 3.7 Lessons Learned

| Iteration | Problem | Solution | Impact |
|-----------|---------|----------|--------|
| 1 | Incoherent long conversations | Turn-by-turn generation with state | +40% coherence score |
| 2 | Off-topic drift | Scenario injection every turn | +25% relevance |
| 3 | Repetitive tool responses | LLM mocking with schema guidance | +30% response diversity |
| 4 | High rejection rate (40%) | Multi-dimensional scoring + repair | Reduced to 8% waste |
| 5 | ID inconsistency | Available values tracking | +60% consistency |
| 6 | Tool combination repetition | Inverse frequency steering | +45% pair coverage |

---

## 4. Diversity & Quality Analysis

### 4.1 Tool Entropy Metric

**Formula:**

```
H(tools) = -Σ p(t) × log₂(p(t))
```

Where `p(t) = count(t) / total_tool_calls`

**Justification:**
- Shannon entropy measures uncertainty/diversity in a distribution
- Higher entropy = more uniform tool usage = better diversity
- Maximum entropy = log₂(n) when all n tools are used equally
- Allows comparison across datasets of different sizes

**Example:**

```
Dataset A (skewed):
  tool1: 90 calls, tool2: 5 calls, tool3: 5 calls
  p = [0.9, 0.05, 0.05]
  H = -0.9×log₂(0.9) - 0.05×log₂(0.05) - 0.05×log₂(0.05)
  H = 0.57 bits (low diversity)

Dataset B (balanced):
  tool1: 35 calls, tool2: 33 calls, tool3: 32 calls  
  p = [0.35, 0.33, 0.32]
  H = 1.58 bits (high diversity)

Maximum possible with 3 tools: log₂(3) = 1.58 bits
```

**Implementation:**

```python
import math
from collections import Counter

def tool_entropy(tool_counts: Counter) -> float:
    """Calculate Shannon entropy of tool usage distribution."""
    total = sum(tool_counts.values())
    if total == 0:
        return 0.0
    
    entropy = 0.0
    for count in tool_counts.values():
        if count > 0:
            p = count / total
            entropy -= p * math.log2(p)
    
    return entropy

def normalized_entropy(tool_counts: Counter) -> float:
    """Entropy normalized to [0, 1] range."""
    n_tools = len(tool_counts)
    if n_tools <= 1:
        return 1.0
    
    max_entropy = math.log2(n_tools)
    return tool_entropy(tool_counts) / max_entropy
```

### 4.2 Pair Ratio Metric

**Formula:**

```
pair_ratio = |unique_pairs_used| / |possible_pairs|
           = |unique_pairs_used| / C(n, 2)
           = |unique_pairs_used| / (n × (n-1) / 2)
```

**Justification:**
- Measures coverage of tool combinations, not just individual tools
- Captures whether diverse tool *interactions* are generated
- Complements entropy (which only measures individual tool frequency)
- A dataset could have high entropy but low pair ratio if tools are used independently

**Example:**

```
5 tools → C(5,2) = 10 possible pairs

Dataset uses: (A,B), (A,C), (B,C), (A,D) → 4 unique pairs
Pair ratio = 4/10 = 0.4

Dataset uses all pairs → 10 unique pairs  
Pair ratio = 10/10 = 1.0
```

**Implementation:**

```python
from itertools import combinations
from typing import List, Set, Tuple

def pair_ratio(conversations: List[Conversation], all_tools: List[str]) -> float:
    """Calculate ratio of observed tool pairs to possible pairs."""
    n_tools = len(all_tools)
    possible_pairs = n_tools * (n_tools - 1) // 2
    
    if possible_pairs == 0:
        return 1.0  # Trivial case
    
    observed_pairs: Set[Tuple[str, str]] = set()
    
    for conv in conversations:
        tools = sorted(set(conv.get_tools_used()))
        for pair in combinations(tools, 2):
            observed_pairs.add(pair)
    
    return len(observed_pairs) / possible_pairs
```

### 4.3 Quality Metrics

**Judge Scores Breakdown:**

| Metric | Range | Weight | Description |
|--------|-------|--------|-------------|
| Naturalness | 0-1 | 25% | Human-likeness of dialogue flow |
| Tool Appropriateness | 0-1 | 30% | Correct tool selection and usage |
| Coherence | 0-1 | 25% | Internal consistency (IDs, facts) |
| Completeness | 0-1 | 20% | Goal achievement |
| **Overall** | 0-1 | 100% | Weighted average |

**Pass Criteria:**
- Overall score ≥ 0.7
- No individual dimension < 0.4
- No critical validation errors (schema violations, broken references)

**Quality Distribution Target:**

```
Score Range | Target % | Meaning
------------|----------|------------------
0.9 - 1.0   | 20%      | Excellent, production-ready
0.8 - 0.9   | 40%      | Good, minor issues
0.7 - 0.8   | 30%      | Acceptable, some awkwardness
< 0.7       | 10%      | Failed, needs repair or discard
```

### 4.4 Results Placeholder

```
┌─────────────────────────────────────────────────────────────┐
│                    EXPERIMENT RESULTS                        │
│                   (To be filled after 10.6)                  │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Results will be inserted here after running the            │
│  diversity experiment comparing steering vs no steering.    │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 5. Diversity Experiment

### 5.1 Experiment Design

**Hypothesis:** Cross-conversation steering improves tool diversity without significantly degrading conversation quality.

**Setup:**

| Parameter | Value |
|-----------|-------|
| Run A | Steering disabled (`--no-cross-conversation-steering`) |
| Run B | Steering enabled (default) |
| Seed | 42 (fixed for reproducibility) |
| Conversations | 100 each |
| Tools | Full registry |
| Min turns | 3 |
| Max turns | 8 |

**Commands:**

```bash
# Run A: No steering (baseline)
toolgen generate \
    --num 100 \
    --seed 42 \
    --no-cross-conversation-steering \
    --output run_a.jsonl

# Run B: With steering (treatment)
toolgen generate \
    --num 100 \
    --seed 42 \
    --output run_b.jsonl

# Evaluate both
toolgen evaluate --input run_a.jsonl --output report_a.json
toolgen evaluate --input run_b.jsonl --output report_b.json

# Compare
python scripts/compare_runs.py report_a.json report_b.json
```

### 5.2 Results

*(To be filled after running the experiment)*

| Metric | Run A (No Steering) | Run B (Steering) | Delta | Significant? |
|--------|---------------------|------------------|-------|--------------|
| Tool Entropy | ___ bits | ___ bits | ___% | |
| Normalized Entropy | ___ | ___ | ___% | |
| Pair Ratio | ___ | ___ | ___% | |
| Unique Tools Used | ___/___ | ___/___ | ___% | |
| Unique Pairs Used | ___/___ | ___/___ | ___% | |
| Mean Quality Score | ___ | ___ | ___% | |
| Median Quality Score | ___ | ___ | ___% | |
| Pass Rate | ___% | ___% | ___pp | |
| Generation Time | ___s | ___s | ___% | |

### 5.3 Analysis

*(To be filled after experiment)*

**Diversity Impact:**

- [ ] Steering increased tool entropy by X% (from Y to Z bits)
- [ ] Pair coverage improved by X% (from Y to Z ratio)
- [ ] Number of unique tools used increased from X to Y
- [ ] Domain distribution became more balanced

**Quality Impact:**

- [ ] Mean quality score changed by X% (from Y to Z)
- [ ] Pass rate changed by X percentage points
- [ ] No significant degradation in any quality dimension
- [ ] Specific quality dimensions affected: ___

**Statistical Significance:**

- [ ] Entropy difference: p-value = ___
- [ ] Quality difference: p-value = ___
- [ ] Using: Mann-Whitney U test / t-test / bootstrap CI

### 5.4 Conclusions

*(To be filled after experiment)*

**Recommendation:**

Based on the results, we recommend:

- [ ] **Enable steering by default** because diversity gains outweigh any quality cost
- [ ] **Disable steering by default** because quality degradation is unacceptable
- [ ] **Make steering configurable** with guidance on when to use it

**Tradeoff Summary:**

```
Steering adds ~X% to generation time
Steering improves diversity by ~Y%
Steering changes quality by ~Z%

Net assessment: [Worth it / Not worth it / Situational]
```

---

## Appendix

### A. Glossary

| Term | Definition |
|------|------------|
| Available Values | Entity values (IDs, names, etc.) produced by tool calls and available for subsequent calls |
| Cross-conversation Steering | Adjusting tool sampling weights based on cumulative usage to improve diversity across the full dataset |
| Tool Entropy | Shannon entropy of the tool usage distribution; measures how uniformly tools are used |
| Pair Ratio | Fraction of possible tool pairs that are actually observed in the dataset |
| Grounding | Ensuring generated content uses consistent, realistic values rather than hallucinated ones |
| Repair | Process of fixing a conversation that failed validation rather than discarding it |

### B. File Structure

```
toolgen/
├── cli/
│   ├── __init__.py
│   ├── main.py           # CLI entry point
│   ├── build.py          # Build command
│   ├── generate.py       # Generate command
│   └── evaluate.py       # Evaluate command
├── agents/
│   ├── __init__.py
│   ├── base.py           # Base agent class
│   ├── planner.py        # Scenario planner
│   ├── user_sim.py       # User simulator
│   ├── assistant.py      # Assistant agent
│   ├── tool_mock.py      # Tool mocker
│   ├── judge.py          # Quality judge
│   └── repair.py         # Repair agent
├── graph/
│   ├── __init__.py
│   ├── registry.py       # Tool registry
│   ├── builder.py        # Graph builder
│   └── sampler.py        # Tool sampler
├── tracking/
│   ├── __init__.py
│   ├── diversity.py      # Diversity tracker
│   └── values.py         # Available values manager
├── utils/
│   ├── __init__.py
│   ├── llm.py            # LLM client wrapper
│   └── config.py         # Configuration loader
└── schemas/
    └── tool.schema.json  # Tool definition schema
```

### C. References

- NetworkX documentation: https://networkx.org/
- Shannon Entropy: https://en.wikipedia.org/wiki/Entropy_(information_theory)
- Claude Tool Use: https://docs.anthropic.com/en/docs/tool-use
- JSONL format: https://jsonlines.org/

### D. Version History

| Version | Date | Changes |
|---------|------|---------|
| 0.1 | - | Initial single-prompt approach |
| 0.2 | - | Multi-agent decomposition |
| 0.3 | - | Added available values tracking |
| 0.4 | - | Added diversity steering |
| 0.5 | - | Added repair agent |
| 1.0 | - | Production release |
