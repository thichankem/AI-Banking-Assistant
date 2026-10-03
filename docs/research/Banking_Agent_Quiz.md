# Banking Agent Quiz

## Question 1
In the LangGraph Orchestrator's `AgentState` definition, why is the `messages` attribute annotated with `add_messages`?

- [x] It specifies a custom Reducer that appends new messages to the chat history rather than overwriting existing entries.
- [ ] It automatically converts user inputs into JSON-RPC 2.0 payloads for tool execution.
- [ ] It computes vector embeddings for all messages before passing them to the router.
- [ ] It enforces PostgreSQL Row-Level Security directly on session state objects.

**Hint:** Consider how state attributes prevent context loss when multiple sub-agents write updates to the same history.

## Question 2
During Giai đoạn 5 (Evaluator-Optimizer & HITL) of the data flow, what specific trigger causes the orchestrator to execute the `interrupt()` function?

- [x] The detection of a high-risk financial action, such as locking a card, requiring explicit user approval.
- [ ] An invalid JSON schema structure during MCP tool parameter validation.
- [ ] A Critic Node Faithfulness score falling below $0.4$.
- [ ] The active session cache in Redis reaching its 24-hour TTL expiration.

**Hint:** Focus on actions that create irreversible side-effects on a user's banking account.

## Question 3
Why does the benchmark framework prioritize State-Verification Testing over traditional static Q&A pair matching for banking agents?

- [x] It inspects backend database state changes in a sandbox to verify actual operational side-effects rather than relying on textual claims.
- [ ] It guarantees that the LLM produces identical word-for-word responses across evaluation runs.
- [ ] It eliminates the necessity of running asynchronous background memory workers.
- [ ] It bypasses Row-Level Security checks to speed up evaluation pipeline throughput.

**Hint:** Think about what happens if an agent states 'Your card is locked' but fails to call the locking API.

## Question 4
How does the Late Chunking method resolve the 'Coreference Trap' in banking policy and regulation documents?

- [x] It processes the entire document through a long-context Transformer Encoder to generate global token embeddings before mean pooling into chunks.
- [ ] It cuts text strictly at paragraph boundaries when cosine embedding similarity drops below a dynamic variance threshold.
- [ ] It parses syntax trees using Tree-Sitter to extract structured fee tables without breaking rows.
- [ ] It splits documents into fixed 512-token segments and appends global metadata to each segment header.

**Hint:** Consider how context is captured across the entire document prior to dividing vector representations.

## Question 5
In the Corrective RAG (CRAG) pipeline, what decision path is taken when the evaluator score $S_{eval}$ falls in the ambiguous range of $0.4 \le S_{eval} \le 0.75$?

- [x] It triggers a Decompose-then-Recompose workflow to break the query into sub-questions and re-retrieve information.
- [ ] It feeds the retrieved context directly into the reasoning engine without alteration.
- [ ] It discards all retrieved documents and redirects the user query to an external web search fallback.
- [ ] It immediately flags the interaction as a Prompt Injection attempt.

**Hint:** This middle confidence tier indicates uncertainty that can be resolved by refining the search query.

## Question 6
How does PostgreSQL Row-Level Security (RLS) isolate multi-tenant user data in the `semantic_memories` table?

- [x] It evaluates RLS policies matching `tenant_id` and `user_id` columns against dynamic session settings `app.current_tenant_id` and `app.current_user_id`.
- [ ] It encrypts vector embeddings with tenant-specific RSA keys before writing to the HNSW index.
- [ ] It provisions separate physical database instances for every tenant via FastAPI ingress routing.
- [ ] It purges non-tenant memory records from the Redis buffer on every incoming HTTP request.

**Hint:** Focus on how database engine policies check session configuration variables.

## Question 7
Based on the memory decay function $S_{memory}(t) = w_i \cdot I_{base} + w_r \cdot e^{-\lambda \cdot (t - t_{last})} + w_f \cdot \log(1 + C_{access})$, how does repeated access to a semantic fact affect its priority?

- [x] It increases the access count $C_{access}$, raising the logarithmic term value and counteracting temporal decay.
- [ ] It resets the baseline importance score $I_{base}$ to zero to prevent buffer saturation.
- [ ] It causes the decay coefficient $\lambda$ to increase exponentially over time.
- [ ] It forces the time difference $(t - t_{last})$ to become negative.

**Hint:** Analyze the impact of $C_{access}$ in the logarithmic component of the formula.

## Question 8
For which scenario would the orchestrator select the Tree-of-Thoughts (ToT) / MCTS reasoning engine over standard ReAct?

- [x] Formulating an optimal financial plan distributing 500 million VND across savings, cash-back cards, and investment options.
- [ ] Checking the current balance of a customer's primary checking account.
- [ ] Retrieving standard operating hours for a local bank branch.
- [ ] Locating the nearest physical ATM given user coordinates.

**Hint:** Distinguish between fast, routine tool lookups and multi-scenario optimization tasks.

## Question 9
What standard protocol and payload structure are used by Subgraph Agent Nodes to invoke MCP Tool Layer servers?

- [x] JSON-RPC 2.0 payloads delivered over Server-Sent Events (SSE) or stdio streams.
- [ ] GraphQL mutations over gRPC transport streams.
- [ ] SOAP XML structures transmitted via HTTP POST requests.
- [ ] Protocol Buffer binary objects over raw WebSocket channels.

**Hint:** Identify the protocol explicitly defined for tool execution between subgraphs and MCP servers.

## Question 10
Under what condition does the LLM-as-a-Judge evaluation framework automatically route an agent response to a Human Reviewer?

- [x] When the standard deviation $\sigma$ across $N=5$ evaluation runs exceeds $0.5$.
- [ ] When the average evaluation score $\bar{S}$ falls strictly below $0.75$.
- [ ] When the JSON schema precision rate for tool parameters hits $100\%$.
- [ ] When the P95 latency penalty value equals $0.05$.

**Hint:** Look for the statistical metric measuring score dispersion across repeated judge runs.

## Question 11
Which of the following business subgraphs are defined in the execution path of the LangGraph Stateful Orchestrator? (Select all that apply.)

- [x] Subgraph 1: Quy trình & Thủ tục (Open Account, Card Registration, Loan Filing)
- [x] Subgraph 2: Vận hành & Xử lý Sự cố (Transaction Error, Card Block, OTP, App Login)
- [x] Subgraph 3: Tư vấn Sản phẩm & Tài chính (Card Comparison, Loan Repayment Schedule)
- [x] Subgraph 4: Tra cứu Chung & Policy FAQ (Fee Matrix, Operating Hours, Terms)
- [ ] Subgraph 5: Giao dịch Chứng khoán Tự động (Automated Stock Trading & Execution)

**Hint:** Recall the four core business subgraphs detailed in the LangGraph architecture specifications.

## Question 12
Which processing actions can be taken by Giai đoạn 5 (Evaluator-Optimizer & HITL) after evaluating raw output from a Subgraph Agent Node? (Select all that apply.)

- [x] Instruct the Subgraph to regenerate the response if hallucination or missing details are flagged (up to 2 times).
- [x] Trigger a LangGraph `interrupt()` and wait for user confirmation if a dangerous financial action is detected.
- [x] Forward the validated response directly to the output response stream if quality and safety standards are satisfied.
- [ ] Purge all historical tenant facts stored in PostgreSQL when a hallucination flag is raised.

**Hint:** Identify pathways corresponding to output retry, human confirmation, and direct streaming.

## Question 13
Which techniques belong to the system's Adaptive Chunking Suite? (Select all that apply.)

- [x] Dynamic Semantic Chunking based on continuous sentence embedding variance.
- [x] Late Chunking leveraging long-context Transformer Encoders.
- [x] Tree-Sitter AST Chunking for structured API specs and tabular fee matrices.
- [ ] Static character slicing ignoring syntax boundaries and sentence structure.

**Hint:** Select methods designed to adaptively preserve semantic or structural context.

## Question 14
Which of the following belong to the four Dynamic Evaluation Methods implemented in the evaluation framework? (Select all that apply.)

- [x] State-Verification Testing against a Mock Banking Database Sandbox.
- [x] TAU-Bench Style Agent Simulation employing User Simulator Agents with varying personas.
- [x] Deterministic Algorithmic Scoring such as Tree Edit Distance for tool trajectory matching.
- [ ] Single-turn static text similarity evaluation using BLEU or ROUGE metrics against fixed golden answers.

**Hint:** Filter out traditional static text-matching methods in favor of dynamic evaluation techniques.

## Question 15
What tasks are performed asynchronously by the background Async Memory Consolidation Worker? (Select all that apply.)

- [x] Extracting atomic facts from completed conversation turns using an LLM.
- [x] Checking for existing duplicate or conflicting facts using vector similarity search.
- [x] Resolving memory conflicts by marking outdated contradicting facts as `is_active = False`.
- [ ] Synchronously locking the user interface stream until PostgreSQL database writes finish.

**Hint:** Consider tasks involved in background fact extraction, deduplication, deactivation, and non-blocking database writes.

## Question 16
In the End-to-End System Score formula $\text{Score}_{\text{System}} = w_1 \cdot Pass@1 + w_2 \cdot SVR + w_3 \cdot Faithfulness + w_4 \cdot IS - w_5 \cdot \text{Penalty}_{\text{Latency}}$, which components contribute positively to the score? (Select all that apply.)

- [x] $Pass@1$ (End-to-end task completion rate in TAU-Bench simulation)
- [x] $SVR$ (State-Verification Success Rate in database sandbox)
- [x] $Faithfulness$ (RAG factual faithfulness score)
- [x] $IS$ (Multi-Tenant Isolation Score)
- [ ] $\text{Penalty}_{\text{Latency}}$ (Latency threshold penalty factor)

**Hint:** Identify metrics that represent positive performance achievements rather than subtracted deductions.

## Question 17
In the Hybrid Search retriever, dense vector HNSW search scores and sparse BM25 search scores are combined using the ___ (RRF) algorithm.


**Hint:** Name the ranking algorithm that sums inverted document positions across multiple search algorithms.

## Question 18
To pause orchestration during Human-in-the-Loop workflows, LangGraph invokes the ___ function call when detecting high-risk operations.


**Hint:** Provide the exact LangGraph function name used to suspend graph execution for approval.

## Question 19
The Short-Term Buffer tier in the multi-tenant memory taxonomy utilizes a Redis Cluster configured with a Time-To-Live (TTL) of ___ hours.


**Hint:** State the numerical duration in hours designated for active session caching in Redis.

## Question 20
Deterministic verification of interest computations and repayment schedules relies on comparing agent outputs directly against an independent AST ___ Engine.


**Hint:** Specify the term paired with AST that designates the deterministic calculation module.

## Question 21
Hard negative mining packages low-scoring conversation traces into prompt, chosen, and rejected trajectories to produce datasets for Direct Preference Optimization, abbreviated as ___.


**Hint:** Provide the three-letter acronym for preference optimization model tuning.

## Question 22
Explain how the Human-in-the-Loop (HITL) mechanism operates when a customer requests the AI Banking Agent to lock a lost bank card.


**Hint:** Outline the steps from risk identification by the Critic Node to graph pausing and explicit UI approval.

## Question 23
Why is State-Verification Testing considered superior to static text-matching evaluation for conversational banking agents?


**Hint:** Contrast evaluating surface text output against inspecting database state mutations.

## Question 24
How does the Async Memory Consolidation Worker handle memory updates when a user provides new personal information that contradicts a previously stored semantic memory fact?


**Hint:** Describe the workflow from vector similarity checking to setting deactivation flags on outdated facts.

## Question 25
What structural advantage does GraphRAG using Leiden Community Detection offer when answering complex cross-product banking questions?


**Hint:** Focus on entity-relationship graphs and community summarization for multi-product comparison queries.

## Question 26
How does Deterministic Algorithmic Scoring measure the trajectory efficiency of an agent's reasoning and tool execution sequence?


**Hint:** Explain how actual and optimal tool execution sequences are compared using edit distance algorithms.
