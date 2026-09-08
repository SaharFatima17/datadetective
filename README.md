# DataDetective

**Autonomous Data Investigation Agent** — a multi-agent system that takes a business
question and a dataset, forms hypotheses, tests them with trusted tools, verifies every
number, forecasts what happens next, and recommends an action — with each claim traceable
to the calculation that produced it.

> Setup instructions: **[SETUP.md](SETUP.md)**

---

## What it does

```
source → profile → clean → question → hypotheses → ask for missing evidence
       → test with tools → compare with history → critique → verify
       → forecast → recommend → versioned report
```

Three principles run through the whole system:

1. **The LLM never computes a number.** It plans, phrases hypotheses, critiques and
   narrates. Every figure comes from pandas, DuckDB, SciPy or statsmodels.
2. **Nothing is overwritten.** Cleaning creates a new dataset version with a lineage
   record; the original stays byte-for-byte intact.
3. **Missing evidence is requested, not invented.** A hypothesis that cannot be tested is
   marked unresolved and the user is asked for what is needed.

---

## Proposal coverage

| § | Requirement | Where |
|---|---|---|
| 6 | Multi-agent architecture | `app/agents/` |
| 7 | 24-step workflow | `app/agents/orchestrator.py` |
| 9 | MCP tool layer | `app/tools/registry.py`, `mcp_server.py` |
| 10 | Profiling, cleaning, lineage | `app/services/profiling.py`, `cleaning.py` |
| 11 | Hypothesis generation and testing | `app/agents/investigation_agents.py` |
| 12 | Forecasting with backtesting | `app/services/forecasting.py` |
| 13 | Evidence-backed recommendations | `app/services/recommendations.py` |
| 14 | RAG and vector retrieval | `app/services/rag.py` |
| 15 | Database design (22 tables) | `app/models/` |
| 17 | User interface | `static/index.html` |
| 18 | Evidence-backed reporting | `app/agents/orchestrator.py` |
| 19 | Security and responsible design | throughout — see below |
| 19 | Authentication, RBAC, encryption at rest, sensitive-column masking | `app/core/` |
| 20 | Four-way architecture comparison and ablations | `app/evaluation/`, `scripts/run_comparison.py` |
| 21 | Six benchmark scenarios with withheld forecast periods | `scripts/generate_benchmark.py` |
| 22 | Implementation phases 1–14 | complete |

---

## Design decisions that depart from the proposal

Each of these was a deliberate choice made while building, and each is defensible in the
FYP report.

**Four "agents" became tools.** The proposal lists sixteen agents, but Profiler, Cleaning,
SQL and Statistical are deterministic pipelines with no judgement to exercise. Making them
LLM agents would add a network hop, latency, cost and a failure mode for no capability
gain. They are implemented in `app/services/` and exposed through the tool registry. The
agents that remain — Supervisor, Hypothesis, Evidence Gap, Critic, History, Report — are
the ones that genuinely need a model.

**The Verifier is not an LLM.** Asking a model to check its own arithmetic is circular.
Instead every tool call is recorded in `tool_runs` with a checksum of its result, and
verification re-executes the recorded call and compares checksums. Verification either
passes mechanically or it does not.

**Findings are typed.** §18 requires observation, statistical evidence and interpretation
to be distinguishable, so every finding carries a `finding_type`:

- `measurement` — what changed (a trend, a total). Not an explanation.
- `association` — two things move together. Not causal.
- `driver` — the change is concentrated in a specific segment.

Only `driver` findings may produce a recommendation. This is what stops the system
answering "why did revenue fall?" with "revenue fell 34%".

**Contribution analysis, not group size.** The first implementation ranked segments by
total size and rejected the true cause in the benchmark. The question is which group drove
the *change*, so `period_contribution` compares per-period values across two periods —
matching §11 and the §18 worked example. A small segment that collapses is found; a large
stable one is not.

**Impact estimates state their assumption.** §13's "recover 8–12% of lost revenue" is a
counterfactual, not a measurement. `impact_method` records the exact arithmetic and says
plainly that the recovery fraction is an assumption tied to confidence, not a causal
quantity.

**Vectors in JSONB, not pgvector.** §16 allows pgvector *or* Qdrant. At FYP scale, cosine
similarity in Python over a JSONB column avoids a difficult build step and is fast enough.
Swapping in pgvector later touches two files.

**Forecasting has hard minimums.** ARIMA will fit eight monthly points and return
confident nonsense. Below 12 periods no forecast is produced at all; below 24 no seasonal
model is used; a forecast that fails backtesting or loses to a naive baseline is marked
`low_confidence` with the reason shown.

---

## Safety (§19)

- No `exec` anywhere — `run_dataframe_code` is a whitelist of five named operations
- SQL is SELECT-only, keyword-filtered and row-capped, on both the ingestion and query paths
- Database credentials for external sources are used once and never persisted
- Destructive cleaning requires explicit per-operation approval
- Soft deletes only — lineage is never destroyed
- Correlation is never reported as causation; forecasts are never reported as certainty
- Recommendations are advisory and never execute against any system

---

## Verified results

```
pytest -q                                83 passed
python scripts/run_evaluation.py         Root-cause accuracy: 6/6 = 100%
python scripts/run_quality_evaluation.py profiling, cleaning and internals
python scripts/run_comparison.py --ablations
```

Seven benchmark scenarios. Six carry a root cause injected on purpose:

| Scenario | True cause | System's lead finding |
|---|---|---|
| regional_decline | South region collapses | region='South' contributed 94% of the decline |
| stock_shortage | Beta availability falls | product='Beta' contributed 93% of the decline |
| price_increase | elastic demand | price and revenue correlated, r=−0.93 |
| missing_evidence | not in the data | asks for explanatory data instead of asserting |
| multi_source | warehouse incident, only in a document | region='North' found; document retrieved |
| changed_driver | driver differs between two periods | product='Y', and the change from region 'A' is reported |

Two of these carry most of the weight. `missing_evidence` checks that the system measures
the drop, finds no column that could explain it, and asks rather than naming a cause.
`changed_driver` checks that a second investigation on a later period notices its answer
is no longer the same as last time.

The seventh, `dirty_data`, carries eight defects injected on purpose instead — missing
values, negative revenue, extreme outliers, inconsistent casing, a typo label, unparseable
dates, duplicate identifiers and exact duplicate rows — which is what makes data-quality
precision and recall measurable rather than a matter of opinion.

Every scenario also withholds its final three periods from the file entirely, so forecasts
are scored against actuals the system never saw — not only against their own backtest.

### Measured behaviour (proposal Sec.20)

`scripts/run_quality_evaluation.py` scores the layers underneath the conclusion:

| Metric | Result |
|---|---|
| Data-quality detection recall / precision | 1.0 / 1.0 |
| Defect resolution after cleaning | 1.0 |
| Valid data retention (nothing over-cleaned) | 1.0 |
| Numerical accuracy vs. injected magnitudes | 1.0 |
| Hypothesis relevance and resolution rate | 1.0 |
| Verification accuracy | 1.0 |
| Statistical test appropriateness | 1.0 |

Three detected defects are deliberately left unrepaired and scored separately rather than
counted as failures: fuzzy label typos (merging on similarity alone destroys real
distinctions), duplicate identifiers (only the user knows which row is authoritative), and
outliers (an outlier is frequently the thing being investigated).

### Architecture comparison (proposal Sec.20)

`scripts/run_comparison.py` runs Baseline A (LLM with a data summary only), Baseline B
(single agent with tools), Baseline C (B plus retrieval), the proposed system, and four
ablations, then scores them identically.

**The baselines require a real LLM provider.** With `LLM_PROVIDER=mock` they exercise the
plumbing and nothing more; the script prints a warning and records
`llm_provider_was_mock` in its output. Do not quote mock-mode baseline numbers as a result.

The ablations are meaningful under either provider, because the proposed system runs
without model judgement. They isolate one change at a time, which the four-way comparison
cannot do — Baseline C and the proposed system differ in five ways at once.

---

## Layout

```
app/
  models/       22 SQLAlchemy tables (proposal Sec.15)
  services/     ingestion, profiling, cleaning, analytics, rag, forecasting, recommendations
  tools/        the tool registry — one entry point, every call logged and checksummed
  agents/       supervisor, investigation agents, orchestrator state machine
  llm/          provider abstraction (mock | anthropic | openai | gemini)
  api/routes/   36 endpoints
static/         single-page workspace UI
scripts/        benchmark generator, evaluation runner
tests/          pytest suite for the deterministic layers
mcp_server.py   MCP surface over the same registry
```

---

## Remaining work

- **Phase 15** — the final FYP report.
- **Run the comparison with a real LLM.** Everything is built; the baselines need a
  provider key before their numbers mean anything.
- **Frontend** — the API exposes charts, the investigation timeline, the evidence table,
  SQL and URL ingestion, drift, and the evidence-supply endpoint. The web UI currently
  surfaces Health, Clean, Ask and Report only, so some working backend features are not
  visible in the browser.
