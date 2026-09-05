# ArmadaCrew

Multi-agent runtime for incident response and customer ops. Supervisor routes to researcher / SRE / support, a critic can send work back, and high-risk side effects pause for a human. State is checkpointed to SQLite after every node so a process death is a resume, not a restart.

This is the LangGraph production shape (state, nodes, conditional edges, reducers, checkpointer, interrupt) implemented in-tree. There is no `langgraph` dependency; the mapping is in the table below if you already use that stack.

## Run

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest -q
python -m armadacrew serve
```

http://127.0.0.1:8080

```bash
python -m armadacrew run "Draft a postmortem for yesterday's auth outage"
python -m armadacrew run "API p99 latency spiked on payments-api in prod" --approve
```

Default LLM is a local structured-JSON mock (`ARMADACREW_LLM_PROVIDER=mock`). Optional extras: `.[openai]` / `.[anthropic]`.

## Graph

```
task → supervisor → researcher → supervisor → sre | support
                         ↓
                      critic ──reject──► specialist
                         ↓ approve
                      writer → END
                         ↓
              high risk → hitl (interrupt) → resume
```

HITL is mandatory for prod restarts, refunds over $500, and outbound customer email.

| LangGraph | here |
| --- | --- |
| StateGraph | `GraphState` in `runtime.py` |
| nodes / conditional edges | `agents.py`, `supervisor.py` |
| reducers | `LIST_APPEND_FIELDS` |
| SqliteSaver | `SqliteCheckpointer` |
| interrupt | `NodeResult.interrupt` + `POST /v1/runs/{id}/resume` |
| ToolNode | `ToolRegistry` (schema, timeout, retry, audit) |

## Sample tasks

- `API p99 latency spiked on payments-api in prod`
- `Customer Ada Lovelace wants a $720 refund for duplicate charge`
- `Draft a postmortem for yesterday's auth outage`

Runbooks and KB live under `data/`. Tools talk to those fixtures (metrics, logs, CRM), not the public internet.

## API

```
POST /v1/runs                 { "task", "mode" }   mode: incident | customer_ops | auto
GET  /v1/runs/{id}
POST /v1/runs/{id}/resume     { "approved", "comment" }
GET  /v1/runs/{id}/events     SSE
```

MIT.
