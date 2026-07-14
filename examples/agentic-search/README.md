# Wiring ml-commons to the agent server `/invoke` endpoint

These are example registrations for connecting an OpenSearch cluster's
agentic-search path to the agent server's `/invoke` endpoint, targeting the
`dsl_generator` agent. They are **configuration, not code** — JSON you POST to a
running cluster's REST API. Fill in your own agent-server URL and credential
before using them.

## What's here

| File | What it is |
|---|---|
| `01_connector.json` | ml-commons **connector** — describes the HTTP call to `/invoke` (dsl_generator agent). |
| `02_flow_agent.json` | ml-commons **FLOW agent** with one `ConnectorTool` that calls the connector. |

No ml-commons Java is written: the connector and FLOW agent are registered via
the cluster's REST API. The only code involved is the agent server's
`dsl_generator` agent (`src/server/dsl_agent.py`), reached through `/invoke`.

## How the DSL reaches neural-search

`extractFlowAgentResult` (neural-search) reads the DSL only from
`ModelTensor.result` (a string). A stock `ConnectorTool` would park an HTTP JSON
response in `dataAsMap`, leaving `result` empty. The connector sends
`response_format: inference_results`, so `/invoke` returns the ml-commons
envelope; the built-in
`post_process_function: connector.post_process.mlcommons.passthrough` then copies
`output[0].result` into `ModelTensor.result`.

## Register against a cluster

Requires a running OpenSearch cluster with ml-commons (e.g. `./gradlew run` in
the ml-commons repo) and a reachable agent server.

```bash
# Enable the agentic-search feature flag if needed:
#   PUT _cluster/settings {"persistent":{"plugins.ml_commons.agentic_search_enabled":true}}

# 1. Set the agent-server URL in 01_connector.json (parameters.endpoint), then
#    register the connector (returns a connector_id):
curl -s -XPOST http://localhost:9200/_plugins/_ml/connectors/_create \
  -H 'Content-Type: application/json' --data-binary @examples/agentic-search/01_connector.json
#    -> {"connector_id":"abc123..."}

# 2. Paste that id into 02_flow_agent.json (replace PASTE_CONNECTOR_ID_FROM_STEP_1),
#    then register the FLOW agent (returns an agent_id):
curl -s -XPOST http://localhost:9200/_plugins/_ml/agents/_register \
  -H 'Content-Type: application/json' --data-binary @examples/agentic-search/02_flow_agent.json
#    -> {"agent_id":"xyz789..."}

# 3. Execute the agent to generate DSL for a question:
curl -s -XPOST http://localhost:9200/_plugins/_ml/agents/<agent_id>/_execute \
  -H 'Content-Type: application/json' \
  -d '{"parameters":{"question":"active items","index_name":"my-index"}}'
```

The `_execute` response carries the generated DSL in its `result` field,
confirming the connector → passthrough → `ModelTensor.result` chain.

## Notes

- Set `parameters.endpoint` in `01_connector.json` to your agent server's host.
- If the cluster's trusted-endpoints regex blocks the host, add it via
  `plugins.ml_commons.trusted_connector_endpoints_regex` (do not widen this in
  production).
- If the agent server is on a private IP (e.g. `localhost` or an in-VPC host),
  ml-commons blocks the connector with "host name has private ip address"
  unless `plugins.ml_commons.connector.private_ip_enabled` is set to `true`.
- The `_comment*` fields in the JSON are documentation; the connector parser
  ignores unknown fields, but remove them if a stricter cluster rejects them.
