# Wiring ml-commons to the agent server `/invoke` endpoint

Example registrations that connect an OpenSearch cluster's agentic-search path to
the agent server's `/invoke` endpoint, targeting the `agentic_search` agent. This
is **configuration, not code** — JSON you POST to a running cluster's REST API. No
ml-commons Java is involved; the only code is the agent server's `agentic_search`
agent (`src/agents/agentic_search/`), reached through `/invoke`.

## How the DSL reaches neural-search

`extractFlowAgentResult` (neural-search) reads the generated DSL only from
`ModelTensor.result` (a string). A stock `ConnectorTool` would park an HTTP JSON
response in `dataAsMap`, leaving `result` empty. The connector sends
`response_format: inference_results`, so `/invoke` returns the ml-commons envelope;
the built-in `post_process_function: connector.post_process.mlcommons.passthrough`
then copies `output[0].result` into `ModelTensor.result`.

## Register against a cluster

Requires a running OpenSearch cluster with ml-commons (e.g. `./gradlew run`) and a
reachable agent server.

### 1. Register the connector

Describes the HTTP call to `/invoke`. `${parameters.question}` and
`${parameters.index_name}` are filled from the FLOW agent's runtime parameters at
call time; `agent` selects `agentic_search`; `response_format: inference_results`
makes `/invoke` return the envelope the passthrough reads. Set
`parameters.endpoint` to your agent server's host. The `Authorization` header must
use `${credential.*}` — security-sensitive headers reject `${parameters.*}`.

```bash
curl -s -XPOST http://localhost:9200/_plugins/_ml/connectors/_create \
  -H 'Content-Type: application/json' -d '{
  "name": "Agentic Search Remote DSL Connector",
  "description": "Calls the agent server /invoke endpoint (agentic_search agent).",
  "version": "1",
  "protocol": "http",
  "parameters": { "endpoint": "127.0.0.1:8001" },
  "credential": { "token": "replace-with-agent-server-token" },
  "actions": [{
    "action_type": "predict",
    "method": "POST",
    "url": "http://${parameters.endpoint}/invoke",
    "headers": {
      "Content-Type": "application/json",
      "Authorization": "Bearer ${credential.token}"
    },
    "request_body": "{ \"query\": \"${parameters.question}\", \"agent\": \"agentic_search\", \"context\": { \"index_name\": \"${parameters.index_name}\" }, \"response_format\": \"inference_results\" }",
    "post_process_function": "connector.post_process.mlcommons.passthrough"
  }]
}'
#    -> {"connector_id":"abc123..."}
```

### 2. Register the FLOW agent

A single-tool FLOW agent returns its tool's output directly. Paste the
`connector_id` from step 1; `connector_action: predict` matches the connector's
action above.

```bash
curl -s -XPOST http://localhost:9200/_plugins/_ml/agents/_register \
  -H 'Content-Type: application/json' -d '{
  "name": "Agentic Search Remote DSL Agent",
  "type": "flow",
  "description": "Single-tool FLOW agent that generates OpenSearch DSL via the agent server.",
  "tools": [{
    "type": "ConnectorTool",
    "name": "remote_dsl_generator",
    "description": "Generates OpenSearch DSL from a natural-language question.",
    "parameters": { "connector_id": "PASTE_CONNECTOR_ID_FROM_STEP_1", "connector_action": "predict" }
  }]
}'
#    -> {"agent_id":"xyz789..."}
```

### 3. Execute the agent

```bash
curl -s -XPOST http://localhost:9200/_plugins/_ml/agents/<agent_id>/_execute \
  -H 'Content-Type: application/json' \
  -d '{"parameters":{"question":"active items","index_name":"my-index"}}'
```

The `_execute` response carries the generated DSL in its `result` field,
confirming the connector → passthrough → `ModelTensor.result` chain.

## Notes

- If the cluster's trusted-endpoints regex blocks the host, add it via
  `plugins.ml_commons.trusted_connector_endpoints_regex` (do not widen this in
  production).
- If the agent server is on a private IP (e.g. `localhost` or an in-VPC host),
  ml-commons rejects the connector with "host name has private ip address" unless
  `plugins.ml_commons.connector.private_ip_enabled` is `true`.
