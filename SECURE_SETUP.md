# Secured Local Development Setup

Build everything from source and run the full stack with security enabled (HTTPS, OBO token forwarding) across 4 terminal windows.

## Prerequisites

- Java 21 (e.g. Amazon Corretto)
- Node.js 22+ and Yarn 1
- Python 3.12+
- `uv` (Python package manager: `curl -LsSf https://astral.sh/uv/install.sh | sh`)
- `jq`, `curl`, `unzip`
- AWS credentials configured (for Bedrock LLM)

## Step 0: Set Up Working Directory

```bash
mkdir -p agent-quickstart && cd agent-quickstart
WORKSPACE=$(pwd)
```

All repos will be cloned into this directory.

## Step 1: Clone and Build OpenSearch

```bash
# Clone OpenSearch
git clone --depth 1 https://github.com/opensearch-project/OpenSearch.git
cd OpenSearch

# Build the min distribution (tar.gz without bundled JDK)
./gradlew :distribution:archives:no-jdk-darwin-tar:assemble -x test 2>&1 | tail -3

# Publish to local Maven (required for the security plugin build)
./gradlew publishToMavenLocal -x test -x javadoc --parallel 2>&1 | tail -3

cd $WORKSPACE
```

> **Linux users:** Replace `no-jdk-darwin-tar` with `no-jdk-linux-tar`.

## Step 2: Clone and Build the Security Plugin

The security plugin must be built from source to match the OpenSearch version (e.g. 3.6.0-SNAPSHOT).

```bash
git clone --depth 1 https://github.com/opensearch-project/security.git
cd security

./gradlew assemble -x test -x integrationTest 2>&1 | tail -3

# Verify the zip was built
ls build/distributions/opensearch-security-*.zip

cd $WORKSPACE
```

## Step 3: Clone ml-commons (Optional, for ML features)

```bash
git clone https://github.com/opensearch-project/ml-commons.git
cd ml-commons

# Build the plugin zip
OPENSEARCH_CORE_PATH=$WORKSPACE/OpenSearch \
  ./gradlew :opensearch-ml-plugin:bundlePlugin -x test -Pcrypto.standard=FIPS-140-3 2>&1 | tail -3

# Verify
ls plugin/build/distributions/opensearch-ml-*.zip

cd $WORKSPACE
```

## Step 4: Assemble the Secured OpenSearch Cluster

Extract the distribution and install plugins in dependency order.

```bash
OS_HOME=$WORKSPACE/opensearch-secure

# Extract the min distribution
mkdir -p $OS_HOME
tar xzf OpenSearch/distribution/archives/no-jdk-darwin-tar/build/distributions/opensearch-min-*.tar.gz \
  --strip-components=1 -C $OS_HOME

# Copy bc-fips jar (required by security plugin, not included in min distribution)
BC_FIPS=$(find ~/.gradle/caches/modules-2 -name "bc-fips-*.jar" 2>/dev/null | head -1)
if [ -n "$BC_FIPS" ]; then
  cp "$BC_FIPS" $OS_HOME/lib/
  echo "Copied $(basename $BC_FIPS) to lib/"
fi

# 1) Install job-scheduler (dependency for ml-commons)
JS_ZIP=$(find ~/.gradle/caches/modules-2 -name "opensearch-job-scheduler-*.zip" 2>/dev/null | head -1)
if [ -n "$JS_ZIP" ]; then
  $OS_HOME/bin/opensearch-plugin install --batch "file://$JS_ZIP"
fi

# 2) Install security plugin
SECURITY_ZIP=$(ls security/build/distributions/opensearch-security-*.zip | head -1)
$OS_HOME/bin/opensearch-plugin install --batch "file://$SECURITY_ZIP"

# 3) Install ml-commons (optional)
ML_ZIP=$(ls ml-commons/plugin/build/distributions/opensearch-ml-*.zip 2>/dev/null | head -1)
if [ -n "$ML_ZIP" ]; then
  $OS_HOME/bin/opensearch-plugin install --batch "file://$ML_ZIP"
fi

# 4) Run security demo configuration (generates TLS certs + initial security index)
export OPENSEARCH_INITIAL_ADMIN_PASSWORD='MyStr0ngP@ss!'
bash $OS_HOME/plugins/opensearch-security/tools/install_demo_configuration.sh -y -i -s

# 5) Enable single-node discovery
echo "discovery.type: single-node" >> $OS_HOME/config/opensearch.yml
```

> **Plugin install order matters:** job-scheduler must be installed before ml-commons (dependency). Security can be installed at any point before starting.

## Step 5: Configure OBO (On-Behalf-Of) Tokens

OBO tokens allow Dashboards to mint short-lived JWTs that carry the logged-in user's identity and permissions.

**1. Edit the security config:**

```bash
vi $OS_HOME/config/opensearch-security/config.yml
```

Add the `on_behalf_of` block directly under the `dynamic:` key (same indentation level as `http:`, `authc:`, etc.):

```yaml
    config:
      dynamic:
        on_behalf_of:
          enabled: true
          signing_key: "VGhpcyBpcyB0aGUgand0IHNpZ25pbmcga2V5IGZvciBhbiBvbiBiZWhhbGYgb2YgdG9rZW4gYXV0aGVudGljYXRpb24gYmFja2VuZCBmb3IgdGVzdGluZyBvZiBleHRlbnNpb25z"
          encryption_key: "VGhpcyBpcyB0aGUgand0IGVuY3J5cHRpb24ga2V5IGZvciBhbiBvbiBiZWhhbGYgb2YgdG9rZW4gYXV0aGVudGljYXRpb24gYmFja2VuZCBmb3IgdGVzdGluZyBvZiBleHRlbnNpb25z"
        http:
          # ... rest of existing config
```

Alternatively, you can use `sed` to inject it (macOS):

```bash
# Copy clean config from security source
cp security/config/config.yml $OS_HOME/config/opensearch-security/config.yml

# Inject OBO config after the "dynamic:" line
sed -i '' '/^  dynamic:$/a\
\    on_behalf_of:\
\      enabled: true\
\      signing_key: "VGhpcyBpcyB0aGUgand0IHNpZ25pbmcga2V5IGZvciBhbiBvbiBiZWhhbGYgb2YgdG9rZW4gYXV0aGVudGljYXRpb24gYmFja2VuZCBmb3IgdGVzdGluZyBvZiBleHRlbnNpb25z"\
\      encryption_key: "VGhpcyBpcyB0aGUgand0IGVuY3J5cHRpb24ga2V5IGZvciBhbiBvbiBiZWhhbGYgb2YgdG9rZW4gYXV0aGVudGljYXRpb24gYmFja2VuZCBmb3IgdGVzdGluZyBvZiBleHRlbnNpb25z"
' $OS_HOME/config/opensearch-security/config.yml
```

> **Important:** The REST API `PATCH /_plugins/_security/api/securityconfig` returns `403 FORBIDDEN` for config changes. You **must** use `securityadmin.sh`.

## Terminal 1: Start OpenSearch

```bash
cd $WORKSPACE/opensearch-secure

OPENSEARCH_JAVA_HOME=$JAVA_HOME bin/opensearch
```

Wait for it to start, then verify in another terminal:

```bash
AUTH=$(echo -n 'admin:MyStr0ngP@ss!' | base64)
curl -sk -H "Authorization: Basic $AUTH" https://localhost:9200
```

> **Note:** The `@` in the password breaks curl's `-u` flag. Always use base64-encoded Authorization headers.

### Apply OBO Config (first time only)

After OpenSearch is running, apply the security config:

```bash
OS_HOME=$WORKSPACE/opensearch-secure

JAVA_HOME=$JAVA_HOME bash \
  $OS_HOME/plugins/opensearch-security/tools/securityadmin.sh \
  -f $OS_HOME/config/opensearch-security/config.yml \
  -t config \
  -icl -nhnv \
  -key $OS_HOME/config/kirk-key.pem \
  -cert $OS_HOME/config/kirk.pem \
  -cacert $OS_HOME/config/root-ca.pem
```

Flags: `-f <file> -t config` uploads the file as config type, `-icl` ignores cluster name, `-nhnv` skips hostname verification, `-key/-cert/-cacert` are admin TLS certs generated by demo config.

Verify OBO tokens work:

```bash
AUTH=$(echo -n 'admin:MyStr0ngP@ss!' | base64)
curl -sk -H "Authorization: Basic $AUTH" -X POST \
  'https://localhost:9200/_plugins/_security/api/obo/token' \
  -H 'Content-Type: application/json' \
  -d '{"description": "test"}'
```

You should see a response containing `authenticationToken`.

### Fix Disk Watermark Blocks (if needed)

If you see `cluster create-index blocked` or `flood-stage watermark` errors:

```bash
AUTH=$(echo -n 'admin:MyStr0ngP@ss!' | base64)

curl -sk -H "Authorization: Basic $AUTH" -X PUT \
  'https://localhost:9200/_cluster/settings' \
  -H 'Content-Type: application/json' \
  -d '{"persistent":{"cluster.routing.allocation.disk.watermark.flood_stage":"99%","cluster.routing.allocation.disk.watermark.high":"98%","cluster.routing.allocation.disk.watermark.low":"97%"}}'

curl -sk -H "Authorization: Basic $AUTH" -X PUT \
  'https://localhost:9200/.kibana_1/_settings' \
  -H 'Content-Type: application/json' \
  -d '{"index.blocks.read_only_allow_delete":null}'
```

## Terminal 2: MCP Server (port 3030)

The MCP server must use header-based auth (`OPENSEARCH_HEADER_AUTH=true`) so it forwards the OBO Bearer token to OpenSearch instead of using basic auth.

```bash
export OPENSEARCH_URL="https://localhost:9200"
export OPENSEARCH_HEADER_AUTH=true
export OPENSEARCH_SSL_VERIFY=false
# Do NOT set OPENSEARCH_USERNAME or OPENSEARCH_PASSWORD

uv tool run opensearch-mcp-server-py --transport stream --port 3030
```

Verify: you should see `Uvicorn running on http://0.0.0.0:3030` in the output.

When requests come in, the logs should show `[HEADER AUTH] Using Authorization Bearer header` (not `[BASIC AUTH]`).

## Terminal 3: Agent Server (port 8001)

```bash
cd opensearch-agent-server
source .venv/bin/activate
python run_server.py
```

The agent server reads `.env` for configuration:

```bash
# .env
MCP_SERVER_URL=http://localhost:3030/mcp
AG_UI_AUTH_ENABLED=false
AG_UI_CORS_ORIGINS=http://localhost:5601
```

Verify: `http://localhost:8001/health` should return OK.

## Terminal 4: OpenSearch Dashboards (port 5601)

### Clone and Bootstrap

```bash
cd $WORKSPACE
git clone --depth 1 https://github.com/opensearch-project/OpenSearch-Dashboards.git
cd OpenSearch-Dashboards

yarn osd bootstrap
```

### Configure Dashboards

Edit `config/opensearch_dashboards.yml` — set these values (uncomment or add):

```yaml
# Connect to secured OpenSearch
opensearch.hosts: ["https://localhost:9200"]
opensearch.username: "admin"
opensearch.password: "MyStr0ngP@ss!"
opensearch.ssl.verificationMode: none

# Enable new home page UI (required for chat button)
uiSettings:
  overrides:
    "home:useNewHomePage": true
    

# Enable chat with AG-UI agent server + OBO token forwarding
chat:
  enabled: true
  agUiUrl: "http://localhost:8001/runs"
  forwardCredentials: true

# Enable context provider (sends page context to agent)
contextProvider:
  enabled: true
```

### Start Dashboards

```bash
yarn start --no-base-path
```

Wait for `bundles compiled successfully` in the output (takes a few minutes on first run).

### Open in Browser

Navigate to **http://localhost:5601** and log in:

- **Username:** `admin`
- **Password:** `MyStr0ngP@ss!`

The chat button should appear in the top header. Send a message to test the full flow.

## Startup Order

Start components in this order (each must be ready before the next):

1. **OpenSearch** — wait for `https://localhost:9200` to respond
2. **MCP Server** — wait for `Uvicorn running on http://0.0.0.0:3030`
3. **Agent Server** — wait for `Starting OpenSearch Agent Server on 0.0.0.0:8001`
4. **Dashboards** — wait for `bundles compiled successfully`

## Architecture (Secured OBO Flow)

```
Browser
  -> http://localhost:5601 (Dashboards)
    -> POST /api/chat/proxy (Dashboards server-side)
      1. Mint OBO token: POST https://localhost:9200/_plugins/_security/api/obo/token
      2. Forward to Agent Server with Authorization: Bearer <obo_token>
        -> http://localhost:8001/runs (Agent Server)
          -> http://localhost:3030/mcp (MCP Server, OPENSEARCH_HEADER_AUTH=true)
            -> https://localhost:9200 (OpenSearch, validates Bearer JWT)
```

With `forwardCredentials: true`, Dashboards generates a short-lived OBO JWT (5 min TTL) for the logged-in user and sends it as a Bearer token through the entire chain. OpenSearch validates the JWT signature, extracts the user identity (`sub`) and encrypted roles, and enforces that user's permissions.

## Creating Test Users

Create users with different permission levels to test role-based access via OBO:

```bash
AUTH=$(echo -n 'admin:MyStr0ngP@ss!' | base64)

# Create analyst (read-only)
curl -sk -H "Authorization: Basic $AUTH" -X PUT \
  'https://localhost:9200/_plugins/_security/api/internalusers/analyst' \
  -H 'Content-Type: application/json' \
  -d '{"password":"R3adOnly@sec","backend_roles":["readall"],"description":"Read-only analyst user"}'

# Create developer (read + write)
curl -sk -H "Authorization: Basic $AUTH" -X PUT \
  'https://localhost:9200/_plugins/_security/api/internalusers/developer' \
  -H 'Content-Type: application/json' \
  -d '{"password":"Dev1@pass123","backend_roles":["readall"],"description":"Developer user"}'
```

### Create Custom Roles

```bash
# Analyst role: read-only + cluster monitor + OBO token generation
curl -sk -H "Authorization: Basic $AUTH" -X PUT \
  'https://localhost:9200/_plugins/_security/api/roles/analyst_role' \
  -H 'Content-Type: application/json' \
  -d '{
    "cluster_permissions": ["cluster_monitor","security:obo/create"],
    "index_permissions": [{"index_patterns":["*"],"allowed_actions":["read","indices_monitor","get"]}]
  }'

# Developer role: read + write + manage indices + OBO
curl -sk -H "Authorization: Basic $AUTH" -X PUT \
  'https://localhost:9200/_plugins/_security/api/roles/developer_role' \
  -H 'Content-Type: application/json' \
  -d '{
    "cluster_permissions": ["cluster_monitor","cluster_manage_index_templates","security:obo/create"],
    "index_permissions": [{"index_patterns":["*"],"allowed_actions":["crud","create_index","manage","indices_monitor"]}]
  }'
```

> **Note:** The `security:obo/create` permission is required for non-admin users to generate OBO tokens.

### Map Users to Roles

```bash
# Map to custom roles
curl -sk -H "Authorization: Basic $AUTH" -X PUT \
  'https://localhost:9200/_plugins/_security/api/rolesmapping/analyst_role' \
  -H 'Content-Type: application/json' -d '{"users":["analyst"]}'

curl -sk -H "Authorization: Basic $AUTH" -X PUT \
  'https://localhost:9200/_plugins/_security/api/rolesmapping/developer_role' \
  -H 'Content-Type: application/json' -d '{"users":["developer"]}'

# Map to built-in roles for index read + Dashboards access
curl -sk -H "Authorization: Basic $AUTH" -X PUT \
  'https://localhost:9200/_plugins/_security/api/rolesmapping/readall' \
  -H 'Content-Type: application/json' -d '{"users":["analyst","developer"]}'

curl -sk -H "Authorization: Basic $AUTH" -X PUT \
  'https://localhost:9200/_plugins/_security/api/rolesmapping/kibana_user' \
  -H 'Content-Type: application/json' -d '{"users":["analyst","developer"]}'
```

### Test Users

| User | Password | Can Read | Can Write | OBO Token |
|------|----------|----------|-----------|-----------|
| `admin` | `MyStr0ngP@ss!` | Yes | Yes | Yes |
| `analyst` | `R3adOnly@sec` | Yes | No | Yes |
| `developer` | `Dev1@pass123` | Yes | Yes | Yes |

To test: log in as `analyst` in Dashboards, open chat, and ask the agent to create an index. It should fail with a permissions error. Log in as `developer` and the same request should succeed.

> **Note:** The agent server caches agents per name. Restart the agent server between user switches to ensure the new user's OBO token is used.

## Credentials

| Key | Value |
|-----|-------|
| Admin password | `MyStr0ngP@ss!` |
| OBO signing key (base64) | `VGhpcyBpcyB0aGUgand0IHNpZ25pbmcga2V5IGZvciBhbiBvbiBiZWhhbGYgb2YgdG9rZW4gYXV0aGVudGljYXRpb24gYmFja2VuZCBmb3IgdGVzdGluZyBvZiBleHRlbnNpb25z` |
| OBO signing key (decoded) | `This is the jwt signing key for an on behalf of token authentication backend for testing of extensions` |
| OBO encryption key (base64) | `VGhpcyBpcyB0aGUgand0IGVuY3J5cHRpb24ga2V5IGZvciBhbiBvbiBiZWhhbGYgb2YgdG9rZW4gYXV0aGVudGljYXRpb24gYmFja2VuZCBmb3IgdGVzdGluZyBvZiBleHRlbnNpb25z` |

## Stopping Everything

```bash
# Stop OpenSearch (if running in daemon mode)
kill $(cat $WORKSPACE/opensearch-secure/opensearch.pid)

# Stop MCP Server, Agent Server, Dashboards
# Ctrl+C in each terminal
```

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `Unauthorized` with curl | The `@` in the password breaks curl's `-u` flag. Use base64 auth header instead (see examples above). |
| `cluster create-index blocked` | Disk watermark triggered. Clear with the cluster settings commands in the OpenSearch section. |
| `registerPPLValidationProvider is not a function` | Stale build cache. Run `yarn osd bootstrap` then restart Dashboards. |
| `MCP client session is not running` on 2nd chat | Restart the agent server. Fixed by the `mcp_client.start()` change in `default_agent.py`. |
| `fetch failed` on chat | Agent server not running on port 8001. |
| `Failed to start MCP client` | MCP server not running on port 3030. |
| `[BASIC AUTH]` in MCP logs | MCP server using basic auth instead of OBO token. Restart with `OPENSEARCH_HEADER_AUTH=true` and unset `OPENSEARCH_USERNAME`/`OPENSEARCH_PASSWORD`. |
| `Password is similar to user name` | Security plugin rejects passwords that resemble the username. Use a different password. |
| `no permissions for [security:obo/create]` | User's role is missing `security:obo/create` cluster permission. Add it to the role. |
| Security plugin `NoClassDefFoundError: BouncyCastleFipsProvider` | Copy `bc-fips-*.jar` from `~/.gradle/caches/modules-2/` into `$OS_HOME/lib/`. |
| `Missing plugin [opensearch-job-scheduler]` when installing ml-commons | Install job-scheduler before ml-commons. |
| Security REST API returns `403 FORBIDDEN` for config changes | Use `securityadmin.sh` instead of the REST API for security config modifications. |
