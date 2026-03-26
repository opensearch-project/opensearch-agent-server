#!/usr/bin/env bash
# =============================================================================
# OpenSearch Agent Server — Quickstart
#
# Sets up and starts all services needed for the OpenSearch Agent + Search
# Relevance Workbench development environment:
#
#   1. OpenSearch (via search-relevance repo's ./gradlew start)
#   2. OpenSearch Dashboards (with dashboards-search-relevance plugin)
#   3. OpenSearch MCP Server
#   4. OpenSearch Agent Server
#   5. Search Relevance demo data
#
# Usage:
#   ./scripts/quickstart.sh              # full setup + start
#   ./scripts/quickstart.sh --start      # start only (skip clone/build)
#   ./scripts/quickstart.sh --stop       # stop all running services
#   ./scripts/quickstart.sh --status     # check service status
#
# Prerequisites:
#   - Java 21 (e.g. Amazon Corretto 21)
#   - Node.js 20.x (via nvm)
#   - Python 3.12+
#   - uv (pip install uv, or curl -LsSf https://astral.sh/uv/install.sh | sh)
#   - jq, curl
# =============================================================================
set -euo pipefail

# Restore cursor and clean up on exit/interrupt
cleanup() {
  printf '\033[?25h' 2>/dev/null || true
}
trap cleanup EXIT
trap 'cleanup; exit 130' INT TERM

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
WORKSPACE="$PROJECT_ROOT/agent-quickstart"
PID_DIR="$WORKSPACE/.pids"
LOG_DIR="$WORKSPACE/.logs"

# --- Repo URLs ---------------------------------------------------------------
SEARCH_RELEVANCE_REPO="https://github.com/opensearch-project/search-relevance.git"
DASHBOARDS_REPO="https://github.com/opensearch-project/OpenSearch-Dashboards.git"
DASHBOARDS_SEARCH_RELEVANCE_REPO="https://github.com/opensearch-project/dashboards-search-relevance.git"

# --- Ports -------------------------------------------------------------------
OS_PORT=9200
DASHBOARDS_PORT=5601
MCP_PORT=3001
AGENT_PORT=8001

# --- Colors ------------------------------------------------------------------
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

info()  { echo -e "${CYAN}[INFO]${NC}  $*"; }
ok()    { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
err()   { echo -e "${RED}[ERROR]${NC} $*"; }

# =============================================================================
# Helpers
# =============================================================================

# Strips ANSI escape codes from input
strip_ansi() {
  sed 's/\x1b\[[0-9;]*[a-zA-Z]//g; s/\x1b[(][0-9;]*[a-zA-Z]//g; s/\r//g'
}

LOG_LINES=5

# Renders a spinner header + trailing log lines. Call once to "open" the
# display, then call again each tick — it moves the cursor back up and
# redraws.  clear_spinner_display() wipes all lines when done.
#
# Usage: render_spinner_display <first_call:0|1> <spin_char> <header> <logfile>
render_spinner_display() {
  local first=$1 ch=$2 header=$3 logfile=$4
  local term_width
  term_width=$(tput cols 2>/dev/null || echo 80)

  # Move cursor up to overwrite previous frame (skip on first call)
  if [[ "$first" -eq 0 ]]; then
    printf '\033[%dA' $((LOG_LINES + 1))
  fi

  # Spinner header
  printf '\033[2K'
  echo -e "  ${CYAN}${ch}${NC} ${header}"

  # Log tail lines
  local lines=()
  if [[ -n "$logfile" && -f "$logfile" ]]; then
    while IFS= read -r line; do
      lines+=("$line")
    done < <(tail -$LOG_LINES "$logfile" 2>/dev/null | strip_ansi)
  fi

  local i
  for ((i = 0; i < LOG_LINES; i++)); do
    printf '\033[2K'
    if [[ $i -lt ${#lines[@]} ]]; then
      echo "    ${lines[$i]:0:$((term_width - 6))}"
    else
      echo ""
    fi
  done
}

# Clears the spinner display (header + log lines)
clear_spinner_display() {
  printf '\033[%dA' $((LOG_LINES + 1))
  local i
  for ((i = 0; i <= LOG_LINES; i++)); do
    printf '\033[2K\n'
  done
  printf '\033[%dA' $((LOG_LINES + 1))
}

# Runs a command in the background with a spinner + log tail.
# Usage: run_with_spinner "label" command arg1 arg2 ...
run_with_spinner() {
  local label=$1; shift
  local tmplog
  tmplog=$(mktemp)
  local -a spin_chars=('|' '/' '-' '\\')
  local spin_i=0 elapsed=0

  "$@" > "$tmplog" 2>&1 &
  local cmd_pid=$!

  printf '\033[?25l'
  local first=1

  while kill -0 "$cmd_pid" 2>/dev/null; do
    local ch="${spin_chars[spin_i % ${#spin_chars[@]}]}"
    render_spinner_display "$first" "$ch" "${label} [${elapsed}s]" "$tmplog"
    first=0
    spin_i=$((spin_i + 1))
    sleep 1
    elapsed=$((elapsed + 1))
  done

  wait "$cmd_pid"
  local exit_code=$?

  clear_spinner_display
  printf '\033[?25h'

  if [[ $exit_code -eq 0 ]]; then
    ok "$label done (${elapsed}s)"
  else
    err "$label failed (exit code $exit_code). Last output:"
    tail -10 "$tmplog"
    rm -f "$tmplog"
    return $exit_code
  fi

  rm -f "$tmplog"
}

check_prereqs() {
  local missing=()
  command -v java  >/dev/null 2>&1 || missing+=("java (Java 21+)")
  command -v node  >/dev/null 2>&1 || missing+=("node (Node.js 20.x)")
  command -v yarn  >/dev/null 2>&1 || missing+=("yarn")
  command -v python3 >/dev/null 2>&1 || missing+=("python3 (3.12+)")
  command -v uv    >/dev/null 2>&1 || missing+=("uv (https://astral.sh/uv/install.sh)")
  command -v jq    >/dev/null 2>&1 || missing+=("jq")
  command -v curl  >/dev/null 2>&1 || missing+=("curl")

  if [[ ${#missing[@]} -gt 0 ]]; then
    err "Missing prerequisites:"
    for m in "${missing[@]}"; do
      echo "  - $m"
    done
    exit 1
  fi

  local java_ver
  java_ver=$(java -version 2>&1 | head -1 | grep -oE '[0-9]+' | head -1)
  if [[ "$java_ver" -lt 21 ]]; then
    err "Java 21+ is required (found Java $java_ver). Set JAVA_HOME to a JDK 21 installation."
    exit 1
  fi

  local node_ver
  node_ver=$(node --version 2>&1 | grep -oE '[0-9]+' | head -1)
  if [[ "$node_ver" -lt 20 ]]; then
    err "Node.js 20+ is required (found Node $node_ver). Use mise/nvm to switch versions."
    exit 1
  fi

  ok "All prerequisites met"
}

wait_for_port() {
  local port=$1 name=$2 max_wait=${3:-120} logfile=${4:-} bg_pid=${5:-}
  local elapsed=0
  local -a spin_chars=('|' '/' '-' '\\')
  local spin_i=0

  printf '\033[?25l'
  local first=1

  while ! curl -sk -o /dev/null -w '' "http://localhost:$port" 2>/dev/null && \
        ! curl -sk -o /dev/null -w '' "https://localhost:$port" 2>/dev/null; do
    # Fail fast if the background process has exited
    if [[ -n "$bg_pid" ]] && ! kill -0 "$bg_pid" 2>/dev/null; then
      [[ "$first" -eq 0 ]] && clear_spinner_display
      printf '\033[?25h'
      err "$name process exited unexpectedly. Check logs: $LOG_DIR/"
      return 1
    fi

    local ch="${spin_chars[spin_i % ${#spin_chars[@]}]}"
    render_spinner_display "$first" "$ch" "${name} on :${port} [${elapsed}s/${max_wait}s]" "$logfile"
    first=0
    spin_i=$((spin_i + 1))
    sleep 1
    elapsed=$((elapsed + 1))

    if [[ $elapsed -ge $max_wait ]]; then
      clear_spinner_display
      printf '\033[?25h'
      err "$name did not start within ${max_wait}s. Check logs: $LOG_DIR/"
      return 1
    fi
  done

  [[ "$first" -eq 0 ]] && clear_spinner_display
  printf '\033[?25h'
  ok "$name is ready on port $port (${elapsed}s)"
}

save_pid() {
  local name=$1 pid=$2
  mkdir -p "$PID_DIR"
  echo "$pid" > "$PID_DIR/$name.pid"
}

read_pid() {
  local name=$1
  local pidfile="$PID_DIR/$name.pid"
  if [[ -f "$pidfile" ]]; then
    cat "$pidfile"
  fi
}

stop_service() {
  local name=$1
  local pid port
  pid=$(read_pid "$name")

  # Determine the port for this service
  case $name in
    opensearch)    port=$OS_PORT ;;
    dashboards)    port=$DASHBOARDS_PORT ;;
    mcp-server)    port=$MCP_PORT ;;
    agent-server)  port=$AGENT_PORT ;;
  esac

  # Try PID-based stop first
  if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
    info "Stopping $name (PID $pid)..."
    kill "$pid" 2>/dev/null || true
    sleep 2
    if kill -0 "$pid" 2>/dev/null; then
      kill -9 "$pid" 2>/dev/null || true
    fi
  fi

  # Also kill anything still on the port (covers child processes)
  if [[ -n "$port" ]]; then
    local port_pids
    port_pids=$(lsof -ti ":$port" 2>/dev/null || true)
    if [[ -n "$port_pids" ]]; then
      echo "$port_pids" | xargs kill 2>/dev/null || true
      sleep 1
      port_pids=$(lsof -ti ":$port" 2>/dev/null || true)
      if [[ -n "$port_pids" ]]; then
        echo "$port_pids" | xargs kill -9 2>/dev/null || true
      fi
    fi
  fi

  ok "$name stopped"
  rm -f "$PID_DIR/$name.pid"
}

# =============================================================================
# Step 1: Clone search-relevance
# =============================================================================

setup_search_relevance() {
  info "=== Step 1: Clone search-relevance ==="
  local sr_dir="$WORKSPACE/search-relevance"

  if [[ -d "$sr_dir" ]]; then
    info "search-relevance already cloned"
  else
    run_with_spinner "Cloning search-relevance" \
      git clone --depth 1 "$SEARCH_RELEVANCE_REPO" "$sr_dir"
  fi
}

# =============================================================================
# Step 2: Clone OpenSearch Dashboards
# =============================================================================

clone_dashboards() {
  info "=== Step 2: Clone OpenSearch Dashboards ==="
  local osd_dir="$WORKSPACE/OpenSearch-Dashboards"
  local plugin_dir="$osd_dir/plugins/dashboards-search-relevance"

  if [[ -d "$osd_dir" ]]; then
    info "Dashboards already cloned"
  else
    run_with_spinner "Cloning OpenSearch Dashboards" \
      git clone --depth 1 "$DASHBOARDS_REPO" "$osd_dir"
  fi

  if [[ -d "$plugin_dir" ]]; then
    info "dashboards-search-relevance plugin already cloned"
  else
    mkdir -p "$osd_dir/plugins"
    run_with_spinner "Cloning dashboards-search-relevance plugin" \
      git clone --depth 1 "$DASHBOARDS_SEARCH_RELEVANCE_REPO" "$plugin_dir"
  fi
}

# =============================================================================
# Step 3: Setup OpenSearch Dashboards
# =============================================================================

setup_dashboards() {
  info "=== Step 3: Setup OpenSearch Dashboards ==="
  local osd_dir="$WORKSPACE/OpenSearch-Dashboards"

  cp "$PROJECT_ROOT/opensearch_dashboards.example.yml" "$osd_dir/config/opensearch_dashboards.yml"

  run_with_spinner "Bootstrapping Dashboards" \
    bash -c "cd '$osd_dir' && yarn osd bootstrap --single-version=loose"
}

# =============================================================================
# Step 4: Start OpenSearch
# =============================================================================

start_opensearch() {
  info "=== Step 4: Start OpenSearch ==="
  local sr_dir="$WORKSPACE/search-relevance"
  mkdir -p "$LOG_DIR"

  bash -c "cd '$sr_dir' && exec ./gradlew run --preserve-data" \
    > "$LOG_DIR/opensearch.log" 2>&1 &
  disown
  local bg_pid=$!
  save_pid "opensearch" $bg_pid

  wait_for_port $OS_PORT "OpenSearch" 300 "$LOG_DIR/opensearch.log" "$bg_pid"
}

# =============================================================================
# Step 5: Start MCP Server
# =============================================================================

start_mcp_server() {
  info "=== Step 5: Start MCP Server ==="
  mkdir -p "$LOG_DIR"

  OPENSEARCH_URL="http://localhost:$OS_PORT" \
  OPENSEARCH_HEADER_AUTH=true \
    bash -c "exec uv tool run opensearch-mcp-server-py --transport stream --port $MCP_PORT" \
    > "$LOG_DIR/mcp-server.log" 2>&1 &
  disown
  local bg_pid=$!
  save_pid "mcp-server" $bg_pid

  wait_for_port $MCP_PORT "MCP Server" 60 "$LOG_DIR/mcp-server.log" "$bg_pid"
}

# =============================================================================
# Step 6: Start OpenSearch Dashboards
# =============================================================================

start_dashboards() {
  info "=== Step 6: Start OpenSearch Dashboards ==="
  local osd_dir="$WORKSPACE/OpenSearch-Dashboards"
  mkdir -p "$LOG_DIR"

  bash -c "cd '$osd_dir' && exec yarn start --no-base-path" \
    > "$LOG_DIR/dashboards.log" 2>&1 &
  disown
  local bg_pid=$!
  save_pid "dashboards" $bg_pid

  wait_for_port $DASHBOARDS_PORT "OpenSearch Dashboards" 180 "$LOG_DIR/dashboards.log" "$bg_pid"
  warn "First launch compiles optimizer bundles (~5 min). May show errors in browser until complete."
}

# =============================================================================
# Step 7: Start Agent Server
# =============================================================================

start_agent_server() {
  info "=== Step 7: Start Agent Server ==="
  mkdir -p "$LOG_DIR"

  # Set up venv if not present
  if [[ ! -d "$PROJECT_ROOT/.venv" ]]; then
    run_with_spinner "Setting up Python virtual environment" \
      bash -c "cd '$PROJECT_ROOT' && uv venv && uv pip install -e '.[dev]'"
  fi

  bash -c "cd '$PROJECT_ROOT' && source .venv/bin/activate && exec python run_server.py" \
    > "$LOG_DIR/agent-server.log" 2>&1 &
  disown
  local bg_pid=$!
  save_pid "agent-server" $bg_pid

  wait_for_port $AGENT_PORT "Agent Server" 30 "$LOG_DIR/agent-server.log" "$bg_pid"
}

# =============================================================================
# Step 8: Configure workspace
# =============================================================================

setup_workspace() {
  info "=== Step 8: Configure workspace ==="
  local osd_url="http://localhost:$DASHBOARDS_PORT"

  # Create a data source pointing to local OpenSearch
  info "Creating local OpenSearch data source..."
  local ds_response
  ds_response=$(curl -s -X POST "$osd_url/api/saved_objects/data-source" \
    -H "Content-Type: application/json" \
    -H "osd-xsrf: true" \
    -d '{
      "attributes": {
        "title": "Local OpenSearch",
        "endpoint": "http://localhost:'"$OS_PORT"'",
        "dataSourceVersion": "",
        "auth": {
          "type": "no_auth",
          "credentials": {}
        }
      }
    }')

  local ds_id
  ds_id=$(echo "$ds_response" | jq -r '.id // empty')
  if [[ -z "$ds_id" ]]; then
    warn "Failed to create data source: $ds_response"
    return 1
  fi
  ok "Data source created (id: $ds_id)"

  # Create a workspace with the data source
  info "Creating Search workspace..."
  local ws_response
  ws_response=$(curl -s -X POST "$osd_url/api/workspaces" \
    -H "Content-Type: application/json" \
    -H "osd-xsrf: true" \
    -d '{
      "attributes": {
        "name": "Search",
        "description": "Search relevance workspace",
        "features": ["use-case-search"]
      },
      "settings": {
        "dataSources": ["'"$ds_id"'"]
      }
    }')

  local ws_id
  ws_id=$(echo "$ws_response" | jq -r '.result.id // empty')
  if [[ -z "$ws_id" ]]; then
    warn "Failed to create workspace: $ws_response"
    return 1
  fi
  ok "Workspace created (id: $ws_id)"
}

# =============================================================================
# Step 9: Load demo data
# =============================================================================

setup_demo_data() {
  info "=== Step 9: Load demo data ==="
  local sr_dir="$WORKSPACE/search-relevance"

  local scripts_dir="$sr_dir/src/test/scripts"
  if [[ ! -f "$scripts_dir/demo.sh" ]]; then
    err "demo.sh not found at $scripts_dir/demo.sh"
    return 1
  fi

  run_with_spinner "Loading demo data" \
    bash -c "cd '$scripts_dir' && bash demo.sh"
}

# =============================================================================
# Step 10: Verify services
# =============================================================================

run_smoke_test() {
  info "=== Step 10: Verify services ==="

  # Test OpenSearch
  local os_status
  os_status=$(curl -s -o /dev/null -w "%{http_code}" "http://localhost:$OS_PORT")
  if [[ "$os_status" == "200" ]]; then
    ok "OpenSearch          :$OS_PORT  HTTP $os_status"
  else
    err "OpenSearch          :$OS_PORT  HTTP $os_status"
  fi

  # Test Dashboards
  local osd_status
  osd_status=$(curl -s -o /dev/null -w "%{http_code}" "http://localhost:$DASHBOARDS_PORT")
  if [[ "$osd_status" == "200" || "$osd_status" == "302" ]]; then
    ok "Dashboards          :$DASHBOARDS_PORT  HTTP $osd_status"
  else
    err "Dashboards          :$DASHBOARDS_PORT  HTTP $osd_status"
  fi

  # Test MCP Server
  local mcp_status
  mcp_status=$(curl -s -o /dev/null -w "%{http_code}" "http://localhost:$MCP_PORT/mcp")
  if [[ "$mcp_status" == "307" || "$mcp_status" == "200" ]]; then
    ok "MCP Server          :$MCP_PORT  HTTP $mcp_status"
  else
    err "MCP Server          :$MCP_PORT  HTTP $mcp_status"
  fi

  # Test Agent Server
  local agent_status
  agent_status=$(curl -s -o /dev/null -w "%{http_code}" "http://localhost:$AGENT_PORT/health")
  if [[ "$agent_status" == "200" ]]; then
    ok "Agent Server        :$AGENT_PORT  HTTP $agent_status"
  else
    err "Agent Server        :$AGENT_PORT  HTTP $agent_status"
  fi

  # Test agent run
  info "Sending test query to agent..."
  local response
  response=$(curl -s -N -X POST "http://localhost:$AGENT_PORT/runs" \
    -H "Content-Type: application/json" \
    -d '{
      "threadId": "quickstart-test",
      "runId": "quickstart-run-1",
      "state": {},
      "messages": [{"id": "msg-1", "role": "user", "content": "list all indices"}]
    }' 2>&1 | grep -c "TOOL_CALL_RESULT" || true)

  if [[ "$response" -gt 0 ]]; then
    ok "Agent run succeeded (received tool call results)"
  else
    warn "Agent run did not return tool call results — check agent-server logs"
  fi

  echo ""
  info "All services running. Open http://localhost:$DASHBOARDS_PORT in your browser."
}

# =============================================================================
# Commands: --stop, --status, --start, --clean
# =============================================================================

do_stop() {
  info "Stopping all services..."
  stop_service "agent-server"
  stop_service "mcp-server"
  stop_service "dashboards"
  stop_service "opensearch"
  ok "All services stopped"
}

do_status() {
  echo ""
  info "Service status:"
  echo "  -----------------------------------------------------------"
  for svc in opensearch dashboards mcp-server agent-server; do
    local pid port name
    pid=$(read_pid "$svc")
    case $svc in
      opensearch)    port=$OS_PORT;         name="OpenSearch" ;;
      dashboards)    port=$DASHBOARDS_PORT; name="Dashboards" ;;
      mcp-server)    port=$MCP_PORT;        name="MCP Server" ;;
      agent-server)  port=$AGENT_PORT;      name="Agent Server" ;;
    esac

    local pid_alive=false port_open=false
    [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null && pid_alive=true
    (curl -sk -o /dev/null -w '' "http://localhost:$port" 2>/dev/null || \
     curl -sk -o /dev/null -w '' "https://localhost:$port" 2>/dev/null) && port_open=true

    if $pid_alive && $port_open; then
      echo -e "  ${GREEN}RUNNING${NC}  $name (PID $pid, port $port)"
    elif $port_open; then
      echo -e "  ${YELLOW}RUNNING${NC}  $name (port $port open, stale PID file)"
    elif $pid_alive; then
      echo -e "  ${YELLOW}STARTING${NC} $name (PID $pid, port $port not ready)"
    else
      echo -e "  ${RED}STOPPED${NC}  $name (port $port)"
    fi
  done
  echo "  -----------------------------------------------------------"
  echo ""
}

do_clean() {
  if [[ ! -d "$WORKSPACE" ]]; then
    info "Nothing to clean — $WORKSPACE does not exist"
    return
  fi

  # Stop services first if any are running
  do_stop

  info "Removing $WORKSPACE..."
  rm -rf "$WORKSPACE"
  ok "Cleaned up all cloned repos, logs, and PID files"
}

do_start() {
  if [[ ! -d "$WORKSPACE/search-relevance" || ! -d "$WORKSPACE/OpenSearch-Dashboards" ]]; then
    err "Repos not set up yet. Run without --start first for full setup."
    exit 1
  fi
  info "Starting services..."
  start_opensearch
  start_mcp_server
  start_dashboards
  start_agent_server
  run_smoke_test
}

do_full_setup() {
  info "=========================================="
  info " OpenSearch Agent Server — Full Quickstart"
  info "=========================================="
  echo ""

  check_prereqs

  mkdir -p "$WORKSPACE"

  setup_search_relevance    # Step 1
  clone_dashboards          # Step 2
  setup_dashboards          # Step 3
  start_opensearch          # Step 4
  start_mcp_server          # Step 5
  start_dashboards          # Step 6
  start_agent_server        # Step 7
  setup_workspace           # Step 8
  setup_demo_data           # Step 9
  run_smoke_test            # Step 10
}

# =============================================================================
# Main
# =============================================================================

case "${1:-}" in
  --stop)    do_stop ;;
  --status)  do_status ;;
  --start)   do_start ;;
  --clean)   do_clean ;;
  --help|-h)
    echo "Usage: $0 [--start|--stop|--status|--clean|--help]"
    echo ""
    echo "  (no args)   Full setup: clone, build, start all services, load demo data"
    echo "  --start     Start services only (skip clone/build)"
    echo "  --stop      Stop all running services"
    echo "  --status    Check which services are running"
    echo "  --clean     Stop services and remove all cloned repos/data"
    ;;
  "")        do_full_setup ;;
  *)         err "Unknown option: $1. Use --help for usage."; exit 1 ;;
esac
