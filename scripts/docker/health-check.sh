#!/bin/bash
# =============================================================================
# RAG Templates Framework - Health Check Script (macOS-compatible, no Redis)
# =============================================================================
# Verifies that essential services are running correctly and provides
# health status. Designed to work on macOS default bash (v3.2).
#
# Changes:
# - Removed Redis checks (no Redis dependency)
# - System resource checks (disk/memory) are informational and do not block readiness
# - Optional HTTP checks can be skipped with SKIP_HTTP=1
# =============================================================================

set -euo pipefail

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
COMPOSE_FILE="${PROJECT_ROOT}/docker-compose.full.yml"
LOG_FILE="${PROJECT_ROOT}/logs/health-check.log"

# Default options
VERBOSE=false
JSON_OUTPUT=false
CONTINUOUS=false
INTERVAL=30

# Storage (portable replacement for associative arrays)
SERVICES=""

sanitize_key() {
  echo "$1" | tr '[:lower:]' '[:upper:]' | sed -E 's/[^A-Z0-9_]+/_/g'
}

has_service() {
  local s="$1"
  case " $SERVICES " in *" $s "*) return 0 ;; *) return 1 ;; esac
}

add_service() {
  local s="$1"
  if ! has_service "$s"; then
    if [ -z "$SERVICES" ]; then SERVICES="$s"; else SERVICES="$SERVICES $s"; fi
  fi
}

set_health() {
  # set_health <service_name> <status> [details]
  local service="$1" ; local status="$2" ; local details="${3:-}"
  local key ; key="$(sanitize_key "$service")"
  add_service "$service"
  eval "HEALTH_STATUS_${key}=\"\$status\""
  if [ -n "$details" ]; then
    eval "HEALTH_DETAILS_${key}=\"\$details\""
  else
    eval "unset HEALTH_DETAILS_${key} 2>/dev/null || true"
  fi
}

get_health_status()  { local key; key="$(sanitize_key "$1")"; eval "printf '%s' \"\${HEALTH_STATUS_${key}:-}\""; }
get_health_details() { local key; key="$(sanitize_key "$1")"; eval "printf '%s' \"\${HEALTH_DETAILS_${key}:-}\""; }

print_message() {
  local color=$1 ; local message=$2
  if [[ "$JSON_OUTPUT" != true ]]; then echo -e "${color}[HEALTH]${NC} ${message}"; fi
}

log_message() {
  local message=$1 ; local timestamp
  timestamp="$(date '+%Y-%m-%d %H:%M:%S')"
  echo "[$timestamp] $message" >> "$LOG_FILE"
}

check_docker_service() {
  local service=$1
  print_message "$BLUE" "Checking Docker service: $service"

  if docker-compose -f "$COMPOSE_FILE" ps "$service" | grep -q "Up" 2>/dev/null; then
    local container_id started_at memory_usage cpu_usage details health_status
    container_id="$(docker-compose -f "$COMPOSE_FILE" ps -q "$service")"
    started_at="$(docker inspect --format='{{.State.StartedAt}}' "$container_id" 2>/dev/null || true)"
    memory_usage="$(docker stats --no-stream --format "{{.MemUsage}}" "$container_id" 2>/dev/null || true)"
    cpu_usage="$(docker stats --no-stream --format "{{.CPUPerc}}" "$container_id" 2>/dev/null || true)"
    details="uptime=$started_at,memory=$memory_usage,cpu=$cpu_usage"
    health_status="$(docker inspect --format='{{.State.Health.Status}}' "$container_id" 2>/dev/null || echo "unknown")"

    if [[ "$health_status" == "healthy" ]]; then
      set_health "$service" "healthy" "$details"; print_message "$GREEN" "✓ $service is healthy"
    elif [[ "$health_status" == "unhealthy" ]]; then
      set_health "$service" "unhealthy" "$details"; print_message "$RED" "✗ $service is unhealthy"
    else
      set_health "$service" "running" "$details"; print_message "$YELLOW" "○ $service is running (no health check)"
    fi
  else
    set_health "$service" "stopped" "container not running"
    print_message "$RED" "✗ $service is not running"
  fi
}

check_http_endpoint() {
  local service=$1 ; local url=$2 ; local expected_status=${3:-200}
  print_message "$BLUE" "Checking HTTP endpoint: $service ($url)"

  local response_code response_time
  response_code="$(curl -s -o /dev/null -w "%{http_code}" "$url" 2>/dev/null || echo "000")"
  response_time="$(curl -s -o /dev/null -w "%{time_total}" "$url" 2>/dev/null || echo "0")"

  local svc="${service}_http"
  if [[ "$response_code" == "$expected_status" ]]; then
    set_health "$svc" "healthy" "status=$response_code,response_time=${response_time}s"
    print_message "$GREEN" "✓ $service HTTP endpoint is responding"
  else
    set_health "$svc" "unhealthy" "status=$response_code,response_time=${response_time}s"
    print_message "$RED" "✗ $service HTTP endpoint failed (status: $response_code)"
  fi
}

check_database_connectivity() {
  print_message "$BLUE" "Checking IRIS database connectivity"

  local db_test
  db_test=$(docker-compose -f "$COMPOSE_FILE" exec -T iris_db iris session iris -U%SYS << 'EOF' 2>/dev/null || echo "ERROR"
write "DB_CONNECTION_OK"
halt
EOF
)
  if echo "$db_test" | grep -q "DB_CONNECTION_OK"; then
    set_health "database" "healthy" "connection successful"
    print_message "$GREEN" "✓ Database connectivity OK"

    local table_check
    table_check=$(docker-compose -f "$COMPOSE_FILE" exec -T iris_db iris session iris -U%SYS << 'EOF' 2>/dev/null || echo "0"
set stmt = ##class(%SQL.Statement).%New()
set result = stmt.%ExecDirect("SELECT COUNT(*) FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_NAME LIKE 'rag_%'")
if result.%Next() {
    write result.%GetData(1)
} else {
    write "0"
}
halt
EOF
)
    table_check=$(echo "$table_check" | tr -d '\r\n' | sed 's/[^0-9]//g')
    if [[ -n "$table_check" && "$table_check" -gt 0 ]]; then
      print_message "$GREEN" "✓ RAG database schema exists ($table_check tables)"
    else
      print_message "$YELLOW" "○ RAG database schema not found"
    fi
  else
    set_health "database" "unhealthy" "connection failed"
    print_message "$RED" "✗ Database connectivity failed"
  fi
}

check_system_resources() {
  print_message "$BLUE" "Checking system resources"

  # Disk usage: informational only
  local disk_usage
  disk_usage="$(df -h "$PROJECT_ROOT" | tail -n 1 | awk '{print $5}' | sed 's/%//')"
  print_message "$BLUE" "Disk usage: ${disk_usage}%"
  set_health "disk" "info" "usage=${disk_usage}%"

  # Memory usage: informational only (portable calculation)
  local mem_percent="0"
  if command -v free >/dev/null 2>&1; then
    mem_percent="$(free -m | awk '/Mem:/ {printf \"%.1f\", $3/$2*100}')"
  elif command -v vm_stat >/dev/null 2>&1; then
    local free_pages active_pages inactive_pages speculative_pages wired_pages purgeable_pages used_pages total_pages
    free_pages=$(vm_stat | awk '/Pages free/ {print $3}' | tr -d '.')
    active_pages=$(vm_stat | awk '/Pages active/ {print $3}' | tr -d '.')
    inactive_pages=$(vm_stat | awk '/Pages inactive/ {print $3}' | tr -d '.')
    speculative_pages=$(vm_stat | awk '/Pages speculative/ {print $3}' | tr -d '.')
    wired_pages=$(vm_stat | awk '/Pages wired down/ {print $4}' | tr -d '.')
    purgeable_pages=$(vm_stat | awk '/Pages purgeable/ {print $3}' | tr -d '.')
    used_pages=$((active_pages + inactive_pages + speculative_pages + wired_pages - purgeable_pages))
    total_pages=$((used_pages + free_pages))
    if [[ "${total_pages:-0}" -gt 0 ]]; then
      mem_percent=$(awk -v u="${used_pages:-0}" -v t="${total_pages:-1}" 'BEGIN { printf "%.1f", (u/t)*100 }')
    fi
  fi
  print_message "$BLUE" "Memory usage: ${mem_percent}%"
  set_health "memory" "info" "usage=${mem_percent}%"

  set_health "system" "info" "disk=${disk_usage:-unknown}%,memory=${mem_percent}%"
}

perform_health_check() {
  local timestamp ; timestamp="$(date '+%Y-%m-%d %H:%M:%S')"
  if [[ "$JSON_OUTPUT" != true ]]; then
    print_message "$BLUE" "Starting health check at $timestamp"
    echo "==========================================="
  fi

  SERVICES=""

  # Essential core services only (no Redis)
  check_docker_service "iris_db"
  check_docker_service "rag_api"
  check_docker_service "streamlit_app"

  # Optional services (if running)
  if docker-compose -f "$COMPOSE_FILE" ps jupyter | grep -q "Up" 2>/dev/null; then check_docker_service "jupyter"; fi
  if docker-compose -f "$COMPOSE_FILE" ps nginx | grep -q "Up" 2>/dev/null; then check_docker_service "nginx"; fi
  if docker-compose -f "$COMPOSE_FILE" ps monitoring | grep -q "Up" 2>/dev/null; then check_docker_service "monitoring"; fi

  # Connectivity checks
  check_database_connectivity

  # HTTP endpoint checks (optional)
  if [[ "${SKIP_HTTP:-0}" != "1" ]]; then
    check_http_endpoint "rag_api" "http://localhost:8000/health"
    check_http_endpoint "streamlit_app" "http://localhost:8501/_stcore/health"
  else
    print_message "$YELLOW" "Skipping HTTP endpoint checks (SKIP_HTTP=1)"
  fi

  # System resource checks (informational)
  check_system_resources

  log_message "Health check completed at $timestamp"
  for service in $SERVICES; do
    local details ; details="$(get_health_details "$service")"
    log_message "Service $service: $(get_health_status "$service") - ${details:-}"
  done
}

get_overall_status() {
  local unhealthy=0 total=0
  for service in $SERVICES; do
    local status ; status="$(get_health_status "$service")"
    # Only count truly unhealthy/stopped services
    if [[ "$status" != "healthy" && "$status" != "running" && "$status" != "info" ]]; then
      ((unhealthy++))
    fi
    ((total++))
  done

  if [[ $unhealthy -eq 0 ]]; then
    echo "healthy"
  elif [[ $unhealthy -lt 3 ]]; then
    echo "degraded"
  else
    echo "unhealthy"
  fi
}

output_results() {
  if [[ "$JSON_OUTPUT" == true ]]; then
    echo "{"
    echo "  \"timestamp\": \"$(date -Iseconds)\","
    echo "  \"overall_status\": \"$(get_overall_status)\","
    echo "  \"services\": {"
    local first=true
    for service in $SERVICES; do
      if [[ "$first" != true ]]; then echo ","; fi
      local status details ; status="$(get_health_status "$service")" ; details="$(get_health_details "$service")"
      printf "    \"%s\": {\"status\": \"%s\"" "$service" "$status"
      if [[ -n "$details" ]]; then printf ", \"details\": \"%s\"" "$details"; fi
      printf "}"
      first=false
    done
    echo
    echo "  }"
    echo "}"
  else
    echo "==========================================="
    print_message "$BLUE" "Health Check Summary"
    echo "==========================================="

    local healthy=0 total=0
    for service in $SERVICES; do
      local status details ; status="$(get_health_status "$service")" ; details="$(get_health_details "$service")"
      case "$status" in
        "healthy") echo -e "  ${GREEN}✓${NC} $service: $status"; ((healthy++)) ;;
        "running") echo -e "  ${YELLOW}○${NC} $service: $status" ;;
        "info")    echo -e "  ${BLUE}i${NC} $service: $details" ;;
        "degraded"|"warning") echo -e "  ${YELLOW}⚠${NC} $service: $status" ;;
        *)         echo -e "  ${RED}✗${NC} $service: $status" ;;
      esac
      if [[ "$VERBOSE" == true && -n "$details" ]]; then echo "    Details: $details"; fi
      ((total++))
    done

    echo "==========================================="
    local overall ; overall="$(get_overall_status)"
    case "$overall" in
      "healthy")  print_message "$GREEN" "Overall Status: $overall ($healthy/$total services healthy)" ;;
      "degraded") print_message "$YELLOW" "Overall Status: $overall ($healthy/$total services healthy)" ;;
      *)          print_message "$RED" "Overall Status: $overall ($healthy/$total services healthy)" ;;
    esac
  fi
}

show_usage() {
  cat << EOF
Usage: $0 [OPTIONS]

Health check script for RAG Templates Framework

OPTIONS:
  -v, --verbose           Show detailed information
  -j, --json              Output results in JSON format
  -c, --continuous        Run continuously with interval
  -i, --interval SECONDS  Interval for continuous mode [default: 30]
  -h, --help              Show this help message

Env:
  SKIP_HTTP=1  Skip HTTP endpoint checks

EOF
}

while [[ $# -gt 0 ]]; do
  case $1 in
    -v|--verbose)    VERBOSE=true; shift ;;
    -j|--json)       JSON_OUTPUT=true; shift ;;
    -c|--continuous) CONTINUOUS=true; shift ;;
    -i|--interval)   INTERVAL="$2"; shift 2 ;;
    -h|--help)       show_usage; exit 0 ;;
    *)               print_message "$RED" "Unknown option: $1"; show_usage; exit 1 ;;
  esac
done

main() {
  cd "$PROJECT_ROOT"
  mkdir -p "$(dirname "$LOG_FILE")"

  if [[ "$CONTINUOUS" == true ]]; then
    while true; do
      perform_health_check
      output_results
      if [[ "$JSON_OUTPUT" != true ]]; then print_message "$BLUE" "Next check in ${INTERVAL}s... (Ctrl+C to stop)"; fi
      sleep "$INTERVAL"
    done
  else
    perform_health_check
    output_results
    local overall ; overall="$(get_overall_status)"
    case "$overall" in
      "healthy")  exit 0 ;;
      "degraded") exit 1 ;;
      *)          exit 2 ;;
    esac
  fi
}

trap 'print_message "$RED" "Health check script failed"' ERR
main "$@"
