#!/bin/bash
# =============================================================================
# RAG Templates Framework - Health Check Script
# =============================================================================
# This script verifies that all services are running correctly and provides
# detailed health status information for troubleshooting
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

# Health check results
declare -A HEALTH_STATUS
declare -A HEALTH_DETAILS

# Function to print colored output
print_message() {
    local color=$1
    local message=$2
    if [[ "$JSON_OUTPUT" != true ]]; then
        echo -e "${color}[HEALTH]${NC} ${message}"
    fi
}

# Function to log messages
log_message() {
    local message=$1
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[$timestamp] $message" >> "$LOG_FILE"
}

# Function to check Docker service status
check_docker_service() {
    local service=$1
    local container_name="${service}"
    
    print_message "$BLUE" "Checking Docker service: $service"
    
    # Check if container exists and is running
    if docker-compose -f "$COMPOSE_FILE" ps "$service" | grep -q "Up" 2>/dev/null; then
        HEALTH_STATUS["$service"]="running"
        
        # Get container details
        local container_id=$(docker-compose -f "$COMPOSE_FILE" ps -q "$service")
        local uptime=$(docker inspect --format='{{.State.StartedAt}}' "$container_id" 2>/dev/null | xargs date -d)
        local memory_usage=$(docker stats --no-stream --format "{{.MemUsage}}" "$container_id" 2>/dev/null)
        local cpu_usage=$(docker stats --no-stream --format "{{.CPUPerc}}" "$container_id" 2>/dev/null)
        
        HEALTH_DETAILS["$service"]="uptime=$uptime,memory=$memory_usage,cpu=$cpu_usage"
        
        # Check health status if available
        local health_status=$(docker inspect --format='{{.State.Health.Status}}' "$container_id" 2>/dev/null || echo "unknown")
        if [[ "$health_status" == "healthy" ]]; then
            HEALTH_STATUS["$service"]="healthy"
            print_message "$GREEN" "✓ $service is healthy"
        elif [[ "$health_status" == "unhealthy" ]]; then
            HEALTH_STATUS["$service"]="unhealthy"
            print_message "$RED" "✗ $service is unhealthy"
        else
            print_message "$YELLOW" "○ $service is running (no health check)"
        fi
    else
        HEALTH_STATUS["$service"]="stopped"
        HEALTH_DETAILS["$service"]="container not running"
        print_message "$RED" "✗ $service is not running"
    fi
}

# Function to check HTTP endpoint
check_http_endpoint() {
    local service=$1
    local url=$2
    local expected_status=${3:-200}
    
    print_message "$BLUE" "Checking HTTP endpoint: $service ($url)"
    
    local response_code=$(curl -s -o /dev/null -w "%{http_code}" "$url" 2>/dev/null || echo "000")
    local response_time=$(curl -s -o /dev/null -w "%{time_total}" "$url" 2>/dev/null || echo "0")
    
    if [[ "$response_code" == "$expected_status" ]]; then
        HEALTH_STATUS["${service}_http"]="healthy"
        HEALTH_DETAILS["${service}_http"]="status=$response_code,response_time=${response_time}s"
        print_message "$GREEN" "✓ $service HTTP endpoint is responding"
    else
        HEALTH_STATUS["${service}_http"]="unhealthy"
        HEALTH_DETAILS["${service}_http"]="status=$response_code,response_time=${response_time}s"
        print_message "$RED" "✗ $service HTTP endpoint failed (status: $response_code)"
    fi
}

# Function to check database connectivity
check_database_connectivity() {
    print_message "$BLUE" "Checking IRIS database connectivity"
    
    # Test database connection
    local db_test=$(docker-compose -f "$COMPOSE_FILE" exec -T iris_db iris session iris -U%SYS << 'EOF' 2>/dev/null || echo "ERROR"
write "DB_CONNECTION_OK"
halt
EOF
)
    
    if echo "$db_test" | grep -q "DB_CONNECTION_OK"; then
        HEALTH_STATUS["database"]="healthy"
        HEALTH_DETAILS["database"]="connection successful"
        print_message "$GREEN" "✓ Database connectivity OK"
        
        # Check if RAG tables exist
        local table_check=$(docker-compose -f "$COMPOSE_FILE" exec -T iris_db iris session iris -U%SYS << 'EOF' 2>/dev/null || echo "0"
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
        HEALTH_STATUS["database"]="unhealthy"
        HEALTH_DETAILS["database"]="connection failed"
        print_message "$RED" "✗ Database connectivity failed"
    fi
}

# Function to check Redis connectivity
check_redis_connectivity() {
    print_message "$BLUE" "Checking Redis connectivity"
    
    local redis_test=$(docker-compose -f "$COMPOSE_FILE" exec -T redis redis-cli ping 2>/dev/null || echo "ERROR")
    
    if echo "$redis_test" | grep -q "PONG"; then
        HEALTH_STATUS["redis_conn"]="healthy"
        
        # Get Redis info
        local redis_info=$(docker-compose -f "$COMPOSE_FILE" exec -T redis redis-cli info memory 2>/dev/null | grep "used_memory_human" | cut -d: -f2 | tr -d '\r\n')
        HEALTH_DETAILS["redis_conn"]="ping successful,memory_used=$redis_info"
        print_message "$GREEN" "✓ Redis connectivity OK (memory: $redis_info)"
    else
        HEALTH_STATUS["redis_conn"]="unhealthy"
        HEALTH_DETAILS["redis_conn"]="ping failed"
        print_message "$RED" "✗ Redis connectivity failed"
    fi
}

# Function to check system resources
check_system_resources() {
    print_message "$BLUE" "Checking system resources"
    
    # Check disk space
    local disk_usage=$(df -h "$PROJECT_ROOT" | tail -n 1 | awk '{print $5}' | sed 's/%//')
    if [[ "$disk_usage" -lt 80 ]]; then
        HEALTH_STATUS["disk"]="healthy"
        print_message "$GREEN" "✓ Disk usage OK (${disk_usage}%)"
    elif [[ "$disk_usage" -lt 90 ]]; then
        HEALTH_STATUS["disk"]="warning"
        print_message "$YELLOW" "○ Disk usage warning (${disk_usage}%)"
    else
        HEALTH_STATUS["disk"]="critical"
        print_message "$RED" "✗ Disk usage critical (${disk_usage}%)"
    fi
    
    # Check memory usage
    local memory_info=$(free -m | grep "Mem:" | awk '{printf "%.1f", $3/$2 * 100}')
    if (( $(echo "$memory_info < 80" | bc -l) )); then
        HEALTH_STATUS["memory"]="healthy"
        print_message "$GREEN" "✓ Memory usage OK (${memory_info}%)"
    elif (( $(echo "$memory_info < 90" | bc -l) )); then
        HEALTH_STATUS["memory"]="warning"
        print_message "$YELLOW" "○ Memory usage warning (${memory_info}%)"
    else
        HEALTH_STATUS["memory"]="critical"
        print_message "$RED" "✗ Memory usage critical (${memory_info}%)"
    fi
    
    HEALTH_DETAILS["system"]="disk=${disk_usage}%,memory=${memory_info}%"
}

# Function to perform comprehensive health check
perform_health_check() {
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    if [[ "$JSON_OUTPUT" != true ]]; then
        print_message "$BLUE" "Starting health check at $timestamp"
        echo "==========================================="
    fi
    
    # Clear previous results
    HEALTH_STATUS=()
    HEALTH_DETAILS=()
    
    # Core services
    check_docker_service "iris_db"
    check_docker_service "redis"
    check_docker_service "rag_api"
    check_docker_service "streamlit_app"
    
    # Optional services (if running)
    if docker-compose -f "$COMPOSE_FILE" ps jupyter | grep -q "Up" 2>/dev/null; then
        check_docker_service "jupyter"
    fi
    
    if docker-compose -f "$COMPOSE_FILE" ps nginx | grep -q "Up" 2>/dev/null; then
        check_docker_service "nginx"
    fi
    
    if docker-compose -f "$COMPOSE_FILE" ps monitoring | grep -q "Up" 2>/dev/null; then
        check_docker_service "monitoring"
    fi
    
    # Connectivity checks
    check_database_connectivity
    check_redis_connectivity
    
    # HTTP endpoint checks
    check_http_endpoint "rag_api" "http://localhost:8000/health"
    check_http_endpoint "streamlit_app" "http://localhost:8501/_stcore/health"
    
    # System resource checks
    check_system_resources
    
    # Log results
    log_message "Health check completed at $timestamp"
    for service in "${!HEALTH_STATUS[@]}"; do
        log_message "Service $service: ${HEALTH_STATUS[$service]} - ${HEALTH_DETAILS[$service]:-}"
    done
}

# Function to output results
output_results() {
    if [[ "$JSON_OUTPUT" == true ]]; then
        # JSON output
        echo "{"
        echo "  \"timestamp\": \"$(date -Iseconds)\","
        echo "  \"overall_status\": \"$(get_overall_status)\","
        echo "  \"services\": {"
        
        local first=true
        for service in "${!HEALTH_STATUS[@]}"; do
            if [[ "$first" != true ]]; then
                echo ","
            fi
            echo -n "    \"$service\": {"
            echo -n "\"status\": \"${HEALTH_STATUS[$service]}\""
            if [[ -n "${HEALTH_DETAILS[$service]:-}" ]]; then
                echo -n ", \"details\": \"${HEALTH_DETAILS[$service]}\""
            fi
            echo -n "}"
            first=false
        done
        echo
        echo "  }"
        echo "}"
    else
        # Human-readable output
        echo "==========================================="
        print_message "$BLUE" "Health Check Summary"
        echo "==========================================="
        
        local healthy=0
        local total=0
        
        for service in "${!HEALTH_STATUS[@]}"; do
            local status="${HEALTH_STATUS[$service]}"
            local details="${HEALTH_DETAILS[$service]:-}"
            
            case "$status" in
                "healthy")
                    echo -e "  ${GREEN}✓${NC} $service: $status"
                    ((healthy++))
                    ;;
                "running")
                    echo -e "  ${YELLOW}○${NC} $service: $status"
                    ;;
                "warning")
                    echo -e "  ${YELLOW}⚠${NC} $service: $status"
                    ;;
                *)
                    echo -e "  ${RED}✗${NC} $service: $status"
                    ;;
            esac
            
            if [[ "$VERBOSE" == true && -n "$details" ]]; then
                echo "    Details: $details"
            fi
            ((total++))
        done
        
        echo "==========================================="
        local overall=$(get_overall_status)
        case "$overall" in
            "healthy")
                print_message "$GREEN" "Overall Status: $overall ($healthy/$total services healthy)"
                ;;
            "degraded")
                print_message "$YELLOW" "Overall Status: $overall ($healthy/$total services healthy)"
                ;;
            *)
                print_message "$RED" "Overall Status: $overall ($healthy/$total services healthy)"
                ;;
        esac
    fi
}

# Function to get overall status
get_overall_status() {
    local unhealthy=0
    local total=0
    
    for status in "${HEALTH_STATUS[@]}"; do
        if [[ "$status" != "healthy" && "$status" != "running" ]]; then
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

# Function to show usage
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

EXAMPLES:
    $0                      # Basic health check
    $0 --verbose            # Detailed health check
    $0 --json               # JSON output for monitoring
    $0 --continuous         # Monitor continuously

EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -j|--json)
            JSON_OUTPUT=true
            shift
            ;;
        -c|--continuous)
            CONTINUOUS=true
            shift
            ;;
        -i|--interval)
            INTERVAL="$2"
            shift 2
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        *)
            print_message "$RED" "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Main execution
main() {
    # Change to project root
    cd "$PROJECT_ROOT"
    
    # Create log directory
    mkdir -p "$(dirname "$LOG_FILE")"
    
    if [[ "$CONTINUOUS" == true ]]; then
        # Continuous monitoring
        while true; do
            perform_health_check
            output_results
            
            if [[ "$JSON_OUTPUT" != true ]]; then
                print_message "$BLUE" "Next check in ${INTERVAL}s... (Ctrl+C to stop)"
            fi
            
            sleep "$INTERVAL"
        done
    else
        # Single check
        perform_health_check
        output_results
        
        # Exit with appropriate code
        local overall=$(get_overall_status)
        case "$overall" in
            "healthy")
                exit 0
                ;;
            "degraded")
                exit 1
                ;;
            *)
                exit 2
                ;;
        esac
    fi
}

# Error handling
trap 'print_message "$RED" "Health check script failed"' ERR

# Run main function
main "$@"