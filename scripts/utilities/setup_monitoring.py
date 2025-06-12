import sys
import logging
import os
import json
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def setup_monitoring():
    """Set up monitoring configuration and log directories"""
    logging.info("Setting up monitoring infrastructure...")
    
    # Create monitoring directories
    directories = [
        "logs/performance",
        "logs/ingestion", 
        "logs/benchmarks",
        "logs/health_checks",
        "logs/errors"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logging.info(f"Created directory: {directory}")
    
    # Create monitoring configuration
    monitoring_config = {
        "created_at": datetime.now().isoformat(),
        "log_retention_days": 30,
        "performance_thresholds": {
            "vector_query_max_ms": 100,
            "ingestion_rate_min_docs_per_sec": 10,
            "memory_usage_max_percent": 85,
            "disk_usage_max_percent": 90
        },
        "alert_settings": {
            "enable_alerts": True,
            "alert_log_file": "logs/alerts.log",
            "critical_threshold_breaches": 3
        },
        "health_check_schedule": {
            "interval_minutes": 15,
            "full_check_interval_hours": 6
        }
    }
    
    config_file = "config/monitoring.json"
    os.makedirs(os.path.dirname(config_file), exist_ok=True)
    
    with open(config_file, 'w') as f:
        json.dump(monitoring_config, f, indent=2)
    
    logging.info(f"Monitoring configuration saved to {config_file}")
    
    # Create initial log files
    log_files = [
        "logs/system.log",
        "logs/performance/vector_queries.log",
        "logs/ingestion/progress.log",
        "logs/health_checks/status.log"
    ]
    
    for log_file in log_files:
        if not os.path.exists(log_file):
            with open(log_file, 'w') as f:
                f.write(f"# Log file created at {datetime.now().isoformat()}\n")
            logging.info(f"Created log file: {log_file}")
    
    # Create monitoring scripts
    create_monitoring_scripts()
    
    logging.info("✅ Monitoring setup completed successfully!")
    return True

def create_monitoring_scripts():
    """Create basic monitoring scripts"""
    
    # Simple performance monitor script
    perf_monitor_script = """#!/bin/bash
# Simple performance monitoring script
# Run this periodically to log system performance

echo "$(date): Running performance check..." >> logs/performance/system_performance.log

# Check memory usage
free -h >> logs/performance/system_performance.log

# Check disk usage  
df -h >> logs/performance/system_performance.log

# Check IRIS container status
docker stats --no-stream iris_db_rag_licensed >> logs/performance/system_performance.log 2>/dev/null || echo "IRIS container not found" >> logs/performance/system_performance.log

echo "---" >> logs/performance/system_performance.log
"""
    
    with open("scripts/monitor_performance.sh", 'w') as f:
        f.write(perf_monitor_script)
    
    os.chmod("scripts/monitor_performance.sh", 0o755)
    logging.info("Created scripts/monitor_performance.sh")
    
    # Log rotation script
    log_rotation_script = """#!/bin/bash
# Log rotation script to prevent logs from growing too large

LOG_DIR="logs"
MAX_SIZE="100M"

find $LOG_DIR -name "*.log" -size +$MAX_SIZE -exec gzip {} \;
find $LOG_DIR -name "*.log.gz" -mtime +30 -delete

echo "$(date): Log rotation completed" >> logs/system.log
"""
    
    with open("scripts/rotate_logs.sh", 'w') as f:
        f.write(log_rotation_script)
    
    os.chmod("scripts/rotate_logs.sh", 0o755)
    logging.info("Created scripts/rotate_logs.sh")

if __name__ == "__main__":
    success = setup_monitoring()
    sys.exit(0 if success else 1)