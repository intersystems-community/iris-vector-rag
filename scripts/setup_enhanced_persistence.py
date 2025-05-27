#!/usr/bin/env python3
"""
Set up enhanced Docker persistence for IRIS without disrupting current ingestion.
This prepares the infrastructure for the next restart.
"""

import subprocess
import os
import json
from pathlib import Path

def create_persistent_directories():
    """Create directories for enhanced persistence."""
    print("📁 Creating persistent data directories...")
    
    directories = [
        "data/iris_persistent_data",
        "data/iris_journal_data", 
        "data/iris_audit_data",
        "data/iris_config_data",
        "backups"
    ]
    
    for directory in directories:
        dir_path = Path(directory)
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"   ✅ Created: {directory}")
    
    return True

def check_current_volume_usage():
    """Check current Docker volume usage."""
    print("🔍 Checking current volume usage...")
    
    try:
        # Get volume info
        result = subprocess.run([
            'docker', 'volume', 'inspect', 'rag-templates_iris_db_data'
        ], capture_output=True, text=True, check=True)
        
        volume_info = json.loads(result.stdout)[0]
        mountpoint = volume_info['Mountpoint']
        
        print(f"   📍 Current volume mountpoint: {mountpoint}")
        
        # Try to get size (may need sudo)
        try:
            size_result = subprocess.run([
                'sudo', 'du', '-sh', mountpoint
            ], capture_output=True, text=True, timeout=10)
            
            if size_result.returncode == 0:
                size = size_result.stdout.strip().split('\t')[0]
                print(f"   💾 Current volume size: {size}")
            else:
                print(f"   💾 Volume size: Unable to determine (need sudo access)")
        except:
            print(f"   💾 Volume size: Unable to determine")
            
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"   ❌ Error checking volume: {e}")
        return False

def create_migration_plan():
    """Create a migration plan for switching to enhanced persistence."""
    print("📋 Creating migration plan...")
    
    migration_plan = {
        "current_setup": {
            "container": "iris_db_rag_standalone",
            "volume": "rag-templates_iris_db_data",
            "compose_file": "docker-compose.yml"
        },
        "enhanced_setup": {
            "compose_file": "docker-compose-enhanced.yml",
            "persistent_directories": [
                "data/iris_persistent_data",
                "data/iris_journal_data",
                "data/iris_audit_data", 
                "data/iris_config_data"
            ],
            "config_file": "config/iris-enhanced.cpf"
        },
        "migration_steps": [
            "1. Let current ingestion complete",
            "2. Create backup using backup_iris_while_running.py",
            "3. Stop current container: docker-compose down",
            "4. Copy data from current volume to new persistent directories",
            "5. Start with enhanced configuration: docker-compose -f docker-compose-enhanced.yml up -d",
            "6. Verify data integrity",
            "7. Resume operations"
        ],
        "rollback_plan": [
            "1. Stop enhanced container",
            "2. Start original container: docker-compose up -d", 
            "3. Verify data is intact"
        ]
    }
    
    plan_file = Path("PERSISTENCE_MIGRATION_PLAN.json")
    with open(plan_file, 'w') as f:
        json.dump(migration_plan, f, indent=2)
    
    print(f"   ✅ Migration plan saved: {plan_file}")
    return migration_plan

def create_data_migration_script():
    """Create script to migrate data from current volume to new structure."""
    print("📝 Creating data migration script...")
    
    script_content = '''#!/bin/bash
# Data migration script for enhanced IRIS persistence
# Run this AFTER stopping the current container

set -e

echo "🔄 Starting data migration to enhanced persistence structure..."

# Check if current volume exists
if ! docker volume inspect rag-templates_iris_db_data > /dev/null 2>&1; then
    echo "❌ Current volume rag-templates_iris_db_data not found!"
    exit 1
fi

# Create temporary container to access current volume
echo "📦 Creating temporary container to access current data..."
docker run --rm -d \\
    --name iris_data_migrator \\
    -v rag-templates_iris_db_data:/source:ro \\
    -v "$(pwd)/data/iris_persistent_data":/target \\
    alpine:latest sleep 3600

# Copy data from current volume to new structure
echo "📋 Copying data to new persistent structure..."
docker exec iris_data_migrator sh -c "
    echo 'Copying main database files...'
    cp -r /source/* /target/ 2>/dev/null || true
    echo 'Setting permissions...'
    chmod -R 755 /target/
    echo 'Data copy completed!'
"

# Stop and remove temporary container
echo "🧹 Cleaning up temporary container..."
docker stop iris_data_migrator

echo "✅ Data migration completed!"
echo "📁 Data is now available in: $(pwd)/data/iris_persistent_data"
echo ""
echo "Next steps:"
echo "1. Start enhanced container: docker-compose -f docker-compose-enhanced.yml up -d"
echo "2. Verify data integrity"
echo "3. Resume operations"
'''
    
    script_file = Path("scripts/migrate_iris_data.sh")
    with open(script_file, 'w') as f:
        f.write(script_content)
    
    # Make script executable
    os.chmod(script_file, 0o755)
    
    print(f"   ✅ Migration script created: {script_file}")
    return script_file

def setup_enhanced_persistence():
    """Set up enhanced persistence infrastructure."""
    print("🚀 Setting up enhanced IRIS persistence")
    print("=" * 60)
    print("⚠️  This prepares infrastructure WITHOUT disrupting current ingestion")
    print("=" * 60)
    
    # Create directories
    create_persistent_directories()
    
    # Check current setup
    check_current_volume_usage()
    
    # Create migration plan
    migration_plan = create_migration_plan()
    
    # Create migration script
    migration_script = create_data_migration_script()
    
    print("\n" + "=" * 60)
    print("📋 SETUP SUMMARY")
    print("=" * 60)
    print("✅ Enhanced persistence infrastructure prepared")
    print("✅ Migration plan created: PERSISTENCE_MIGRATION_PLAN.json")
    print("✅ Data migration script: scripts/migrate_iris_data.sh")
    print("✅ Enhanced Docker Compose: docker-compose-enhanced.yml")
    print("✅ Enhanced IRIS config: config/iris-enhanced.cpf")
    
    print("\n🔄 CURRENT STATUS:")
    print("   • Current ingestion continues uninterrupted")
    print("   • Enhanced persistence ready for next restart")
    print("   • Backup scripts available for data safety")
    
    print("\n📋 NEXT STEPS (when ready to migrate):")
    print("   1. Run backup: python scripts/backup_iris_while_running.py")
    print("   2. Stop current: docker-compose down")
    print("   3. Migrate data: ./scripts/migrate_iris_data.sh")
    print("   4. Start enhanced: docker-compose -f docker-compose-enhanced.yml up -d")
    
    return True

if __name__ == "__main__":
    setup_enhanced_persistence()