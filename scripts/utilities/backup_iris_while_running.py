#!/usr/bin/env python3
"""
Backup IRIS database while ingestion is running.
Uses IRIS backup utilities that work with active databases.
"""

import subprocess
import datetime
import json
import os
from pathlib import Path

def create_backup_directory():
    """Create timestamped backup directory."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = Path(f"backups/iris_backup_{timestamp}")
    backup_dir.mkdir(parents=True, exist_ok=True)
    return backup_dir

def backup_iris_database(backup_dir):
    """Create IRIS database backup using built-in backup utilities."""
    print(f"🔄 Starting IRIS database backup to {backup_dir}")
    
    try:
        # Create backup using IRIS backup utility
        backup_file = backup_dir / "iris_database.cbk"
        
        # Use IRIS backup command that works with running database
        backup_cmd = [
            'docker', 'exec', 'iris_db_rag_standalone',
            'iris', 'session', 'iris', '-U', '%SYS',
            f'&sql("BACKUP DATABASE TO DEVICE \\"{backup_file}\\" USING %SYSTEM.Backup")'
        ]
        
        print("📦 Creating database backup...")
        result = subprocess.run(backup_cmd, capture_output=True, text=True, check=True)
        
        if result.returncode == 0:
            print(f"✅ Database backup completed: {backup_file}")
            return True
        else:
            print(f"❌ Backup failed: {result.stderr}")
            return False
            
    except subprocess.CalledProcessError as e:
        print(f"❌ Backup command failed: {e}")
        return False

def backup_docker_volume(backup_dir):
    """Backup the Docker volume using tar."""
    print("🔄 Creating Docker volume backup...")
    
    try:
        # Create a tar backup of the volume
        volume_backup = backup_dir / "iris_volume_backup.tar.gz"
        
        # Use docker run with volume mounted to create backup
        backup_cmd = [
            'docker', 'run', '--rm',
            '-v', 'rag-templates_iris_db_data:/source:ro',
            '-v', f'{backup_dir.absolute()}:/backup',
            'alpine:latest',
            'tar', 'czf', '/backup/iris_volume_backup.tar.gz', '-C', '/source', '.'
        ]
        
        result = subprocess.run(backup_cmd, capture_output=True, text=True, check=True)
        
        if result.returncode == 0:
            print(f"✅ Volume backup completed: {volume_backup}")
            return True
        else:
            print(f"❌ Volume backup failed: {result.stderr}")
            return False
            
    except subprocess.CalledProcessError as e:
        print(f"❌ Volume backup command failed: {e}")
        return False

def get_backup_metadata():
    """Collect metadata about the current state."""
    metadata = {
        "timestamp": datetime.datetime.now().isoformat(),
        "database_size": None,
        "container_status": None,
        "volume_info": None
    }
    
    try:
        # Get database size
        size_result = subprocess.run([
            'docker', 'exec', 'iris_db_rag_standalone',
            'du', '-sh', '/usr/irissys/mgr/user/'
        ], capture_output=True, text=True)
        
        if size_result.returncode == 0:
            metadata["database_size"] = size_result.stdout.strip()
    except:
        pass
    
    try:
        # Get container status
        status_result = subprocess.run([
            'docker', 'inspect', 'iris_db_rag_standalone'
        ], capture_output=True, text=True)
        
        if status_result.returncode == 0:
            container_info = json.loads(status_result.stdout)[0]
            metadata["container_status"] = {
                "state": container_info["State"]["Status"],
                "started_at": container_info["State"]["StartedAt"],
                "image": container_info["Config"]["Image"]
            }
    except:
        pass
    
    try:
        # Get volume info
        volume_result = subprocess.run([
            'docker', 'volume', 'inspect', 'rag-templates_iris_db_data'
        ], capture_output=True, text=True)
        
        if volume_result.returncode == 0:
            volume_info = json.loads(volume_result.stdout)[0]
            metadata["volume_info"] = {
                "mountpoint": volume_info["Mountpoint"],
                "created": volume_info["CreatedAt"]
            }
    except:
        pass
    
    return metadata

def create_backup():
    """Create a complete backup of the IRIS system."""
    print("🚀 Starting IRIS backup while ingestion is running")
    print("=" * 60)
    
    # Create backup directory
    backup_dir = create_backup_directory()
    print(f"📁 Backup directory: {backup_dir}")
    
    # Collect metadata
    print("📊 Collecting system metadata...")
    metadata = get_backup_metadata()
    
    # Save metadata
    metadata_file = backup_dir / "backup_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"✅ Metadata saved: {metadata_file}")
    
    # Create volume backup (safer for running system)
    volume_success = backup_docker_volume(backup_dir)
    
    # Try database backup (may not work if system is very busy)
    # db_success = backup_iris_database(backup_dir)
    
    # Create summary
    summary = {
        "backup_completed": datetime.datetime.now().isoformat(),
        "backup_directory": str(backup_dir),
        "volume_backup_success": volume_success,
        # "database_backup_success": db_success,
        "database_size_at_backup": metadata.get("database_size", "Unknown")
    }
    
    summary_file = backup_dir / "backup_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "=" * 60)
    print("📋 BACKUP SUMMARY")
    print("=" * 60)
    print(f"📁 Backup location: {backup_dir}")
    print(f"💾 Database size: {metadata.get('database_size', 'Unknown')}")
    print(f"📦 Volume backup: {'✅ Success' if volume_success else '❌ Failed'}")
    # print(f"🗄️  Database backup: {'✅ Success' if db_success else '❌ Failed'}")
    print(f"⏰ Completed at: {datetime.datetime.now()}")
    
    return backup_dir

if __name__ == "__main__":
    backup_dir = create_backup()
    print(f"\n🎉 Backup completed! Files saved to: {backup_dir}")