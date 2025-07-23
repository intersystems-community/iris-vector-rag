#!/usr/bin/env python3
"""
Monitor Index Performance Improvements

This script monitors ingestion performance in real-time to validate
that the new indexes are providing the expected performance improvements.
"""

import time
import sys
import os
from datetime import datetime
import json

# Add the project root to the path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from common.iris_connector import get_iris_connection

class PerformanceMonitor:
    def __init__(self):
        self.start_time = time.time()
        self.last_check_time = self.start_time
        self.last_doc_count = 0
        self.last_token_count = 0
        self.performance_log = []
        
    def check_current_performance(self):
        """Check current ingestion performance and compare to baseline."""
        try:
            conn = get_iris_connection()
            if not conn:
                print("❌ Failed to connect to database")
                return None
            
            cursor = conn.cursor()
            
            # Get current counts
            cursor.execute("SELECT COUNT(*) FROM RAG.SourceDocuments")
            current_doc_count = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM RAG.DocumentTokenEmbeddings")
            current_token_count = cursor.fetchone()[0]
            
            current_time = time.time()
            time_elapsed = current_time - self.last_check_time
            
            # Calculate rates
            docs_added = current_doc_count - self.last_doc_count
            tokens_added = current_token_count - self.last_token_count
            
            docs_per_sec = docs_added / time_elapsed if time_elapsed > 0 else 0
            tokens_per_sec = tokens_added / time_elapsed if time_elapsed > 0 else 0
            
            # Store performance data
            performance_data = {
                "timestamp": datetime.now().isoformat(),
                "total_docs": current_doc_count,
                "total_tokens": current_token_count,
                "docs_added": docs_added,
                "tokens_added": tokens_added,
                "time_elapsed": time_elapsed,
                "docs_per_sec": docs_per_sec,
                "tokens_per_sec": tokens_per_sec,
                "avg_tokens_per_doc": current_token_count / current_doc_count if current_doc_count > 0 else 0
            }
            
            self.performance_log.append(performance_data)
            
            # Update tracking variables
            self.last_check_time = current_time
            self.last_doc_count = current_doc_count
            self.last_token_count = current_token_count
            
            cursor.close()
            conn.close()
            
            return performance_data
            
        except Exception as e:
            print(f"❌ Error checking performance: {e}")
            return None
    
    def display_performance_update(self, data):
        """Display a formatted performance update."""
        if not data:
            return
            
        print(f"\n📊 PERFORMANCE UPDATE - {datetime.now().strftime('%H:%M:%S')}")
        print("=" * 50)
        print(f"📈 Current Status:")
        print(f"   Documents: {data['total_docs']:,} (+{data['docs_added']} in {data['time_elapsed']:.1f}s)")
        print(f"   Tokens: {data['total_tokens']:,} (+{data['tokens_added']} in {data['time_elapsed']:.1f}s)")
        print(f"   Avg tokens/doc: {data['avg_tokens_per_doc']:.1f}")
        
        print(f"\n⚡ Current Rates:")
        print(f"   Documents: {data['docs_per_sec']:.2f} docs/sec")
        print(f"   Tokens: {data['tokens_per_sec']:.1f} tokens/sec")
        
        # Performance assessment
        if data['docs_per_sec'] >= 20:
            print("   ✅ EXCELLENT performance - indexes working great!")
        elif data['docs_per_sec'] >= 15:
            print("   ✅ GOOD performance - significant improvement!")
        elif data['docs_per_sec'] >= 10:
            print("   ⚠️  MODERATE performance - some improvement")
        else:
            print("   ❌ POOR performance - may need additional optimization")
        
        # Estimate completion time
        remaining_docs = 100000 - data['total_docs']
        if data['docs_per_sec'] > 0 and remaining_docs > 0:
            estimated_hours = (remaining_docs / data['docs_per_sec']) / 3600
            print(f"\n🎯 Estimated completion: {estimated_hours:.1f} hours")
            
            if estimated_hours <= 3:
                print("   ✅ Excellent completion time!")
            elif estimated_hours <= 6:
                print("   ✅ Good completion time")
            elif estimated_hours <= 12:
                print("   ⚠️  Moderate completion time")
            else:
                print("   ❌ Long completion time - consider further optimization")
    
    def analyze_performance_trend(self):
        """Analyze performance trends over time."""
        if len(self.performance_log) < 3:
            return
            
        print(f"\n📈 PERFORMANCE TREND ANALYSIS")
        print("=" * 35)
        
        # Get recent performance data
        recent_data = self.performance_log[-3:]
        rates = [d['docs_per_sec'] for d in recent_data if d['docs_per_sec'] > 0]
        
        if len(rates) >= 2:
            trend = "improving" if rates[-1] > rates[0] else "declining" if rates[-1] < rates[0] else "stable"
            avg_rate = sum(rates) / len(rates)
            
            print(f"   Recent average rate: {avg_rate:.2f} docs/sec")
            print(f"   Trend: {trend}")
            
            # Compare to baseline (pre-index performance was ~15 docs/sec declining to much lower)
            baseline_rate = 15.0
            improvement = ((avg_rate - baseline_rate) / baseline_rate) * 100
            
            print(f"   Improvement vs baseline: {improvement:+.1f}%")
            
            if improvement >= 30:
                print("   🚀 MAJOR improvement - indexes working excellently!")
            elif improvement >= 10:
                print("   ✅ GOOD improvement - indexes helping significantly")
            elif improvement >= 0:
                print("   ⚠️  MINOR improvement - indexes helping somewhat")
            else:
                print("   ❌ NO improvement - may need additional optimization")
    
    def save_performance_log(self):
        """Save performance log to file for analysis."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"performance_monitoring_{timestamp}.json"
        
        try:
            with open(filename, 'w') as f:
                json.dump(self.performance_log, f, indent=2)
            print(f"\n💾 Performance log saved to: {filename}")
        except Exception as e:
            print(f"❌ Error saving performance log: {e}")

def main():
    """Main monitoring function."""
    print("🚀 INDEX PERFORMANCE IMPROVEMENT MONITORING")
    print("=" * 50)
    print(f"⏰ Monitoring started at: {datetime.now()}")
    print("\nThis script will monitor ingestion performance every 30 seconds.")
    print("Press Ctrl+C to stop monitoring.\n")
    
    monitor = PerformanceMonitor()
    
    try:
        # Initial baseline check
        print("📊 Getting initial baseline...")
        initial_data = monitor.check_current_performance()
        if initial_data:
            monitor.display_performance_update(initial_data)
        
        # Monitor performance every 30 seconds
        while True:
            time.sleep(30)  # Wait 30 seconds between checks
            
            data = monitor.check_current_performance()
            if data:
                monitor.display_performance_update(data)
                monitor.analyze_performance_trend()
            
            # Save log every 10 checks (5 minutes)
            if len(monitor.performance_log) % 10 == 0:
                monitor.save_performance_log()
                
    except KeyboardInterrupt:
        print(f"\n\n🛑 Monitoring stopped by user")
        monitor.save_performance_log()
        
        # Final summary
        if monitor.performance_log:
            print(f"\n📊 FINAL SUMMARY:")
            print(f"   Total monitoring time: {(time.time() - monitor.start_time)/60:.1f} minutes")
            print(f"   Data points collected: {len(monitor.performance_log)}")
            
            if len(monitor.performance_log) >= 2:
                first_rate = monitor.performance_log[0]['docs_per_sec']
                last_rate = monitor.performance_log[-1]['docs_per_sec']
                
                if first_rate > 0:
                    change = ((last_rate - first_rate) / first_rate) * 100
                    print(f"   Performance change: {change:+.1f}%")
                
                avg_rate = sum(d['docs_per_sec'] for d in monitor.performance_log if d['docs_per_sec'] > 0) / len([d for d in monitor.performance_log if d['docs_per_sec'] > 0])
                print(f"   Average rate: {avg_rate:.2f} docs/sec")
        
        print(f"\n✅ Monitoring completed at: {datetime.now()}")
    
    except Exception as e:
        print(f"❌ Error during monitoring: {e}")
        monitor.save_performance_log()

if __name__ == "__main__":
    main()