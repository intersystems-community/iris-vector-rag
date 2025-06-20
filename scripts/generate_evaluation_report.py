#!/usr/bin/env python3
"""
Comprehensive Evaluation Report Generator
Combines RAGAS evaluation results with performance benchmarks.
"""

import os
import sys
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from common.database_schema_manager import get_schema_manager

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EvaluationReportGenerator:
    """Generate comprehensive evaluation reports."""
    
    def __init__(self):
        self.schema = get_schema_manager()
        self.timestamp = datetime.now()
    
    def load_latest_results(self) -> Dict[str, Any]:
        """Load the most recent evaluation and benchmark results."""
        results = {
            'evaluation': None,
            'benchmarks': None,
            'data_status': None
        }
        
        # Load latest evaluation results
        eval_dir = Path("eval_results")
        if eval_dir.exists():
            eval_files = list(eval_dir.glob("standardized_evaluation_*.json"))
            if eval_files:
                latest_eval = max(eval_files, key=lambda p: p.stat().st_mtime)
                with open(latest_eval, 'r') as f:
                    results['evaluation'] = json.load(f)
                logger.info(f"📊 Loaded evaluation: {latest_eval.name}")
        
        # Load latest benchmark results
        bench_dir = Path("benchmarks")
        if bench_dir.exists():
            bench_files = list(bench_dir.glob("performance_report_*.json"))
            if bench_files:
                latest_bench = max(bench_files, key=lambda p: p.stat().st_mtime)
                with open(latest_bench, 'r') as f:
                    results['benchmarks'] = json.load(f)
                logger.info(f"⚡ Loaded benchmarks: {latest_bench.name}")
        
        # Get current data status
        results['data_status'] = self._get_current_data_status()
        
        return results
    
    def _get_current_data_status(self) -> Dict[str, Any]:
        """Get current data status for the report."""
        try:
            from common.iris_connector import get_iris_connection
            connection = get_iris_connection()
            cursor = connection.cursor()
            
            # Check main tables using schema configuration
            table_configs = [
                ('source_documents', 'Main document store'),
                ('document_entities', 'GraphRAG entities'),
                ('document_token_embeddings', 'ColBERT tokens'),
                ('document_chunks', 'CRAG/NodeRAG chunks'),
                ('ifind_index', 'IFind optimization')
            ]
            
            table_status = {}
            total_docs = 0
            
            for table_key, description in table_configs:
                try:
                    table_name = self.schema.get_table_name(table_key, fully_qualified=True)
                    cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
                    count = cursor.fetchone()[0]
                    table_status[table_key] = {
                        'description': description,
                        'table_name': table_name,
                        'count': count,
                        'status': 'ready' if count > 0 else 'empty'
                    }
                    
                    if table_key == 'source_documents':
                        total_docs = count
                        
                except Exception as e:
                    table_status[table_key] = {
                        'description': description,
                        'status': 'error',
                        'error': str(e)
                    }
            
            return {
                'timestamp': datetime.now().isoformat(),
                'total_documents': total_docs,
                'table_status': table_status
            }
            
        except Exception as e:
            logger.error(f"Failed to get data status: {e}")
            return {'error': str(e)}
    
    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate a comprehensive evaluation report."""
        logger.info("📋 Generating Comprehensive Evaluation Report...")
        
        # Load all results
        results = self.load_latest_results()
        
        # Extract key metrics
        evaluation_summary = self._analyze_evaluation_results(results['evaluation'])
        performance_summary = self._analyze_performance_results(results['benchmarks'])
        data_summary = self._analyze_data_status(results['data_status'])
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            evaluation_summary, performance_summary, data_summary
        )
        
        # Create comprehensive report
        report = {
            'report_metadata': {
                'title': 'RAG Templates Comprehensive Evaluation Report',
                'generated_at': self.timestamp.isoformat(),
                'schema_version': 'config-driven',
                'report_version': '1.0'
            },
            'executive_summary': self._create_executive_summary(
                evaluation_summary, performance_summary, data_summary
            ),
            'data_infrastructure': data_summary,
            'pipeline_evaluation': evaluation_summary,
            'performance_analysis': performance_summary,
            'recommendations': recommendations,
            'technical_details': {
                'evaluation_data': results['evaluation'],
                'benchmark_data': results['benchmarks'],
                'data_status': results['data_status']
            }
        }
        
        return report
    
    def _analyze_evaluation_results(self, eval_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze evaluation results."""
        if not eval_data:
            return {'status': 'no_data', 'message': 'No evaluation data available'}
        
        summary = eval_data.get('summary', {})
        pipeline_results = eval_data.get('pipeline_results', {})
        
        # Analyze pipeline performance
        pipeline_analysis = {}
        for pipeline, results in pipeline_results.items():
            if results.get('status') == 'success':
                metrics = results.get('metrics', {})
                pipeline_analysis[pipeline] = {
                    'status': 'operational',
                    'avg_retrieval_score': metrics.get('avg_retrieval_score', 0),
                    'avg_relevance_score': metrics.get('avg_relevance_score', 0),
                    'avg_response_time_ms': metrics.get('avg_response_time_ms', 0),
                    'performance_grade': self._calculate_performance_grade(metrics)
                }
            else:
                pipeline_analysis[pipeline] = {
                    'status': 'failed',
                    'error': results.get('error', 'Unknown error')
                }
        
        return {
            'total_pipelines_tested': summary.get('successful_pipelines', 0),
            'pipelines_failed': summary.get('failed_pipelines', 0),
            'overall_performance': summary.get('overall_metrics', {}),
            'best_pipeline': summary.get('best_pipeline'),
            'pipeline_rankings': summary.get('pipeline_rankings', []),
            'pipeline_analysis': pipeline_analysis
        }
    
    def _calculate_performance_grade(self, metrics: Dict[str, Any]) -> str:
        """Calculate performance grade based on metrics."""
        retrieval_score = metrics.get('avg_retrieval_score', 0)
        relevance_score = metrics.get('avg_relevance_score', 0)
        response_time = metrics.get('avg_response_time_ms', 9999)
        
        # Combined score (retrieval + relevance) / 2
        quality_score = (retrieval_score + relevance_score) / 2
        
        # Performance penalties for slow response times
        time_penalty = 0
        if response_time > 1000:  # > 1 second
            time_penalty = 0.1
        elif response_time > 500:  # > 0.5 seconds
            time_penalty = 0.05
        
        final_score = quality_score - time_penalty
        
        if final_score >= 0.85:
            return 'A'
        elif final_score >= 0.75:
            return 'B'
        elif final_score >= 0.65:
            return 'C'
        elif final_score >= 0.55:
            return 'D'
        else:
            return 'F'
    
    def _analyze_performance_results(self, bench_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze performance benchmark results."""
        if not bench_data:
            return {'status': 'no_data', 'message': 'No benchmark data available'}
        
        summary = bench_data.get('summary_statistics', {})
        bottlenecks = bench_data.get('bottlenecks', [])
        
        # Performance assessment
        avg_time = summary.get('avg_operation_time_ms', 0)
        max_time = summary.get('max_operation_time_ms', 0)
        
        performance_assessment = 'excellent'
        if avg_time > 100:
            performance_assessment = 'poor'
        elif avg_time > 50:
            performance_assessment = 'fair'
        elif avg_time > 25:
            performance_assessment = 'good'
        
        return {
            'performance_assessment': performance_assessment,
            'avg_operation_time_ms': avg_time,
            'max_operation_time_ms': max_time,
            'bottleneck_count': len(bottlenecks),
            'critical_bottlenecks': [b for b in bottlenecks if b.get('severity') == 'high'],
            'memory_usage_mb': summary.get('total_memory_used_mb', 0),
            'bottlenecks': bottlenecks
        }
    
    def _analyze_data_status(self, data_status: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze data infrastructure status."""
        if not data_status or 'error' in data_status:
            return {'status': 'error', 'message': data_status.get('error', 'Unknown error')}
        
        table_status = data_status.get('table_status', {})
        total_docs = data_status.get('total_documents', 0)
        
        # Calculate readiness scores
        ready_tables = sum(1 for t in table_status.values() if t.get('status') == 'ready')
        total_tables = len(table_status)
        readiness_percent = (ready_tables / total_tables) * 100 if total_tables > 0 else 0
        
        # Identify missing components
        missing_components = [
            key for key, status in table_status.items() 
            if status.get('status') in ['empty', 'error']
        ]
        
        return {
            'total_documents': total_docs,
            'table_readiness_percent': readiness_percent,
            'ready_tables': ready_tables,
            'total_tables': total_tables,
            'missing_components': missing_components,
            'table_details': table_status
        }
    
    def _generate_recommendations(self, eval_summary: Dict, perf_summary: Dict, data_summary: Dict) -> List[Dict[str, Any]]:
        """Generate actionable recommendations."""
        recommendations = []
        
        # Data infrastructure recommendations
        if data_summary.get('table_readiness_percent', 0) < 100:
            missing = data_summary.get('missing_components', [])
            recommendations.append({
                'category': 'Data Infrastructure',
                'priority': 'High',
                'title': 'Complete Data Population',
                'description': f'Populate missing data components: {", ".join(missing)}',
                'action': 'Run data population scripts for missing tables',
                'impact': 'Enable additional RAG pipelines'
            })
        
        # Performance recommendations
        bottlenecks = perf_summary.get('critical_bottlenecks', [])
        if bottlenecks:
            recommendations.append({
                'category': 'Performance',
                'priority': 'Medium',
                'title': 'Address Performance Bottlenecks',
                'description': f'Optimize slow operations: {[b["operation"] for b in bottlenecks]}',
                'action': 'Add database indexes or optimize queries',
                'impact': 'Improve overall system responsiveness'
            })
        
        # Pipeline recommendations
        best_pipeline = eval_summary.get('best_pipeline')
        if best_pipeline:
            recommendations.append({
                'category': 'Pipeline Optimization',
                'priority': 'Low',
                'title': f'Optimize Based on {best_pipeline} Success',
                'description': f'{best_pipeline} shows the best performance - analyze its approach for other pipelines',
                'action': 'Study successful pipeline patterns',
                'impact': 'Improve overall pipeline quality'
            })
        
        # Schema standardization
        recommendations.append({
            'category': 'Technical Debt',
            'priority': 'Medium',
            'title': 'Complete Schema Standardization Migration',
            'description': 'Migrate remaining hardcoded table references to use schema manager',
            'action': 'Update population scripts to use config-driven approach',
            'impact': 'Eliminate inconsistencies and improve maintainability'
        })
        
        return recommendations
    
    def _create_executive_summary(self, eval_summary: Dict, perf_summary: Dict, data_summary: Dict) -> Dict[str, Any]:
        """Create executive summary."""
        total_docs = data_summary.get('total_documents', 0)
        ready_pipelines = eval_summary.get('total_pipelines_tested', 0)
        best_pipeline = eval_summary.get('best_pipeline', 'N/A')
        performance = perf_summary.get('performance_assessment', 'unknown')
        
        # Overall system health
        health_score = 0
        health_factors = []
        
        # Data factor (40% weight)
        data_readiness = data_summary.get('table_readiness_percent', 0)
        health_score += (data_readiness / 100) * 0.4
        health_factors.append(f"Data: {data_readiness:.0f}%")
        
        # Pipeline factor (40% weight)
        pipeline_readiness = (ready_pipelines / 7) * 100  # 7 total pipelines
        health_score += (pipeline_readiness / 100) * 0.4
        health_factors.append(f"Pipelines: {ready_pipelines}/7")
        
        # Performance factor (20% weight)
        perf_scores = {'excellent': 1.0, 'good': 0.8, 'fair': 0.6, 'poor': 0.4}
        health_score += perf_scores.get(performance, 0.5) * 0.2
        health_factors.append(f"Performance: {performance}")
        
        health_grade = 'Excellent' if health_score >= 0.9 else \
                      'Good' if health_score >= 0.7 else \
                      'Fair' if health_score >= 0.5 else 'Poor'
        
        return {
            'overall_health': health_grade,
            'health_score': round(health_score * 100, 1),
            'health_factors': health_factors,
            'key_metrics': {
                'total_documents': total_docs,
                'operational_pipelines': f"{ready_pipelines}/7",
                'best_performing_pipeline': best_pipeline,
                'avg_response_time_ms': perf_summary.get('avg_operation_time_ms', 0)
            },
            'next_actions': [
                'Complete data population for remaining pipelines',
                'Address identified performance bottlenecks',
                'Migrate remaining scripts to config-driven approach'
            ]
        }
    
    def save_report(self, report: Dict[str, Any], output_file: str = None) -> str:
        """Save evaluation report to file."""
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"reports/comprehensive_evaluation_report_{timestamp}.json"
        
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"💾 Comprehensive report saved to: {output_path}")
        return str(output_path)
    
    def print_executive_summary(self, report: Dict[str, Any]):
        """Print executive summary to console."""
        exec_summary = report.get('executive_summary', {})
        
        print("\n" + "="*80)
        print("📋 RAG TEMPLATES COMPREHENSIVE EVALUATION REPORT")
        print("="*80)
        
        print(f"🎯 Overall System Health: {exec_summary.get('overall_health', 'Unknown')} ({exec_summary.get('health_score', 0)}%)")
        print(f"📊 Health Factors: {' | '.join(exec_summary.get('health_factors', []))}")
        
        key_metrics = exec_summary.get('key_metrics', {})
        print(f"\n📈 Key Metrics:")
        print(f"  📚 Total Documents: {key_metrics.get('total_documents', 0):,}")
        print(f"  🚰 Operational Pipelines: {key_metrics.get('operational_pipelines', 'N/A')}")
        print(f"  🏆 Best Pipeline: {key_metrics.get('best_performing_pipeline', 'N/A')}")
        print(f"  ⚡ Avg Response Time: {key_metrics.get('avg_response_time_ms', 0):.1f}ms")
        
        next_actions = exec_summary.get('next_actions', [])
        if next_actions:
            print(f"\n🎯 Recommended Next Actions:")
            for i, action in enumerate(next_actions, 1):
                print(f"  {i}. {action}")
        
        recommendations = report.get('recommendations', [])
        if recommendations:
            high_priority = [r for r in recommendations if r.get('priority') == 'High']
            if high_priority:
                print(f"\n🚨 High Priority Recommendations:")
                for rec in high_priority:
                    print(f"  • {rec.get('title', 'Unknown')}")
                    print(f"    {rec.get('description', '')}")
        
        print("="*80)

def main():
    """Main execution function."""
    generator = EvaluationReportGenerator()
    
    # Generate comprehensive report
    report = generator.generate_comprehensive_report()
    
    # Save and display results
    output_file = generator.save_report(report)
    generator.print_executive_summary(report)
    
    logger.info(f"✅ Comprehensive evaluation report completed! Report: {output_file}")

if __name__ == "__main__":
    main()