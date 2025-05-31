#!/usr/bin/env python3
"""
Comprehensive Scaling and Evaluation Orchestrator
Coordinates dataset scaling and comprehensive RAGAS evaluation for all 7 RAG techniques
"""

import sys
import os
import json
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional
import traceback
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from eval.scaling_evaluation_framework import ScalingEvaluationFramework
from scripts.automated_dataset_scaling import AutomatedDatasetScaling
from common.iris_connector_jdbc import get_iris_connection
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ComprehensiveScalingOrchestrator:
    """Orchestrates complete scaling and evaluation pipeline"""
    
    def __init__(self):
        self.connection = get_iris_connection()
        self.scaler = AutomatedDatasetScaling()
        self.evaluator = ScalingEvaluationFramework()
        
        # Complete evaluation plan
        self.evaluation_plan = {
            'dataset_sizes': [1000, 2500, 5000, 10000, 25000, 50000],
            'techniques': [
                'BasicRAG', 'HyDE', 'CRAG', 'ColBERT', 
                'NodeRAG', 'GraphRAG', 'HybridIFindRAG'
            ],
            'ragas_metrics': [
                'answer_relevancy', 'context_precision', 'context_recall',
                'faithfulness', 'answer_similarity', 'answer_correctness',
                'context_relevancy'
            ],
            'performance_metrics': [
                'response_time', 'documents_retrieved', 'similarity_score',
                'answer_length', 'memory_usage', 'success_rate'
            ]
        }
    
    def run_complete_pipeline(self) -> Dict[str, Any]:
        """Run complete scaling and evaluation pipeline"""
        logger.info("🚀 Starting comprehensive scaling and evaluation pipeline...")
        logger.info(f"📋 Plan: {len(self.evaluation_plan['dataset_sizes'])} sizes, {len(self.evaluation_plan['techniques'])} techniques")
        
        pipeline_results = {
            'evaluation_plan': self.evaluation_plan,
            'pipeline_start': datetime.now().isoformat(),
            'scaling_results': {},
            'evaluation_results': {},
            'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S")
        }
        
        # Get current database size
        current_size = self.get_current_document_count()
        logger.info(f"📊 Current database: {current_size:,} documents")
        
        # Scale dataset to each target size and evaluate
        test_sizes = self.evaluation_plan['dataset_sizes']
        logger.info(f"🎯 Will scale and test at sizes: {test_sizes}")
        
        # Run scaling and evaluation at each size
        for size in test_sizes:
            # Scale dataset to target size
            if size > current_size:
                logger.info(f"📈 Scaling dataset from {current_size:,} to {size:,} documents...")
                scaling_result = self.scaler.scale_to_size(size)
                if not scaling_result.get('success', False):
                    logger.error(f"❌ Failed to scale to {size:,} documents")
                    continue
                current_size = self.get_current_document_count()
                logger.info(f"✅ Successfully scaled to {current_size:,} documents")
            
            # Run evaluation at this size
            logger.info(f"\n{'='*60}")
            logger.info(f"🔍 EVALUATING AT {size:,} DOCUMENTS")
            logger.info(f"{'='*60}")
            
            evaluation_result = self.evaluator.run_scaling_evaluation_at_size(size)
            pipeline_results['evaluation_results'][str(size)] = evaluation_result
            
            # Save intermediate results
            self._save_intermediate_results(pipeline_results, size)
        
        # Save final results
        timestamp = pipeline_results['timestamp']
        final_file = f"comprehensive_scaling_pipeline_{timestamp}.json"
        
        with open(final_file, 'w') as f:
            serializable_results = self._make_serializable(pipeline_results)
            json.dump(serializable_results, f, indent=2, default=str)
        
        logger.info(f"💾 Complete pipeline results saved to {final_file}")
        
        # Generate visualizations and report
        self._create_comprehensive_visualizations(pipeline_results, timestamp)
        self._generate_final_report(pipeline_results, timestamp)
        
        logger.info("\n🎉 COMPREHENSIVE SCALING AND EVALUATION PIPELINE COMPLETE!")
        
        return pipeline_results
    
    def get_current_document_count(self) -> int:
        """Get current number of documents in database"""
        try:
            cursor = self.connection.cursor()
            cursor.execute("SELECT COUNT(*) FROM RAG.SourceDocuments")
            count = cursor.fetchone()[0]
            cursor.close()
            return count
        except Exception as e:
            logger.error(f"❌ Failed to get document count: {e}")
            return 0
    
    def _save_intermediate_results(self, results: Dict[str, Any], size: int) -> None:
        """Save intermediate results for recovery"""
        timestamp = results['timestamp']
        intermediate_file = f"pipeline_intermediate_{size}_{timestamp}.json"
        
        with open(intermediate_file, 'w') as f:
            serializable_results = self._make_serializable(results)
            json.dump(serializable_results, f, indent=2, default=str)
        
        logger.info(f"💾 Intermediate results saved to {intermediate_file}")
    
    def _create_comprehensive_visualizations(self, results: Dict[str, Any], timestamp: str) -> None:
        """Create comprehensive visualizations of scaling results"""
        try:
            # Performance vs Scale visualization
            self._create_performance_scale_chart(results, timestamp)
            
            # Quality vs Scale visualization  
            self._create_quality_scale_chart(results, timestamp)
            
            logger.info(f"📊 Comprehensive visualizations created with timestamp: {timestamp}")
            
        except Exception as e:
            logger.error(f"❌ Visualization creation failed: {e}")
    
    def _create_performance_scale_chart(self, results: Dict[str, Any], timestamp: str) -> None:
        """Create performance vs scale chart"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        techniques = self.evaluation_plan['techniques']
        colors = plt.cm.Set3(np.linspace(0, 1, len(techniques)))
        
        for i, technique in enumerate(techniques):
            sizes = []
            response_times = []
            success_rates = []
            
            for size_str, eval_result in results['evaluation_results'].items():
                if technique in eval_result.get('techniques', {}):
                    tech_data = eval_result['techniques'][technique]
                    if tech_data.get('success', False):
                        sizes.append(int(size_str))
                        response_times.append(tech_data['avg_response_time'])
                        success_rates.append(tech_data['success_rate'] * 100)
            
            if sizes:
                ax1.plot(sizes, response_times, 'o-', color=colors[i], label=technique, linewidth=2)
                ax3.plot(sizes, success_rates, '^-', color=colors[i], label=technique, linewidth=2)
        
        # Response Time
        ax1.set_title('Response Time vs Dataset Size', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Dataset Size (documents)')
        ax1.set_ylabel('Response Time (seconds)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xscale('log')
        
        # Success Rate
        ax3.set_title('Success Rate vs Dataset Size', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Dataset Size (documents)')
        ax3.set_ylabel('Success Rate (%)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_xscale('log')
        ax3.set_ylim(0, 105)
        
        # Remove empty subplots
        ax2.remove()
        ax4.remove()
        
        plt.suptitle('RAG Techniques Performance Scaling Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'performance_scaling_analysis_{timestamp}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_quality_scale_chart(self, results: Dict[str, Any], timestamp: str) -> None:
        """Create quality vs scale chart"""
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        techniques = self.evaluation_plan['techniques']
        colors = plt.cm.Set3(np.linspace(0, 1, len(techniques)))
        
        for i, technique in enumerate(techniques):
            sizes = []
            ragas_scores = []
            
            for size_str, eval_result in results['evaluation_results'].items():
                if technique in eval_result.get('techniques', {}):
                    tech_data = eval_result['techniques'][technique]
                    if tech_data.get('ragas_scores'):
                        avg_ragas = np.mean(list(tech_data['ragas_scores'].values()))
                        sizes.append(int(size_str))
                        ragas_scores.append(avg_ragas)
            
            if sizes:
                ax.plot(sizes, ragas_scores, 'o-', color=colors[i], label=technique, linewidth=2, markersize=8)
        
        ax.set_title('RAGAS Quality Scores vs Dataset Size', fontsize=16, fontweight='bold')
        ax.set_xlabel('Dataset Size (documents)', fontsize=12)
        ax.set_ylabel('Average RAGAS Score', fontsize=12)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log')
        
        plt.tight_layout()
        plt.savefig(f'quality_scaling_analysis_{timestamp}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _generate_final_report(self, results: Dict[str, Any], timestamp: str) -> None:
        """Generate comprehensive final report"""
        report_file = f"comprehensive_scaling_report_{timestamp}.md"
        
        with open(report_file, 'w') as f:
            f.write("# Comprehensive RAG Scaling and Evaluation Report\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Executive Summary
            f.write("## Executive Summary\n\n")
            f.write("This report presents the results of a comprehensive scaling and evaluation study ")
            f.write("of 7 RAG techniques across multiple dataset sizes with RAGAS quality metrics.\n\n")
            
            # Evaluation Plan
            plan = results['evaluation_plan']
            f.write("## Evaluation Plan\n\n")
            f.write(f"- **Techniques Tested:** {len(plan['techniques'])}\n")
            f.write(f"- **Dataset Sizes:** {', '.join(map(str, plan['dataset_sizes']))}\n")
            f.write(f"- **RAGAS Metrics:** {', '.join(plan['ragas_metrics'])}\n")
            f.write(f"- **Performance Metrics:** {', '.join(plan['performance_metrics'])}\n\n")
            
            # Results Summary
            f.write("## Results Summary\n\n")
            f.write("### Performance by Technique\n\n")
            f.write("| Technique | Avg Response Time | Success Rate | RAGAS Score |\n")
            f.write("|-----------|-------------------|--------------|-------------|\n")
            
            for technique in plan['techniques']:
                response_times = []
                success_rates = []
                ragas_scores = []
                
                for size_str, eval_result in results['evaluation_results'].items():
                    if technique in eval_result.get('techniques', {}):
                        tech_data = eval_result['techniques'][technique]
                        if tech_data.get('success', False):
                            response_times.append(tech_data['avg_response_time'])
                            success_rates.append(tech_data['success_rate'])
                            if tech_data.get('ragas_scores'):
                                ragas_scores.append(np.mean(list(tech_data['ragas_scores'].values())))
                
                if response_times:
                    avg_rt = np.mean(response_times)
                    avg_sr = np.mean(success_rates) * 100
                    avg_ragas = np.mean(ragas_scores) if ragas_scores else 0
                    
                    f.write(f"| {technique} | {avg_rt:.2f}s | {avg_sr:.0f}% | {avg_ragas:.3f} |\n")
                else:
                    f.write(f"| {technique} | Failed | 0% | N/A |\n")
            
            f.write("\n")
            
            # Detailed Results
            f.write("## Detailed Results\n\n")
            f.write("Detailed results are available in the accompanying JSON files:\n")
            f.write(f"- `comprehensive_scaling_pipeline_{timestamp}.json`\n")
            f.write(f"- Individual intermediate results files\n\n")
            
            # Visualizations
            f.write("## Visualizations\n\n")
            f.write("The following visualizations have been generated:\n")
            f.write(f"- `performance_scaling_analysis_{timestamp}.png`\n")
            f.write(f"- `quality_scaling_analysis_{timestamp}.png`\n\n")
            
            # Recommendations
            f.write("## Recommendations\n\n")
            f.write("### Technique Selection\n")
            f.write("- **GraphRAG**: Best for speed-critical applications\n")
            f.write("- **BasicRAG**: Reliable baseline for production\n")
            f.write("- **CRAG**: Enhanced retrieval with corrective mechanisms\n")
            f.write("- **HyDE**: Quality-focused with hypothetical documents\n")
            f.write("- **NodeRAG**: Maximum coverage for comprehensive retrieval\n")
            f.write("- **HybridIFindRAG**: Multi-modal analysis capabilities\n")
            f.write("- **ColBERT**: Advanced semantic matching (with content limiting)\n\n")
            
            f.write("### Scaling Considerations\n")
            f.write("- Monitor performance degradation with dataset size\n")
            f.write("- Implement caching for frequently asked questions\n")
            f.write("- Consider technique-specific optimizations\n")
            f.write("- Regular quality assessment using RAGAS metrics\n\n")
        
        logger.info(f"📄 Comprehensive report saved to {report_file}")
    
    def _make_serializable(self, data: Any) -> Any:
        """Convert data to JSON-serializable format"""
        if isinstance(data, dict):
            result = {}
            for k, v in data.items():
                if k == 'ragas_scores' and v is not None:
                    result[k] = {key: float(val) for key, val in v.items()}
                else:
                    result[k] = self._make_serializable(v)
            return result
        elif isinstance(data, list):
            return [self._make_serializable(item) for item in data]
        elif isinstance(data, (np.integer, np.floating)):
            return float(data)
        else:
            return data

def main():
    """Main execution function"""
    orchestrator = ComprehensiveScalingOrchestrator()
    
    # Run complete pipeline
    results = orchestrator.run_complete_pipeline()
    
    logger.info("\n🎉 Comprehensive scaling and evaluation pipeline complete!")
    logger.info("📊 Check the generated report and JSON files for detailed results")

if __name__ == "__main__":
    main()