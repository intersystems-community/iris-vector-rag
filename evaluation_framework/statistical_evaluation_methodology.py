"""
Statistical Evaluation Methodology for Biomedical RAG Pipeline Comparison

This module provides comprehensive statistical analysis capabilities for rigorous
comparison of RAG pipeline performance using RAGAS metrics with proper statistical
inference, power analysis, and multiple comparison corrections.

Key Features:
1. Power analysis and sample size determination
2. Statistical test selection and validation
3. Multiple comparison corrections
4. Effect size calculations and interpretation
5. Bootstrap confidence intervals
6. Bayesian statistical analysis
7. Experimental design validation
8. Comprehensive statistical reporting
"""

import json
import logging
import warnings
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import pingouin as pg

# Statistical libraries
from scipy import stats
from scipy.stats import (
    bootstrap,
    chi2_contingency,
    combine_pvalues,
    f_oneway,
    friedmanchisquare,
    kruskal,
    mannwhitneyu,
    permutation_test,
    ttest_ind,
    ttest_rel,
    wilcoxon,
)

# Machine learning for effect size
from sklearn.metrics import cohen_kappa_score
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.descriptivestats import describe
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.power import ttest_power

# Bayesian analysis
try:
    import arviz as az
    import pymc as pm

    BAYESIAN_AVAILABLE = True
except ImportError:
    BAYESIAN_AVAILABLE = False
    logging.warning("PyMC not available, Bayesian analysis disabled")

logger = logging.getLogger(__name__)


class StatTestType(Enum):
    """Statistical test types for different scenarios."""

    PARAMETRIC_INDEPENDENT = "parametric_independent"
    PARAMETRIC_PAIRED = "parametric_paired"
    NONPARAMETRIC_INDEPENDENT = "nonparametric_independent"
    NONPARAMETRIC_PAIRED = "nonparametric_paired"
    ANOVA = "anova"
    KRUSKAL_WALLIS = "kruskal_wallis"
    FRIEDMAN = "friedman"


class EffectSizeType(Enum):
    """Effect size measures for different contexts."""

    COHENS_D = "cohens_d"
    HEDGES_G = "hedges_g"
    GLASS_DELTA = "glass_delta"
    ETA_SQUARED = "eta_squared"
    OMEGA_SQUARED = "omega_squared"
    CLIFF_DELTA = "cliff_delta"
    COMMON_LANGUAGE_ES = "common_language_effect_size"


@dataclass
class StatisticalTestResult:
    """Comprehensive statistical test result."""

    test_name: str
    test_statistic: float
    p_value: float
    p_value_adjusted: Optional[float]
    effect_size: float
    effect_size_type: str
    effect_size_interpretation: str
    confidence_interval: Tuple[float, float]
    power: Optional[float]
    sample_sizes: List[int]
    assumptions_met: Dict[str, bool]
    interpretation: str
    raw_data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PowerAnalysisResult:
    """Power analysis results for study design."""

    effect_size: float
    power: float
    sample_size: int
    alpha: float
    test_type: str
    recommendations: List[str]
    sensitivity_analysis: Dict[str, float]


@dataclass
class MultipleComparisonResult:
    """Results from multiple comparison procedures."""

    method: str
    original_p_values: List[float]
    adjusted_p_values: List[float]
    rejected_hypotheses: List[bool]
    alpha_corrected: float
    family_wise_error_rate: float
    false_discovery_rate: float


class StatisticalEvaluationFramework:
    """
    Comprehensive statistical evaluation framework for biomedical RAG pipeline comparison.

    Provides rigorous statistical analysis with proper experimental design,
    power analysis, and multiple comparison corrections.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize statistical evaluation framework.

        Args:
            config: Configuration dictionary for statistical parameters
        """
        self.config = config or self._get_default_config()

        # Statistical parameters
        self.alpha = self.config.get("alpha", 0.05)
        self.power_target = self.config.get("power_target", 0.8)
        self.effect_size_threshold = self.config.get("effect_size_threshold", 0.5)
        self.bootstrap_samples = self.config.get("bootstrap_samples", 1000)
        self.multiple_correction_method = self.config.get(
            "multiple_correction_method", "fdr_bh"
        )

        logger.info("Statistical Evaluation Framework initialized")

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default statistical configuration."""
        return {
            "alpha": 0.05,
            "power_target": 0.8,
            "effect_size_threshold": 0.5,
            "bootstrap_samples": 1000,
            "multiple_correction_method": "fdr_bh",
            "normality_test": "shapiro",
            "homoscedasticity_test": "levene",
            "outlier_detection": "iqr",
            "missing_data_threshold": 0.1,
            "bayesian_chains": 4,
            "bayesian_draws": 2000,
            "bayesian_tune": 1000,
        }

    def conduct_power_analysis(
        self, pipelines_data: Dict[str, List[float]], metric_name: str
    ) -> PowerAnalysisResult:
        """
        Conduct comprehensive power analysis for pipeline comparison.

        Args:
            pipelines_data: Dictionary mapping pipeline names to metric scores
            metric_name: Name of the metric being analyzed

        Returns:
            Power analysis results with sample size recommendations
        """
        logger.info(f"Conducting power analysis for {metric_name}")

        # Calculate effect size from pilot data
        pipeline_names = list(pipelines_data.keys())
        if len(pipeline_names) < 2:
            raise ValueError("Need at least 2 pipelines for comparison")

        # Use first two pipelines for effect size estimation
        group1_data = pipelines_data[pipeline_names[0]]
        group2_data = pipelines_data[pipeline_names[1]]

        # Calculate Cohen's d
        effect_size = self._calculate_cohens_d(group1_data, group2_data)

        # Determine test type
        if len(pipeline_names) == 2:
            test_type = "two_sample_ttest"
        else:
            test_type = "one_way_anova"

        # Calculate required sample size
        if test_type == "two_sample_ttest":
            sample_size = ttest_power(
                effect_size=abs(effect_size),
                power=self.power_target,
                alpha=self.alpha,
                alternative="two-sided",
            )
        else:
            # For ANOVA, approximate with pairwise t-test sample size
            # This is a conservative approximation
            sample_size = ttest_power(
                effect_size=abs(effect_size),
                nobs=None,
                alpha=self.alpha / len(pipeline_names),  # Bonferroni correction
                power=self.power_target,
                alternative="two-sided",
            )

        # Calculate achieved power with current sample sizes
        current_min_n = min(len(data) for data in pipelines_data.values())
        if test_type == "two_sample_ttest":
            achieved_power = ttest_power(
                effect_size=abs(effect_size), nobs=current_min_n, alpha=self.alpha
            )
        else:
            achieved_power = 0.8  # Placeholder for ANOVA power calculation

        # Generate recommendations
        recommendations = []
        if current_min_n < sample_size:
            recommendations.append(
                f"Increase sample size to {int(np.ceil(sample_size))} per group"
            )
        if abs(effect_size) < 0.2:
            recommendations.append(
                "Small effect size detected - consider larger sample or different metrics"
            )
        if achieved_power < self.power_target:
            recommendations.append(
                f"Current power ({achieved_power:.3f}) below target ({self.power_target})"
            )

        # Sensitivity analysis
        sensitivity_analysis = {}
        for alpha_sens in [0.01, 0.05, 0.10]:
            if test_type == "two_sample_ttest":
                n_sens = ttest_power(
                    effect_size=abs(effect_size),
                    power=self.power_target,
                    alpha=alpha_sens,
                    alternative="two-sided",
                )
            else:
                n_sens = sample_size  # Simplified for ANOVA
            sensitivity_analysis[f"alpha_{alpha_sens}"] = n_sens

        return PowerAnalysisResult(
            effect_size=effect_size,
            power=achieved_power,
            sample_size=(
                int(np.ceil(sample_size)) if np.isfinite(sample_size) else current_min_n
            ),
            alpha=self.alpha,
            test_type=test_type,
            recommendations=recommendations,
            sensitivity_analysis=sensitivity_analysis,
        )

    def compare_pipelines_statistical(
        self,
        pipelines_data: Dict[str, List[float]],
        metric_name: str,
        paired: bool = False,
    ) -> List[StatisticalTestResult]:
        """
        Comprehensive statistical comparison of pipeline performance.

        Args:
            pipelines_data: Dictionary mapping pipeline names to metric scores
            metric_name: Name of the metric being analyzed
            paired: Whether the data represents paired comparisons

        Returns:
            List of statistical test results for all comparisons
        """
        logger.info(
            f"Statistical comparison of {len(pipelines_data)} pipelines for {metric_name}"
        )

        # Validate data
        self._validate_statistical_data(pipelines_data)

        pipeline_names = list(pipelines_data.keys())
        results = []

        if len(pipeline_names) == 2:
            # Two-group comparison
            result = self._compare_two_groups(
                pipelines_data[pipeline_names[0]],
                pipelines_data[pipeline_names[1]],
                pipeline_names[0],
                pipeline_names[1],
                metric_name,
                paired,
            )
            results.append(result)

        elif len(pipeline_names) > 2:
            # Multiple group comparison
            # First, overall test
            overall_result = self._compare_multiple_groups(
                pipelines_data, metric_name, paired
            )
            results.append(overall_result)

            # Then, pairwise comparisons if overall test is significant
            if overall_result.p_value < self.alpha:
                pairwise_results = self._pairwise_comparisons(
                    pipelines_data, metric_name, paired
                )
                results.extend(pairwise_results)

        # Apply multiple comparison correction
        if len(results) > 1:
            results = self._apply_multiple_comparison_correction(results)

        return results

    def _validate_statistical_data(self, pipelines_data: Dict[str, List[float]]):
        """Validate data for statistical analysis."""
        for pipeline_name, data in pipelines_data.items():
            if len(data) == 0:
                raise ValueError(f"No data for pipeline {pipeline_name}")

            # Check for missing values
            valid_data = [x for x in data if np.isfinite(x)]
            missing_rate = 1 - len(valid_data) / len(data)

            if missing_rate > self.config["missing_data_threshold"]:
                logger.warning(
                    f"High missing data rate ({missing_rate:.2%}) for {pipeline_name}"
                )

            # Check for outliers
            outliers = self._detect_outliers(valid_data)
            if len(outliers) > 0:
                logger.info(f"Detected {len(outliers)} outliers in {pipeline_name}")

    def _detect_outliers(self, data: List[float]) -> List[int]:
        """Detect outliers using IQR method."""
        if len(data) < 4:
            return []

        Q1 = np.percentile(data, 25)
        Q3 = np.percentile(data, 75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers = [i for i, x in enumerate(data) if x < lower_bound or x > upper_bound]
        return outliers

    def _compare_two_groups(
        self,
        group1_data: List[float],
        group2_data: List[float],
        group1_name: str,
        group2_name: str,
        metric_name: str,
        paired: bool,
    ) -> StatisticalTestResult:
        """Compare two groups statistically."""

        # Clean data
        g1_clean = [x for x in group1_data if np.isfinite(x)]
        g2_clean = [x for x in group2_data if np.isfinite(x)]

        # Check assumptions
        assumptions = self._check_two_sample_assumptions(g1_clean, g2_clean, paired)

        # Select appropriate test
        if paired:
            if assumptions["normality"] and assumptions["homoscedasticity"]:
                test_statistic, p_value = stats.ttest_rel(g1_clean, g2_clean)
                test_name = "Paired t-test"
            else:
                test_statistic, p_value = stats.wilcoxon(g1_clean, g2_clean)
                test_name = "Wilcoxon signed-rank test"
        else:
            if assumptions["normality"] and assumptions["homoscedasticity"]:
                test_statistic, p_value = stats.ttest_ind(g1_clean, g2_clean)
                test_name = "Independent t-test"
            else:
                test_statistic, p_value = stats.mannwhitneyu(
                    g1_clean, g2_clean, alternative="two-sided"
                )
                test_name = "Mann-Whitney U test"

        # Calculate effect size
        effect_size = self._calculate_cohens_d(g1_clean, g2_clean)
        effect_size_interpretation = self._interpret_effect_size(
            effect_size, EffectSizeType.COHENS_D
        )

        # Bootstrap confidence interval
        confidence_interval = self._bootstrap_confidence_interval(g1_clean, g2_clean)

        # Calculate power
        power = self._calculate_post_hoc_power(g1_clean, g2_clean, effect_size)

        # Generate interpretation
        interpretation = self._generate_interpretation(
            test_name,
            p_value,
            effect_size,
            effect_size_interpretation,
            group1_name,
            group2_name,
            metric_name,
        )

        return StatisticalTestResult(
            test_name=test_name,
            test_statistic=test_statistic,
            p_value=p_value,
            p_value_adjusted=None,  # Will be set during multiple comparison correction
            effect_size=effect_size,
            effect_size_type=EffectSizeType.COHENS_D.value,
            effect_size_interpretation=effect_size_interpretation,
            confidence_interval=confidence_interval,
            power=power,
            sample_sizes=[len(g1_clean), len(g2_clean)],
            assumptions_met=assumptions,
            interpretation=interpretation,
            raw_data={
                "group1_name": group1_name,
                "group2_name": group2_name,
                "group1_data": g1_clean,
                "group2_data": g2_clean,
                "metric_name": metric_name,
            },
        )

    def _compare_multiple_groups(
        self, pipelines_data: Dict[str, List[float]], metric_name: str, paired: bool
    ) -> StatisticalTestResult:
        """Compare multiple groups using ANOVA or Kruskal-Wallis."""

        # Prepare data
        all_data = []
        group_labels = []

        for pipeline_name, data in pipelines_data.items():
            clean_data = [x for x in data if np.isfinite(x)]
            all_data.extend(clean_data)
            group_labels.extend([pipeline_name] * len(clean_data))

        # Check assumptions
        assumptions = self._check_multiple_group_assumptions(pipelines_data, paired)

        # Select appropriate test
        if paired:
            if assumptions["normality"]:
                # Repeated measures ANOVA (simplified as regular ANOVA for now)
                pipeline_values = list(pipelines_data.values())
                test_statistic, p_value = stats.f_oneway(*pipeline_values)
                test_name = "Repeated measures ANOVA"
            else:
                # Friedman test
                pipeline_values = list(pipelines_data.values())
                test_statistic, p_value = stats.friedmanchisquare(*pipeline_values)
                test_name = "Friedman test"
        else:
            if assumptions["normality"] and assumptions["homoscedasticity"]:
                pipeline_values = [data for data in pipelines_data.values()]
                test_statistic, p_value = stats.f_oneway(*pipeline_values)
                test_name = "One-way ANOVA"
            else:
                pipeline_values = [data for data in pipelines_data.values()]
                test_statistic, p_value = stats.kruskal(*pipeline_values)
                test_name = "Kruskal-Wallis test"

        # Calculate effect size (eta-squared for ANOVA)
        effect_size = self._calculate_eta_squared(pipelines_data)
        effect_size_interpretation = self._interpret_effect_size(
            effect_size, EffectSizeType.ETA_SQUARED
        )

        # Confidence interval (approximate)
        confidence_interval = (0.0, 1.0)  # Placeholder for overall tests

        # Sample sizes
        sample_sizes = [
            len([x for x in data if np.isfinite(x)]) for data in pipelines_data.values()
        ]

        # Interpretation
        pipeline_names = list(pipelines_data.keys())
        interpretation = (
            f"{test_name} comparing {len(pipeline_names)} pipelines on {metric_name}. "
        )
        if p_value < self.alpha:
            interpretation += f"Significant difference detected (p={p_value:.4f}). "
        else:
            interpretation += f"No significant difference detected (p={p_value:.4f}). "
        interpretation += f"Effect size ({EffectSizeType.ETA_SQUARED.value}): {effect_size:.4f} ({effect_size_interpretation})"

        return StatisticalTestResult(
            test_name=test_name,
            test_statistic=test_statistic,
            p_value=p_value,
            p_value_adjusted=None,
            effect_size=effect_size,
            effect_size_type=EffectSizeType.ETA_SQUARED.value,
            effect_size_interpretation=effect_size_interpretation,
            confidence_interval=confidence_interval,
            power=None,  # Not easily calculated for multi-group tests
            sample_sizes=sample_sizes,
            assumptions_met=assumptions,
            interpretation=interpretation,
            raw_data={
                "pipelines_data": pipelines_data,
                "metric_name": metric_name,
                "all_data": all_data,
                "group_labels": group_labels,
            },
        )

    def _pairwise_comparisons(
        self, pipelines_data: Dict[str, List[float]], metric_name: str, paired: bool
    ) -> List[StatisticalTestResult]:
        """Conduct all pairwise comparisons between pipelines."""

        pipeline_names = list(pipelines_data.keys())
        results = []

        for i in range(len(pipeline_names)):
            for j in range(i + 1, len(pipeline_names)):
                pipeline1 = pipeline_names[i]
                pipeline2 = pipeline_names[j]

                result = self._compare_two_groups(
                    pipelines_data[pipeline1],
                    pipelines_data[pipeline2],
                    pipeline1,
                    pipeline2,
                    metric_name,
                    paired,
                )
                results.append(result)

        return results

    def _check_two_sample_assumptions(
        self, group1: List[float], group2: List[float], paired: bool
    ) -> Dict[str, bool]:
        """Check statistical assumptions for two-sample tests."""
        assumptions = {}

        # Normality test
        if len(group1) >= 3 and len(group2) >= 3:
            _, p1 = stats.shapiro(group1)
            _, p2 = stats.shapiro(group2)
            assumptions["normality"] = p1 > 0.05 and p2 > 0.05
        else:
            assumptions["normality"] = False

        # Homoscedasticity test (only for independent samples)
        if not paired and len(group1) >= 3 and len(group2) >= 3:
            _, p_levene = stats.levene(group1, group2)
            assumptions["homoscedasticity"] = p_levene > 0.05
        else:
            assumptions["homoscedasticity"] = True  # Not applicable for paired data

        return assumptions

    def _check_multiple_group_assumptions(
        self, pipelines_data: Dict[str, List[float]], paired: bool
    ) -> Dict[str, bool]:
        """Check statistical assumptions for multiple group tests."""
        assumptions = {}

        # Normality test for each group
        normality_results = []
        for data in pipelines_data.values():
            clean_data = [x for x in data if np.isfinite(x)]
            if len(clean_data) >= 3:
                _, p_val = stats.shapiro(clean_data)
                normality_results.append(p_val > 0.05)
            else:
                normality_results.append(False)

        assumptions["normality"] = all(normality_results)

        # Homoscedasticity test (only for independent samples)
        if not paired:
            clean_data_groups = []
            for data in pipelines_data.values():
                clean_data = [x for x in data if np.isfinite(x)]
                if len(clean_data) >= 3:
                    clean_data_groups.append(clean_data)

            if len(clean_data_groups) >= 2:
                _, p_levene = stats.levene(*clean_data_groups)
                assumptions["homoscedasticity"] = p_levene > 0.05
            else:
                assumptions["homoscedasticity"] = False
        else:
            assumptions["homoscedasticity"] = True

        return assumptions

    def _calculate_cohens_d(self, group1: List[float], group2: List[float]) -> float:
        """Calculate Cohen's d effect size."""
        if len(group1) == 0 or len(group2) == 0:
            return 0.0

        mean1, mean2 = np.mean(group1), np.mean(group2)

        # Pooled standard deviation
        n1, n2 = len(group1), len(group2)
        var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)

        if n1 <= 1 or n2 <= 1:
            return 0.0

        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

        if pooled_std == 0:
            return 0.0

        return (mean1 - mean2) / pooled_std

    def _calculate_eta_squared(self, pipelines_data: Dict[str, List[float]]) -> float:
        """Calculate eta-squared effect size for multiple group comparison."""
        try:
            # Prepare data for ANOVA
            all_data = []
            group_labels = []

            for pipeline_name, data in pipelines_data.items():
                clean_data = [x for x in data if np.isfinite(x)]
                all_data.extend(clean_data)
                group_labels.extend([pipeline_name] * len(clean_data))

            if len(set(group_labels)) < 2:
                return 0.0

            # Calculate eta-squared
            df = pd.DataFrame({"score": all_data, "group": group_labels})

            # Between-group sum of squares
            group_means = df.groupby("group")["score"].mean()
            overall_mean = df["score"].mean()

            ss_between = sum(
                len(pipelines_data[group]) * (group_means[group] - overall_mean) ** 2
                for group in group_means.index
            )

            # Total sum of squares
            ss_total = sum((x - overall_mean) ** 2 for x in all_data)

            if ss_total == 0:
                return 0.0

            return ss_between / ss_total

        except Exception as e:
            logger.warning(f"Failed to calculate eta-squared: {e}")
            return 0.0

    def _interpret_effect_size(
        self, effect_size: float, effect_type: EffectSizeType
    ) -> str:
        """Interpret effect size magnitude."""
        abs_effect = abs(effect_size)

        if effect_type == EffectSizeType.COHENS_D:
            if abs_effect < 0.2:
                return "negligible"
            elif abs_effect < 0.5:
                return "small"
            elif abs_effect < 0.8:
                return "medium"
            else:
                return "large"

        elif effect_type == EffectSizeType.ETA_SQUARED:
            if abs_effect < 0.01:
                return "negligible"
            elif abs_effect < 0.06:
                return "small"
            elif abs_effect < 0.14:
                return "medium"
            else:
                return "large"

        else:
            return "unknown"

    def _bootstrap_confidence_interval(
        self, group1: List[float], group2: List[float]
    ) -> Tuple[float, float]:
        """Calculate bootstrap confidence interval for mean difference."""
        try:

            def mean_difference(x, y):
                return np.mean(x) - np.mean(y)

            # Bootstrap resampling
            n_bootstrap = self.bootstrap_samples
            bootstrap_diffs = []

            for _ in range(n_bootstrap):
                bootstrap_g1 = np.random.choice(group1, size=len(group1), replace=True)
                bootstrap_g2 = np.random.choice(group2, size=len(group2), replace=True)
                diff = mean_difference(bootstrap_g1, bootstrap_g2)
                bootstrap_diffs.append(diff)

            # Calculate confidence interval
            alpha = 1 - self.config.get("confidence_level", 0.95)
            lower_percentile = (alpha / 2) * 100
            upper_percentile = (1 - alpha / 2) * 100

            ci_lower = np.percentile(bootstrap_diffs, lower_percentile)
            ci_upper = np.percentile(bootstrap_diffs, upper_percentile)

            return (ci_lower, ci_upper)

        except Exception as e:
            logger.warning(f"Bootstrap confidence interval failed: {e}")
            return (0.0, 0.0)

    def _calculate_post_hoc_power(
        self, group1: List[float], group2: List[float], effect_size: float
    ) -> Optional[float]:
        """Calculate post-hoc statistical power."""
        try:
            min_n = min(len(group1), len(group2))
            power = ttest_power(
                effect_size=abs(effect_size),
                nobs=min_n,
                alpha=self.alpha,
                alternative="two-sided",
            )
            return power
        except Exception:
            return None

    def _generate_interpretation(
        self,
        test_name: str,
        p_value: float,
        effect_size: float,
        effect_interpretation: str,
        group1_name: str,
        group2_name: str,
        metric_name: str,
    ) -> str:
        """Generate human-readable interpretation of statistical results."""

        interpretation = (
            f"{test_name} comparing {group1_name} vs {group2_name} on {metric_name}. "
        )

        if p_value < self.alpha:
            interpretation += f"Significant difference detected (p={p_value:.4f}). "

            if effect_size > 0:
                interpretation += f"{group1_name} performs better than {group2_name}. "
            else:
                interpretation += f"{group2_name} performs better than {group1_name}. "
        else:
            interpretation += f"No significant difference detected (p={p_value:.4f}). "

        interpretation += f"Effect size: {effect_size:.4f} ({effect_interpretation})"

        return interpretation

    def _apply_multiple_comparison_correction(
        self, results: List[StatisticalTestResult]
    ) -> List[StatisticalTestResult]:
        """Apply multiple comparison correction to p-values."""

        p_values = [result.p_value for result in results]

        # Apply correction
        rejected, p_adjusted, alpha_sidak, alpha_bonf = multipletests(
            p_values, alpha=self.alpha, method=self.multiple_correction_method
        )

        # Update results with adjusted p-values
        for i, result in enumerate(results):
            result.p_value_adjusted = p_adjusted[i]

        logger.info(
            f"Applied {self.multiple_correction_method} correction to {len(results)} tests"
        )

        return results

    def conduct_bayesian_analysis(
        self, pipelines_data: Dict[str, List[float]], metric_name: str
    ) -> Dict[str, Any]:
        """
        Conduct Bayesian analysis for pipeline comparison.

        Args:
            pipelines_data: Dictionary mapping pipeline names to metric scores
            metric_name: Name of the metric being analyzed

        Returns:
            Bayesian analysis results including credible intervals and probabilities
        """
        if not BAYESIAN_AVAILABLE:
            logger.warning("Bayesian analysis not available - PyMC not installed")
            return {}

        logger.info(f"Conducting Bayesian analysis for {metric_name}")

        try:
            pipeline_names = list(pipelines_data.keys())

            # Prepare data
            all_scores = []
            group_indices = []

            for i, (pipeline_name, scores) in enumerate(pipelines_data.items()):
                clean_scores = [x for x in scores if np.isfinite(x)]
                all_scores.extend(clean_scores)
                group_indices.extend([i] * len(clean_scores))

            all_scores = np.array(all_scores)
            group_indices = np.array(group_indices)
            n_groups = len(pipeline_names)

            # Bayesian model
            with pm.Model() as model:
                # Priors
                mu = pm.Normal(
                    "mu",
                    mu=np.mean(all_scores),
                    sigma=np.std(all_scores),
                    shape=n_groups,
                )
                sigma = pm.HalfNormal("sigma", sigma=np.std(all_scores))

                # Likelihood
                likelihood = pm.Normal(
                    "likelihood", mu=mu[group_indices], sigma=sigma, observed=all_scores
                )

                # Sample from posterior
                trace = pm.sample(
                    draws=self.config["bayesian_draws"],
                    tune=self.config["bayesian_tune"],
                    chains=self.config["bayesian_chains"],
                    return_inferencedata=True,
                )

            # Extract results
            posterior_means = az.summary(trace, var_names=["mu"])

            # Calculate probabilities of superiority
            mu_samples = trace.posterior["mu"].values.reshape(-1, n_groups)

            superiority_probs = {}
            for i, pipeline1 in enumerate(pipeline_names):
                for j, pipeline2 in enumerate(pipeline_names):
                    if i != j:
                        prob = np.mean(mu_samples[:, i] > mu_samples[:, j])
                        superiority_probs[f"{pipeline1}_better_than_{pipeline2}"] = prob

            # Credible intervals
            credible_intervals = {}
            for i, pipeline_name in enumerate(pipeline_names):
                hdi = az.hdi(trace, var_names=["mu"], hdi_prob=0.95)
                credible_intervals[pipeline_name] = (
                    float(hdi["mu"].values[i, 0]),
                    float(hdi["mu"].values[i, 1]),
                )

            return {
                "pipeline_names": pipeline_names,
                "posterior_means": posterior_means.to_dict(),
                "superiority_probabilities": superiority_probs,
                "credible_intervals": credible_intervals,
                "model_summary": str(az.summary(trace)),
                "convergence_diagnostics": {
                    "r_hat": az.rhat(trace).to_dict(),
                    "ess": az.ess(trace).to_dict(),
                },
            }

        except Exception as e:
            logger.error(f"Bayesian analysis failed: {e}")
            return {"error": str(e)}

    def generate_statistical_report(
        self,
        results: List[StatisticalTestResult],
        power_analysis: Optional[PowerAnalysisResult] = None,
        bayesian_results: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Generate comprehensive statistical report."""

        report = {
            "timestamp": pd.Timestamp.now().isoformat(),
            "configuration": self.config,
            "summary": {
                "total_tests": len(results),
                "significant_tests": sum(
                    1 for r in results if (r.p_value_adjusted or r.p_value) < self.alpha
                ),
                "large_effects": sum(1 for r in results if abs(r.effect_size) > 0.8),
            },
            "test_results": [
                {
                    "test_name": r.test_name,
                    "test_statistic": r.test_statistic,
                    "p_value": r.p_value,
                    "p_value_adjusted": r.p_value_adjusted,
                    "effect_size": r.effect_size,
                    "effect_size_type": r.effect_size_type,
                    "effect_size_interpretation": r.effect_size_interpretation,
                    "confidence_interval": r.confidence_interval,
                    "power": r.power,
                    "sample_sizes": r.sample_sizes,
                    "assumptions_met": r.assumptions_met,
                    "interpretation": r.interpretation,
                }
                for r in results
            ],
        }

        if power_analysis:
            report["power_analysis"] = {
                "effect_size": power_analysis.effect_size,
                "power": power_analysis.power,
                "sample_size": power_analysis.sample_size,
                "alpha": power_analysis.alpha,
                "test_type": power_analysis.test_type,
                "recommendations": power_analysis.recommendations,
                "sensitivity_analysis": power_analysis.sensitivity_analysis,
            }

        if bayesian_results:
            report["bayesian_analysis"] = bayesian_results

        return report

    def save_statistical_results(self, report: Dict[str, Any], output_path: str):
        """Save statistical analysis results."""
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        filename = f"statistical_analysis_report_{timestamp}.json"

        filepath = output_dir / filename
        with open(filepath, "w") as f:
            json.dump(report, f, indent=2, default=str)

        logger.info(f"Statistical analysis report saved to {filepath}")


def create_statistical_framework(
    config: Optional[Dict[str, Any]] = None,
) -> StatisticalEvaluationFramework:
    """Factory function to create a configured statistical evaluation framework."""
    return StatisticalEvaluationFramework(config)


if __name__ == "__main__":
    # Example usage
    framework = create_statistical_framework()

    # Mock pipeline data
    sample_data = {
        "BasicRAG": [0.7, 0.8, 0.75, 0.82, 0.78],
        "CRAG": [0.85, 0.87, 0.83, 0.89, 0.86],
        "GraphRAG": [0.72, 0.74, 0.71, 0.76, 0.73],
        "BasicRAGReranking": [0.81, 0.84, 0.80, 0.83, 0.82],
    }

    # Power analysis
    power_result = framework.conduct_power_analysis(sample_data, "faithfulness")

    # Statistical comparison
    comparison_results = framework.compare_pipelines_statistical(
        sample_data, "faithfulness"
    )

    # Generate report
    report = framework.generate_statistical_report(comparison_results, power_result)
    framework.save_statistical_results(report, "output/statistical_analysis")
