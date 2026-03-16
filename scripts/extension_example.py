import pandas as pd
import numpy as np
from scipy.stats import norm
from cluster_experiments.experiment_analysis import (
    ExperimentAnalysis,
    ConfidenceInterval,
    InferenceResults
)


class SimpleBootstrapNormalAnalysis(ExperimentAnalysis):
    """
    Bootstrap analysis using Normal Approximation.
    Constructs CIs and P-values using the bootstrap Standard Error
    and the Normal distribution.
    """

    def __init__(self, n_bootstrap: int = 1000, seed: int = 42, **kwargs):
        super().__init__(cluster_cols=[], **kwargs)
        self.n_bootstrap = n_bootstrap
        self.rng = np.random.default_rng(seed)

    def _get_ate(self, df: pd.DataFrame) -> float:
        """Point estimate of the Average Treatment Effect"""
        means = df.groupby(self.treatment_col)[self.target_col].mean()
        return means.get(1, 0) - means.get(0, 0)

    def analysis_point_estimate(self, df: pd.DataFrame, verbose: bool = False) -> float:
        return self._get_ate(df)

    def analysis_standard_error(self, df: pd.DataFrame, verbose: bool = False) -> float:
        """Calculates SE as the standard deviation of bootstrap ATEs"""
        boot_ates = [
            self._get_ate(df.sample(
                frac=1.0, replace=True, random_state=self.rng.integers(0, 1e9)))
            for _ in range(self.n_bootstrap)
        ]
        return np.std(boot_ates, ddof=1)

    def analysis_confidence_interval(
            self, df: pd.DataFrame, alpha: float, verbose: bool = False
    ) -> ConfidenceInterval:
        """CI = ATE ± Z * SE"""
        ate = self.analysis_point_estimate(df)
        se = self.analysis_standard_error(df)

        # Critical value from Normal distribution
        z_crit = norm.ppf(1 - alpha / 2)

        return ConfidenceInterval(
            lower=ate - z_crit * se,
            upper=ate + z_crit * se,
            alpha=alpha
        )

    def analysis_pvalue(self, df: pd.DataFrame, verbose: bool = False) -> float:
        """Two-sided p-value using Z-score = ATE / SE"""
        ate = self.analysis_point_estimate(df)
        se = self.analysis_standard_error(df)

        z_score = ate / se
        return 2 * (1 - norm.cdf(abs(z_score)))

    def analysis_inference_results(
            self, df: pd.DataFrame, alpha: float, verbose: bool = False
    ) -> InferenceResults:
        """Combined inference using Normal approximation"""
        ate = self.analysis_point_estimate(df)
        se = self.analysis_standard_error(df)
        p_val = 2 * (1 - norm.cdf(abs(ate / se)))

        z_crit = norm.ppf(1 - alpha / 2)
        ci = ConfidenceInterval(
            lower=ate - z_crit * se,
            upper=ate + z_crit * se,
            alpha=alpha
        )

        return InferenceResults(
            ate=ate,
            p_value=p_val,
            std_error=se,
            conf_int=ci
        )

# --- Usage ---
if __name__ == "__main__":
    df = pd.DataFrame({
        "target": np.random.normal(10, 2, 10_000),
        "treatment": ["A", "B"] * 5_000
    })

    analyser = SimpleBootstrapNormalAnalysis(n_bootstrap=500, treatment="B")
    print(analyser.get_inference_results(df, alpha=0.05))


    # quick experiment to test the consistency of the CI with the p value

    def test_consistency(n_trials=100):
        # Initialize the analysis (B is treatment)
        analyser = SimpleBootstrapNormalAnalysis(n_bootstrap=1000, treatment="B")
        alpha = 0.05

        print(f"{'ATE':>10} | {'P-Value':>10} | {'CI Excludes 0?':>15} | {'P < Alpha?':>10} | {'Consistent?'}")
        print("-" * 75)

        for i in range(n_trials):
            # Generate random data with varying effects to test different scenarios
            effect = np.random.uniform(-1, 2)
            df = pd.DataFrame({
                "target": np.random.normal(loc=0, scale=2, size=10_000),
                "treatment": ["A", "B"] * 5_000
            })
            df.loc[df["treatment"] == "B", "target"] += effect

            # Run analysis
            res = analyser.get_inference_results(df, alpha=alpha)

            # Consistency Logic:
            # 1. Does the CI exclude zero? (Lower > 0 or Upper < 0)
            ci_excludes_zero = (res.conf_int.lower > 0) or (res.conf_int.upper < 0)
            # 2. Is the p-value significant?
            p_is_significant = res.p_value < alpha

            is_consistent = ci_excludes_zero == p_is_significant

            print(
                f"{res.ate:10.4f} | {res.p_value:10.4f} | {str(ci_excludes_zero):>15} | {str(p_is_significant):>10} | {is_consistent}")

            assert is_consistent, f"Inconsistency found at trial {i}!"

    np.random.seed(42)
    test_consistency()