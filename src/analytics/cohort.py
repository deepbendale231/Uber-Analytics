from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

logger = logging.getLogger(__name__)


class CohortAnalyzer:
    """Analyze monthly customer retention cohorts from ride-level data."""

    REQUIRED_COLUMNS = {"Customer ID", "Pickup Time", "Booking Status"}

    def __init__(self, df: pd.DataFrame) -> None:
        """Initialize analyzer and validate required schema.

        Args:
            df: Input ride-level dataframe.

        Raises:
            ValueError: If dataframe is empty or missing required columns.
        """
        if df.empty:
            raise ValueError("Input dataframe is empty. Cohort analysis requires at least one row.")

        missing = self.REQUIRED_COLUMNS.difference(df.columns)
        if missing:
            missing_str = ", ".join(sorted(missing))
            raise ValueError(f"Missing required columns for cohort analysis: {missing_str}")

        self.df = df.copy()
        self._cohort_df: pd.DataFrame | None = None
        self._cohort_counts: pd.DataFrame | None = None
        self._retention_matrix: pd.DataFrame | None = None

        logger.info("Initialized CohortAnalyzer with %s rows.", len(self.df))

    def build_cohorts(self) -> pd.DataFrame:
        """Build customer cohorts with month-based cohort index.

        Converts pickup timestamps, drops unparseable rows, computes each customer's
        cohort month (first completed ride month), and computes month index.

        Returns:
            Dataframe with CohortMonth and CohortIndex columns.

        Raises:
            ValueError: If no valid completed rides remain after preprocessing.
        """
        work = self.df.copy()
        work = work[work["Booking Status"].eq("Success")].copy()

        original_rows = len(work)
        work = work.assign(**{"Pickup Time": pd.to_datetime(work["Pickup Time"], errors="coerce")})
        work = work.dropna(subset=["Pickup Time", "Customer ID"])

        dropped = original_rows - len(work)
        if dropped > 0:
            logger.info("Dropped %s rows with invalid pickup dates or missing customer IDs.", dropped)

        if work.empty:
            raise ValueError("No valid completed rides remain after date parsing and null filtering.")

        work = work.assign(RideMonth=work["Pickup Time"].dt.to_period("M"))
        work = work.assign(CohortMonth=work.groupby("Customer ID")["RideMonth"].transform("min"))

        delta_years = work["RideMonth"].dt.year - work["CohortMonth"].dt.year
        delta_months = work["RideMonth"].dt.month - work["CohortMonth"].dt.month
        work = work.assign(CohortIndex=(delta_years * 12 + delta_months + 1).astype(int))

        self._cohort_df = work
        self._cohort_counts = None
        self._retention_matrix = None

        logger.info("Built cohort dataframe with %s valid completed rides.", len(work))
        return work

    def get_cohort_counts(self) -> pd.DataFrame:
        """Return cohort matrix of unique customer counts.

        Returns:
            Cohort count dataframe indexed by cohort month and columned by cohort index.
        """
        if self._cohort_df is None:
            self.build_cohorts()

        assert self._cohort_df is not None
        cohort_counts = (
            self._cohort_df.groupby(["CohortMonth", "CohortIndex"])["Customer ID"]
            .nunique()
            .unstack(fill_value=0)
            .sort_index()
            .astype(int)
        )

        self._cohort_counts = cohort_counts
        return cohort_counts

    def get_retention_matrix(self) -> pd.DataFrame:
        """Compute row-normalized cohort retention matrix as fractions."""
        if self._cohort_counts is None:
            self.get_cohort_counts()

        assert self._cohort_counts is not None
        if self._cohort_counts.empty:
            self._retention_matrix = self._cohort_counts.copy().astype(float)
            return self._retention_matrix

        if 1 not in self._cohort_counts.columns:
            raise ValueError("Cohort counts matrix has no month-1 cohort size column.")

        cohort_size = self._cohort_counts[1].replace(0, np.nan)
        retention = self._cohort_counts.div(cohort_size, axis=0).fillna(0.0)
        retention = retention.clip(lower=0.0, upper=1.0)

        self._retention_matrix = retention
        return retention

    def plot_retention_heatmap(self) -> go.Figure:
        """Create a plotly retention heatmap figure with percentage annotations."""
        retention = self.get_retention_matrix()

        if retention.empty:
            fig = go.Figure()
            fig.update_layout(
                template="plotly_white",
                title="Customer Cohort Retention",
                xaxis_title="Cohort Index (Months)",
                yaxis_title="Cohort Month",
            )
            return fig

        matrix = retention.copy()
        matrix.index = matrix.index.astype(str)
        text_matrix = matrix.applymap(lambda value: f"{value:.0%}")

        fig = px.imshow(
            matrix,
            x=matrix.columns.astype(int),
            y=matrix.index,
            color_continuous_scale="RdYlGn",
            aspect="auto",
            labels={"x": "Cohort Index (Months)", "y": "Cohort Month", "color": "Retention"},
            title="Customer Cohort Retention",
        )
        fig.update_traces(text=text_matrix.values, texttemplate="%{text}")
        fig.update_layout(template="plotly_white")
        return fig

    def get_retention_summary(self) -> dict[str, Any]:
        """Return summary statistics over the retention matrix."""
        retention = self.get_retention_matrix()
        if retention.empty:
            return {
                "avg_month1_retention": 0.0,
                "best_cohort": None,
                "worst_cohort": None,
                "overall_avg_retention": 0.0,
            }

        month2_col = 2 if 2 in retention.columns else None
        avg_month1_retention = float(retention[month2_col].mean()) if month2_col is not None else 0.0

        comparable = retention.drop(columns=[1], errors="ignore")
        if comparable.empty:
            row_means = retention.mean(axis=1)
        else:
            row_means = comparable.mean(axis=1)

        best_idx = row_means.idxmax() if not row_means.empty else None
        worst_idx = row_means.idxmin() if not row_means.empty else None

        overall_avg = float(retention.to_numpy(dtype=float).mean()) if retention.size else 0.0

        return {
            "avg_month1_retention": avg_month1_retention,
            "best_cohort": str(best_idx) if best_idx is not None else None,
            "worst_cohort": str(worst_idx) if worst_idx is not None else None,
            "overall_avg_retention": overall_avg,
        }

    def export_csvs(self, output_dir: str = "DATA/") -> None:
        """Export cohort counts and retention matrix CSV files.

        Args:
            output_dir: Directory where output CSVs will be written.
        """
        counts = self.get_cohort_counts()
        retention = self.get_retention_matrix()

        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        counts_path = out_dir / "cohort_counts.csv"
        retention_path = out_dir / "cohort_retention_fraction.csv"

        counts.to_csv(counts_path)
        retention.to_csv(retention_path)

        logger.info("Wrote cohort counts to %s", counts_path)
        logger.info("Wrote cohort retention fractions to %s", retention_path)


def run_cohort_pipeline(df: pd.DataFrame) -> dict[str, Any]:
    """Run the full cohort analysis pipeline and return all primary artifacts.

    Args:
        df: Input dataframe containing customer rides.

    Returns:
        Dictionary containing counts, retention matrix, summary, and heatmap figure.
    """
    analyzer = CohortAnalyzer(df)
    analyzer.build_cohorts()
    counts = analyzer.get_cohort_counts()
    retention_matrix = analyzer.get_retention_matrix()
    summary = analyzer.get_retention_summary()
    heatmap_figure = analyzer.plot_retention_heatmap()

    return {
        "counts": counts,
        "retention_matrix": retention_matrix,
        "summary": summary,
        "heatmap_figure": heatmap_figure,
    }
