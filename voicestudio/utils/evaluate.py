"""
Evaluation pipeline for synthesized audio quality assessment.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..metrics import MetricType, ModelConfig, create_calculator
from ..metrics.presets import DatasetType, GenerationMethod, ModelType


class EvaluationPipeline:
    """Pipeline for evaluating synthesized audio quality."""

    def __init__(self, base_dir: Path = Path("results")):
        self.base_dir = Path(base_dir)
        self.ref_dir = self.base_dir / "ref"
        self.syn_dir = self.base_dir / "syn"

    @staticmethod
    def _load_metadata(set_dir: Path) -> dict:
        """Load metadata.json from a set directory.
        
        Args:
            set_dir: Path to the set directory
            
        Returns:
            Dictionary containing metadata, or empty dict if file doesn't exist
        """
        metadata_path = set_dir / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}

    def get_audio_pairs_with_metadata(
        self,
        dataset_type: DatasetType,
        model_type: ModelType,
        method: GenerationMethod
    ) -> list[dict]:
        """Get reference-synthesis audio pairs with metadata for proper grouping.

        Args:
            dataset_type: Dataset type
            model_type: Model type
            method: Generation method

        Returns:
            List of dictionaries containing pair info and metadata
        """
        ref_base = self.ref_dir / dataset_type.value / method.value
        syn_base = self.syn_dir / dataset_type.value / model_type.value / method.value

        pairs = []

        if method == GenerationMethod.METHOD1:
            # Method1: Direct 1:1 pairs
            ref_files = sorted(ref_base.glob("ref_*.wav"))

            for ref_file in ref_files:
                # Extract index from ref_001.wav -> syn_001.wav
                index = ref_file.stem.split('_')[1]
                syn_file = syn_base / f"syn_{index}.wav"

                if syn_file.exists():
                    pairs.append({
                        "ref_path": ref_file,
                        "syn_path": syn_file,
                        "ref_id": index,
                        "target_text": None  # Not needed for Method1
                    })
                else:
                    print(f"Warning: Missing synthesis file {syn_file}")

        elif method == GenerationMethod.METHOD2:
            set_dirs = sorted(d for d in syn_base.iterdir() if d.is_dir() and d.name.startswith('set_'))

            for set_dir in set_dirs:
                ref_index = set_dir.stem.split('_')[1]
                syn_files = sorted(set_dir.glob("syn_*.wav"))

                if len(syn_files) < 2:
                    print(f"Warning: Skipping {set_dir} for METHOD2, found {len(syn_files)} files.")
                    continue

                # Load metadata
                metadata = self._load_metadata(set_dir)

                # For reference-based metrics: (syn_0, syn_1), (syn_0, syn_2), ...
                consistency_ref_file = syn_files[0]
                for syn_file in syn_files[1:]:
                    pairs.append({
                        "ref_path": consistency_ref_file,
                        "syn_path": syn_file,
                        "ref_id": ref_index,
                        "target_text": metadata.get(syn_file.name, {}).get("target_text")
                    })
                
                # Special dummy pair to ensure syn_files[0] is seen by no-reference metrics
                pairs.append({
                    "ref_path": consistency_ref_file,
                    "syn_path": consistency_ref_file,
                    "ref_id": ref_index,
                    "target_text": metadata.get(consistency_ref_file.name, {}).get("target_text")
                })

        elif method == GenerationMethod.METHOD3:
            set_dirs = sorted(d for d in syn_base.iterdir() if d.is_dir() and d.name.startswith('set_'))

            for set_dir in set_dirs:
                ref_index = set_dir.stem.split('_')[1]
                syn_files = sorted(set_dir.glob("syn_*.wav"))

                if len(syn_files) < 3:
                    print(f"Warning: Skipping {set_dir} for METHOD3, found {len(syn_files)} files.")
                    continue

                # Load metadata
                metadata = self._load_metadata(set_dir)

                consistency_ref_file = syn_files[0]
                for syn_file in syn_files[1:]:
                    pairs.append({
                        "ref_path": consistency_ref_file,
                        "syn_path": syn_file,
                        "ref_id": ref_index,
                        "target_text": metadata.get(syn_file.name, {}).get("target_text")
                    })
                
                # Special dummy pair to ensure syn_files[0] is seen by no-reference metrics
                pairs.append({
                    "ref_path": consistency_ref_file,
                    "syn_path": consistency_ref_file,
                    "ref_id": ref_index,
                    "target_text": metadata.get(consistency_ref_file.name, {}).get("target_text")
                })

        return pairs

    @staticmethod
    def evaluate_pairs_with_grouping(
        pairs: list[dict],
        metric_types: list[MetricType],
        batch_size: int = 16
    ) -> dict[MetricType, dict[str, list[float]]]:
        """Evaluate pairs and group results by reference ID.

        Args:
            pairs: List of dictionaries containing pair info and metadata
            metric_types: List of metrics to calculate
            batch_size: Batch size for metric calculation

        Returns:
            Dictionary mapping metric_type -> reference_id -> scores
        """
        results = {}

        for metric_type in metric_types:
            print(f"\nCalculating {metric_type.value}...")

            config = ModelConfig(
                name=metric_type.value,
                batch_size=batch_size,
                device="cuda"
            )

            try:
                with create_calculator(metric_type, config) as calculator:
                    is_no_reference = metric_type in [MetricType.UTMOS]
                    is_wer = metric_type == MetricType.WER
                    
                    if is_no_reference:
                        # For no-reference metrics, we only care about unique synthesis files
                        unique_syn_paths = sorted(set([p["syn_path"] for p in pairs]))
                        # Create dummy pairs for calculate_batch_optimized
                        calc_pairs = [(p, p) for p in unique_syn_paths]
                        valid_pairs = calculator.validate_audio_files(calc_pairs)
                        
                        scores_output = calculator.calculate_batch_optimized(valid_pairs)
                        path_to_score = {valid_pairs[i][1]: scores_output[i] for i in range(len(valid_pairs))}
                        
                        # Build mapping of syn_path to ref_id (each file belongs to one ref_id)
                        syn_to_ref = {}
                        for pair_info in pairs:
                            syn_path = pair_info["syn_path"]
                            ref_id = pair_info["ref_id"]
                            if syn_path not in syn_to_ref:
                                syn_to_ref[syn_path] = ref_id
                        
                        # Group scores by ref_id (each unique file added once)
                        grouped_scores = {}
                        for syn_path, score in path_to_score.items():
                            if score is not None and not np.isnan(score):
                                ref_id = syn_to_ref.get(syn_path)
                                if ref_id is not None:
                                    if ref_id not in grouped_scores:
                                        grouped_scores[ref_id] = []
                                    grouped_scores[ref_id].append(score)
                            
                    else:
                        audio_pairs = [(p["ref_path"], p["syn_path"]) for p in pairs]
                        # Filter out our dummy self-pairs for reference-based metrics
                        audio_pairs = [(r, s) for r, s in audio_pairs if r != s]
                        
                        valid_pairs = calculator.validate_audio_files(audio_pairs)
                        
                        # For WER, we need to pass target_text as kwargs
                        if is_wer:
                            # Build target_text mapping for WER
                            pair_to_target = {}
                            for pair_info in pairs:
                                if pair_info["ref_path"] != pair_info["syn_path"]:
                                    key = (pair_info["ref_path"], pair_info["syn_path"])
                                    pair_to_target[key] = pair_info.get("target_text")
                            
                            # Calculate WER with target texts
                            scores = []
                            for ref_path, syn_path in valid_pairs:
                                target_text = pair_to_target.get((ref_path, syn_path))
                                # WER calculator will use target_text if provided, otherwise transcribe reference
                                score = calculator(synthesis=syn_path, reference=ref_path, target_text=target_text)
                                scores.append(score)
                        else:
                            scores = calculator.calculate_batch_optimized(valid_pairs)

                        grouped_scores = {}
                        pair_to_score = {valid_pairs[i]: scores[i] for i in range(len(valid_pairs))}

                        for pair_info in pairs:
                            ref_path = pair_info["ref_path"]
                            syn_path = pair_info["syn_path"]
                            if ref_path == syn_path:
                                continue  # Skip dummy pairs
                            
                            score = pair_to_score.get((ref_path, syn_path))
                            if score is not None and not np.isnan(score):
                                # Scaling for WER and FFE: clamp to [0, 1]
                                if metric_type in [MetricType.WER, MetricType.FFE]:
                                    score = min(1.0, max(0.0, float(score)))
                                
                                ref_id = pair_info["ref_id"]
                                if ref_id not in grouped_scores:
                                    grouped_scores[ref_id] = []
                                grouped_scores[ref_id].append(score)

                    results[metric_type] = grouped_scores
                    total_scores = sum(len(scores) for scores in grouped_scores.values())
                    print(f"Grouped scores: {total_scores} scores in {len(grouped_scores)} groups")

            except Exception as e:
                print(f"Error calculating {metric_type.value}: {e}")
                results[metric_type] = {}

        return results

    @staticmethod
    def calculate_method1_statistics(
        grouped_results: dict[MetricType, dict[str, list[float]]]
    ) -> dict[str, float]:
        """Calculate statistics for Method1 results (simple averages)."""
        stats = {}

        for metric_type, ref_groups in grouped_results.items():
            if not ref_groups:
                continue

            metric_name = metric_type.value

            # Flatten all scores from all groups
            all_scores = []
            for scores in ref_groups.values():
                all_scores.extend(scores)

            if not all_scores:
                continue

            stats[f"{metric_name}_mean"] = np.mean(all_scores)
            stats[f"{metric_name}_std"] = np.std(all_scores)
            stats[f"{metric_name}_median"] = np.median(all_scores)

        return stats

    @staticmethod
    def calculate_method2_statistics(
        grouped_results: dict[MetricType, dict[str, list[float]]]
    ) -> dict[str, float]:
        """Calculate statistics for Method2 results with proper grouping."""
        stats = {}

        for metric_type, ref_groups in grouped_results.items():
            if not ref_groups:
                continue

            metric_name = metric_type.value
            all_scores = []
            group_stds = []
            group_cvs = []

            # Calculate statistics for each reference group
            for ref_id, scores in ref_groups.items():
                if not scores:
                    continue

                all_scores.extend(scores)

                if len(scores) > 1:
                    group_std = np.std(scores, ddof=1)
                    group_stds.append(group_std)

                    mean_score = np.mean(scores)
                    if mean_score > 0:
                        cv = group_std / mean_score
                        group_cvs.append(cv)

                elif len(scores) == 1:
                    group_stds.append(0.0)
                    group_cvs.append(0.0)

            if not all_scores:
                continue

            # Core statistics
            stats[f"{metric_name}_mean"] = np.mean(all_scores)
            stats[f"{metric_name}_std"] = np.std(all_scores)
            stats[f"{metric_name}_median"] = np.median(all_scores)

            # Speaker consistency metrics (core purpose of Method2)
            if group_stds:
                stats[f"{metric_name}_avg_std"] = np.mean(group_stds)

            if group_cvs:
                stats[f"{metric_name}_avg_cv"] = np.mean(group_cvs)

        return stats

    def evaluate_dataset_model(
        self,
        dataset_type: DatasetType,
        model_type: ModelType,
        metric_types: list[MetricType] = None,
        methods: list[GenerationMethod] = None
    ) -> dict[GenerationMethod, dict[str, float]]:
        """Evaluate a specific dataset-model combination."""
        if metric_types is None:
            metric_types = [MetricType.UTMOS, MetricType.WER, MetricType.SIM, MetricType.FFE, MetricType.MCD]

        if methods is None:
            methods = [GenerationMethod.METHOD1, GenerationMethod.METHOD2, GenerationMethod.METHOD3]

        results = {}

        for method in methods:
            print(f"\n{'='*60}")
            print(f"Evaluating: {dataset_type.value} -> {model_type.value} -> {method.value}")
            print(f"{'='*60}")

            pairs = self.get_audio_pairs_with_metadata(dataset_type, model_type, method)
            print(f"Found {len(pairs)} primary audio pairs/files")

            if not pairs:
                print(f"No audio samples found for {method.value}")
                continue

            grouped_results = self.evaluate_pairs_with_grouping(pairs, metric_types)

            if method == GenerationMethod.METHOD1:
                stats = self.calculate_method1_statistics(grouped_results)
            else:
                stats = self.calculate_method2_statistics(grouped_results)

            results[method] = stats
            self.print_markdown_table(method, stats)

        return results

    @staticmethod
    def print_markdown_table(method: GenerationMethod, stats: dict[str, float]) -> None:
        """Print statistics in a Markdown table format with specific ordering and proper alignment."""
        # Metric order: UTMOS, WER, COS (SIM), FFE, MCD
        # Stat order: Mean, Std, Median, Avg Std, Avg CV
        
        metrics = [
            ("UTMOS", "utmos"),
            ("WER", "wer"),
            ("COS", "sim"),
            ("FFE", "ffe"),
            ("MCD", "mcd")
        ]
        
        stat_keys = [
            ("Mean", "mean"),
            ("Std", "std"),
            ("Median", "median"),
            ("Avg Std", "avg_std"),
            ("Avg CV", "avg_cv")
        ]
        
        # Build data rows
        data_rows = []
        for label, metric_prefix in metrics:
            row_items = [label]
            has_data = False
            for _, stat_suffix in stat_keys:
                key = f"{metric_prefix}_{stat_suffix}"
                val = stats.get(key)
                if val is not None:
                    row_items.append(f"{val:.4f}")
                    has_data = True
                else:
                    row_items.append("-")
            
            if has_data:
                data_rows.append(row_items)

        if not data_rows:
            print(f"\n### Evaluation Results: {method.value}")
            print("No data available")
            return

        # Calculate column widths
        header_row = ["Metric"] + [s[0] for s in stat_keys]
        col_widths = [len(h) for h in header_row]
        
        for row in data_rows:
            for i, item in enumerate(row):
                col_widths[i] = max(col_widths[i], len(item))

        # Add padding
        col_widths = [w + 2 for w in col_widths]

        print(f"\n### Evaluation Results: {method.value}")

        # Print header
        header_str = "|"
        for i, item in enumerate(header_row):
            if i == 0:
                # Left-align metric names
                header_str += " " + item.ljust(col_widths[i] - 1) + "|"
            else:
                # Center-align stat names
                header_str += item.center(col_widths[i]) + "|"
        print(header_str)

        # Print separator
        sep_str = "|"
        # First column left-aligned
        sep_str += " " + ("-" * (col_widths[0] - 2)) + " |"
        # Other columns center-aligned
        for w in col_widths[1:]:
            sep_str += ":" + ("-" * (w - 2)) + ":|"
        print(sep_str)

        # Print data rows
        for row in data_rows:
            row_str = "|"
            for i, item in enumerate(row):
                if i == 0:
                    # Left-align metric names
                    row_str += " " + item.ljust(col_widths[i] - 1) + "|"
                else:
                    # Center-align numbers
                    row_str += item.center(col_widths[i]) + "|"
            print(row_str)
    @staticmethod
    def save_results_to_csv(
        results: dict[GenerationMethod, dict[str, float]],
        dataset_type: DatasetType,
        model_type: ModelType,
        output_dir: Path = Path("results")
    ) -> None:
        """Save evaluation results to CSV files.

        Args:
            results: Evaluation results
            dataset_type: Dataset type
            model_type: Model type
            output_dir: Output directory
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        for method, stats in results.items():
            if not stats:
                continue

            # Convert to DataFrame
            df = pd.DataFrame([stats])
            df.insert(0, 'dataset', dataset_type.value)
            df.insert(1, 'model', model_type.value)
            df.insert(2, 'method', method.value)

            # Save to CSV
            filename = f"{dataset_type.value}_{model_type.value}_{method.value}_results.csv"
            filepath = output_dir / filename
            df.to_csv(filepath, index=False)

            print(f"Saved results to {filepath}")


def main():
    """Main evaluation function."""
    evaluator = EvaluationPipeline()

    methods_to_run = [GenerationMethod.METHOD1, GenerationMethod.METHOD2, GenerationMethod.METHOD3]

    results = evaluator.evaluate_dataset_model(
        dataset_type=DatasetType.LIBRITTS,
        model_type=ModelType.PARLER_TTS_MINI_V1,
        methods=methods_to_run
    )

    # Save results
    evaluator.save_results_to_csv(
        results,
        DatasetType.LIBRITTS,
        ModelType.PARLER_TTS_MINI_V1
    )


if __name__ == "__main__":
    main()
