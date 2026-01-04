"""
Evaluation pipeline for synthesized audio quality assessment.
"""

from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np
from tqdm import tqdm

from ..metrics.presets import DatasetType, ModelType, GenerationMethod
from ..metrics import MetricType, create_calculator, ModelConfig


class EvaluationPipeline:
    """Pipeline for evaluating synthesized audio quality."""

    def __init__(self, base_dir: Path = Path("results")):
        self.base_dir = Path(base_dir)
        self.ref_dir = self.base_dir / "ref"
        self.syn_dir = self.base_dir / "syn"

    def get_audio_pairs_with_metadata(
        self,
        dataset_type: DatasetType,
        model_type: ModelType,
        method: GenerationMethod
    ) -> List[Tuple[Path, Path, Optional[str]]]:
        """Get reference-synthesis audio pairs with metadata for proper grouping.

        Args:
            dataset_type: Dataset type
            model_type: Model type
            method: Generation method

        Returns:
            List of (reference_path, synthesis_path, reference_id) tuples
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
                    pairs.append((ref_file, syn_file, index))
                else:
                    print(f"Warning: Missing synthesis file {syn_file}")

        elif method == GenerationMethod.METHOD2:
            set_dirs = sorted(d for d in syn_base.iterdir() if d.is_dir() and d.name.startswith('set_'))

            for set_dir in set_dirs:
                ref_index = set_dir.stem.split('_')[1]
                syn_files = sorted(set_dir.glob("syn_*.wav"))

                if len(syn_files) < 2:
                    print(f"Warning: Skipping {set_dir} for METHOD2, found {len(syn_files)} files (need >= 2).")
                    continue

                consistency_ref_file = syn_files[0]
                other_syn_files = syn_files[1:]

                for syn_file in other_syn_files:
                    pairs.append((consistency_ref_file, syn_file, ref_index))

        elif method == GenerationMethod.METHOD3:
            set_dirs = sorted(d for d in syn_base.iterdir() if d.is_dir() and d.name.startswith('set_'))

            for set_dir in set_dirs:
                ref_index = set_dir.stem.split('_')[1]
                syn_files = sorted(set_dir.glob("syn_*.wav"))

                if len(syn_files) < 3:
                    print(f"Warning: Skipping {set_dir} for METHOD3, found {len(syn_files)} files (need 3: T1, T2, T3).")
                    continue

                consistency_ref_file = syn_files[0]
                other_syn_files = syn_files[1:]

                for syn_file in other_syn_files:
                    pairs.append((consistency_ref_file, syn_file, ref_index))
        return pairs

    @staticmethod
    def evaluate_pairs_with_grouping(
        pairs: List[Tuple[Path, Path, Optional[str]]],
        metric_types: List[MetricType],
        batch_size: int = 16
    ) -> Dict[MetricType, Dict[str, List[float]]]:
        """Evaluate pairs and group results by reference ID.

        Args:
            pairs: List of (reference, synthesis, reference_id) tuples
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
                    # Extract just the audio paths for validation and calculation
                    audio_pairs = [(ref_path, syn_path) for ref_path, syn_path, _ in pairs]
                    valid_pairs = calculator.validate_audio_files(audio_pairs)
                    print(f"Valid pairs: {len(valid_pairs)}/{len(audio_pairs)}")

                    if not valid_pairs:
                        results[metric_type] = {}
                        continue

                    # Calculate scores
                    scores = calculator.calculate_batch_optimized(valid_pairs)

                    # Group scores by reference ID
                    grouped_scores = {}
                    valid_pair_idx = 0

                    for ref_path, syn_path, ref_id in pairs:
                        if (ref_path, syn_path) in valid_pairs:
                            if valid_pair_idx < len(scores) and not np.isnan(scores[valid_pair_idx]):
                                if ref_id not in grouped_scores:
                                    grouped_scores[ref_id] = []
                                grouped_scores[ref_id].append(scores[valid_pair_idx])
                            valid_pair_idx += 1

                    results[metric_type] = grouped_scores

                    # Print grouping summary
                    total_scores = sum(len(scores) for scores in grouped_scores.values())
                    print(f"Grouped scores: {total_scores} scores in {len(grouped_scores)} groups")

            except Exception as e:
                print(f"Error calculating {metric_type.value}: {e}")
                results[metric_type] = {}

        return results

    @staticmethod
    def calculate_method1_statistics(
        grouped_results: Dict[MetricType, Dict[str, List[float]]]
    ) -> Dict[str, float]:
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
        grouped_results: Dict[MetricType, Dict[str, List[float]]]
    ) -> Dict[str, float]:
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
        metric_types: List[MetricType] = None,
        methods: List[GenerationMethod] = None
    ) -> Dict[GenerationMethod, Dict[str, float]]:
        """Evaluate a specific dataset-model combination.

        Args:
            dataset_type: Dataset to evaluate
            model_type: Model to evaluate
            metric_types: Metrics to calculate (default: all)
            methods: Methods to evaluate (default: both)

        Returns:
            Dictionary mapping methods to their statistics
        """
        if metric_types is None:
            metric_types = [MetricType.UTMOS, MetricType.WER, MetricType.SIM, MetricType.FFE, MetricType.MCD]

        if methods is None:
            methods = [GenerationMethod.METHOD1, GenerationMethod.METHOD2, GenerationMethod.METHOD3]

        results = {}

        for method in methods:
            print(f"\n{'='*60}")
            print(f"Evaluating: {dataset_type.value} -> {model_type.value} -> {method.value}")
            print(f"{'='*60}")

            # Get audio pairs with metadata
            pairs = self.get_audio_pairs_with_metadata(dataset_type, model_type, method)
            print(f"Found {len(pairs)} audio pairs")

            if not pairs:
                print(f"No audio pairs found for {method.value}")
                continue

            # Validate grouping for Method2
            if method == GenerationMethod.METHOD2:
                ref_groups = {}
                for _, _, ref_id in pairs:
                    if ref_id not in ref_groups:
                        ref_groups[ref_id] = 0
                    ref_groups[ref_id] += 1

                print(f"Reference groups: {len(ref_groups)} groups")
                group_sizes = list(ref_groups.values())
                if group_sizes:
                    print(f"Group sizes: min={min(group_sizes)}, max={max(group_sizes)}, avg={np.mean(group_sizes):.1f}")

            # Evaluate pairs with proper grouping
            grouped_results = self.evaluate_pairs_with_grouping(pairs, metric_types)

            # Calculate statistics based on method
            if method == GenerationMethod.METHOD1:
                stats = self.calculate_method1_statistics(grouped_results)
            else:  # METHOD2
                stats = self.calculate_method2_statistics(grouped_results)

            results[method] = stats

            # Print summary
            print(f"\nResults for {method.value}:")
            for key, value in stats.items():
                if isinstance(value, (int, float)):
                    print(f"  {key}: {value:.4f}")
                else:
                    print(f"  {key}: {value}")

        return results

    @staticmethod
    def save_results_to_csv(
        results: Dict[GenerationMethod, Dict[str, float]],
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
