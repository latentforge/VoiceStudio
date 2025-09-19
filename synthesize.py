"""
Main synthesis pipeline for voice cloning evaluation.
"""

from pathlib import Path
from typing import Dict, List

from config.synthesis_config import SynthesisConfig
from config import DatasetType, ModelType, GenerationMethod
from benchmark_datasets import create_dataset, BaseSynthesisDataset
from models import create_synthesizer, BaseSynthesizer
from strategies import create_strategy


class SynthesisPipeline:
    """Main pipeline for synthesis generation."""

    def __init__(self, config: SynthesisConfig = None):
        self.config = config or SynthesisConfig()
        self.datasets: Dict[DatasetType, BaseSynthesisDataset] = {}
        self.models: Dict[ModelType, BaseSynthesizer] = {}

    def setup_dataset(self, dataset_type: DatasetType, root_dir: str = "./data") -> bool:
        """Setup dataset loader.

        Args:
            dataset_type: Type of dataset to load
            root_dir: Root directory for dataset storage

        Returns:
            True if successful, False otherwise
        """
        try:
            dataset_config = self.config.get_dataset_config(dataset_type.value)
            dataset = create_dataset(dataset_type, dataset_config, root_dir=root_dir)
            self.datasets[dataset_type] = dataset
            print(f"Setup dataset: {dataset_type.value}")
            return True
        except Exception as e:
            print(f"Failed to setup dataset {dataset_type.value}: {e}")
            return False

    def setup_model(self, model_type: ModelType) -> bool:
        """Setup TTS model.

        Args:
            model_type: Type of model to load

        Returns:
            True if successful, False otherwise
        """
        try:
            model_config = self.config.get_model_config(model_type.value)
            model = create_synthesizer(model_type, model_config)
            self.models[model_type] = model
            print(f"Setup model: {model_type.value}")
            return True
        except Exception as e:
            print(f"Failed to setup model {model_type.value}: {e}")
            return False

    def run_generation(
            self,
            dataset_type: DatasetType,
            model_type: ModelType,
            methods: List[GenerationMethod] = None
    ) -> bool:
        """Run synthesis generation.

        Args:
            dataset_type: Type of dataset to use
            model_type: Type of model to use
            methods: List of methods to run

        Returns:
            True if successful, False otherwise
        """
        if methods is None:
            methods = [GenerationMethod.METHOD1, GenerationMethod.METHOD2]

        # Check if dataset and model are setup
        if dataset_type not in self.datasets:
            print(f"Dataset {dataset_type.value} not setup")
            return False

        if model_type not in self.models:
            print(f"Model {model_type.value} not setup")
            return False

        dataset = self.datasets[dataset_type]
        model = self.models[model_type]

        success = True

        # Load model once
        with model:
            # Run each method
            for method_type in methods:
                print(f"\n=== Running {method_type.value} for {dataset_type.value} -> {model_type.value} ===")

                try:
                    strategy = create_strategy(method_type, self.config, dataset, model)
                    result = strategy.generate_all(dataset_type.value, model_type.value)
                    success = success and result

                except Exception as e:
                    print(f"Error in {method_type.value}: {e}")
                    success = False

        return success

    def run_all(self) -> bool:
        """Run generation for all configured datasets and models."""
        success = True

        available_datasets = [DatasetType(name) for name in self.config.datasets.keys()]
        available_models = [ModelType(name) for name in self.config.models.keys()]

        # Setup all datasets
        for dataset_type in available_datasets:
            if not self.setup_dataset(dataset_type):
                success = False
                continue

        # Setup all models
        for model_type in available_models:
            if not self.setup_model(model_type):
                success = False
                continue

        # Run generation for all combinations
        for dataset_type in self.datasets.keys():
            for model_type in self.models.keys():
                print(f"\n{'='*50}")
                print(f"Processing: {dataset_type.value} -> {model_type.value}")
                print(f"{'='*50}")

                result = self.run_generation(dataset_type, model_type)
                success = success and result

        return success


def main():
    """Main function for testing."""
    config = SynthesisConfig()
    pipeline = SynthesisPipeline(config)

    print("Setting up synthesis pipeline...")

    pipeline.setup_dataset(DatasetType.LIBRITTS)

    pipeline.setup_model(ModelType.HIGGS_V2)

    pipeline.run_generation(DatasetType.LIBRITTS, ModelType.HIGGS_V2, [GenerationMethod.METHOD1, GenerationMethod.METHOD2])


if __name__ == "__main__":
    main()