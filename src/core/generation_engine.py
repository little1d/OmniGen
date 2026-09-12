import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import random
from ..config import OmniGenConfig
from ..data_loaders import Dataset, DataSample, create_loader_for_source
from ..generators.base import create_generator
from ..models import create_model


class OmniGenEngine:
    """OmniGen core engine: manages model, generator, data loading, generation, and saving."""

    def __init__(self, config: OmniGenConfig):
        self.config = config
        self.dataset: Optional[Dataset] = None
        self.generator = None
        self.model_client = None
        self.used_samples = set()
        self._init_model_client()
        self._init_generator()

    def _init_model_client(self):
        """Initialize model client."""
        model_type = self.config.model_type.lower()
        if self.config.verbose:
            print(f"Initializing model client: {model_type}")
        try:
            self.model_client = create_model(model_type, self.config.model_config)
        except Exception as error:
            raise RuntimeError(f"Failed to initialize model client '{model_type}': {error}") from error

    def _init_generator(self):
        """Initialize generator."""
        self.generator = create_generator(self.config.generator_type, self.config.generator_config, self.model_client)
        if self.generator is None:
            raise ValueError(f"Cannot create generator: {self.config.generator_type}")

    def set_model_client(self, model_client):
        """Set model client and reinitialize generator."""
        self.model_client = model_client
        self._init_generator()

    def load_dataset(self, data_source: Optional[str] = None) -> Dataset:
        """Load dataset from data source (Hugging Face dataset ID or local path)."""
        if data_source is None:
            data_source = self.config.dataset_path

        # Get loader kwargs from generator_config
        loader_kwargs = self.config.generator_config.get("loader_kwargs", {})

        # Create appropriate loader
        loader = create_loader_for_source(data_source)
        if loader is None:
            raise ValueError(f"No suitable loader found for: {data_source}")

        # Load dataset with kwargs
        self.dataset = loader.load(data_source, **loader_kwargs)

        if self.config.verbose:
            stats = self.dataset.get_stats()
            print(f"Dataset loaded: {stats}")
        return self.dataset

    def load_from_json(self, json_path: str) -> Dataset:
        """Load dataset from existing JSON file for incremental generation.

        This is typically used for reason-only generation where draft results already exist.
        Each JSON item should contain at minimum: id, question, solution.
        """
        json_path = Path(json_path)
        if not json_path.exists():
            raise FileNotFoundError(f"JSON file not found: {json_path}")

        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if not isinstance(data, list):
            raise ValueError(f"Expected JSON array, got {type(data)}")

        # Convert JSON items to DataSample objects
        samples = []
        for item in data:
            # Clean up ID: remove cot_ prefix and type suffix
            original_id = item.get('id', f'unknown_{len(samples)}')
            cleaned_id = original_id.replace('cot_', '').replace('_text_only', '').replace('_multimodal', '')

            # Create DataSample with existing content in metadata
            sample = DataSample(
                id=cleaned_id,
                metadata={
                    'existing_question': item.get('question', ''),
                    'existing_solution': item.get('solution', ''),
                    'original_data': item,  # Keep all original data for merging later
                },
                images={},  # No images needed for reason-only generation
            )
            samples.append(sample)

        self.dataset = Dataset(samples=samples)

        if self.config.verbose:
            print(f"Loaded {len(samples)} samples from JSON: {json_path}")

        return self.dataset

    async def generate(self, num_samples: Optional[int] = None, use_sequential: bool = False) -> List[Dict[str, Any]]:
        """Generate data samples.

        Args:
            num_samples: Number of samples to generate. If None, uses config.num_samples
            use_sequential: If True, process samples sequentially instead of randomly.
                           This is useful for incremental generation where order matters.
        """
        if self.dataset is None:
            raise ValueError("Please load dataset first.")
        if num_samples is None:
            num_samples = self.config.num_samples
        if self.config.verbose:
            print(f"Generating {num_samples} samples...")
        generated_data = []

        # Determine which samples to process
        if use_sequential:
            # For incremental generation: process samples in order
            samples_to_process = self.dataset.samples[:num_samples]
        else:
            # For normal generation: select samples randomly
            samples_to_process = None

        for i in range(num_samples):
            if self.config.verbose:
                print(f"Generating sample {i+1}/{num_samples} ...")

            # Select sample
            if use_sequential:
                if i >= len(samples_to_process):
                    break
                sample = samples_to_process[i]
                if self.config.verbose:
                    print(f"Processing sample: {sample.id}")
            else:
                sample = self._select_sample()

            result = await self._generate_with_retry(sample)
            if result is not None:
                # For incremental generation, merge with original data
                if use_sequential and sample.metadata.get('original_data'):
                    original = sample.metadata['original_data']
                    merged = original.copy()
                    merged.update(result)
                    generated_data.append(merged)
                    final_result = merged
                else:
                    generated_data.append(result)
                    final_result = result

                # Incremental save: save after each successful generation
                try:
                    self.save_results_incremental(final_result)
                except Exception as e:
                    if self.config.verbose:
                        print(f"  Warning: Failed to save incrementally: {e}")

                if self.config.verbose:
                    print(f"  Success: {len(generated_data)}/{num_samples}")
            else:
                if self.config.verbose:
                    print(f"  Failed to generate sample {i+1}, skipped after max retries.")
                # For incremental generation, keep original data on failure
                if use_sequential and sample.metadata.get('original_data'):
                    generated_data.append(sample.metadata['original_data'])
                    if self.config.verbose:
                        print(f"  Kept original data without reasoning")

        if self.config.verbose:
            print(f"Generation finished. {len(generated_data)} valid samples generated.")
        return generated_data

    async def _generate_with_retry(self, sample: DataSample) -> Optional[Dict[str, Any]]:
        """Generate a single sample with retry logic."""
        retry_count = 0
        result = None
        while retry_count <= self.config.max_retries and result is None:
            try:
                if self.config.verbose and retry_count > 0:
                    print(f"  Retry {retry_count} ...")
                maybe_result = self.generator.generate_single(sample)
                if maybe_result is not None:
                    result = maybe_result
                    break
                else:
                    if self.config.verbose:
                        print("  Generator returned None, retrying ...")
            except Exception as e:
                if self.config.verbose:
                    print(f"  Exception during generation (retry {retry_count}/{self.config.max_retries}): {e}")
            retry_count += 1
        return result

    def _select_sample(self) -> DataSample:
        """Randomly select a sample, ensuring balanced usage."""
        available_samples = [s for s in self.dataset.samples if s.id not in self.used_samples]
        if not available_samples:
            if self.config.verbose:
                print("All samples used, resetting usage tracker.")
            self.used_samples.clear()
            available_samples = self.dataset.samples

        if not available_samples:
            raise ValueError("No samples available in dataset")

        selected_sample = random.choice(available_samples)
        self.used_samples.add(selected_sample.id)
        if self.config.verbose:
            print(f"Selected sample: {selected_sample.id}")
        return selected_sample

    def save_results(self, results: List[Dict[str, Any]], output_path: Optional[str] = None):
        """Save generated results to JSON file."""
        if output_path is None:
            output_path = self.config.output_path
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # 确保输出文件为JSON格式
        if output_path.suffix.lower() != '.json':
            output_path = output_path.with_suffix('.json')

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        if self.config.verbose:
            print(f"Results saved to: {output_path}")

    def save_results_incremental(self, new_result: Dict[str, Any], output_path: Optional[str] = None):
        """
        Incrementally save a single result to JSON file.

        This method appends new results to existing file, preventing data loss
        in case of interruption during large batch generation.
        """
        if output_path is None:
            output_path = self.config.output_path
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # 确保输出文件为JSON格式
        if output_path.suffix.lower() != '.json':
            output_path = output_path.with_suffix('.json')

        # Load existing results if file exists
        existing_results = []
        if output_path.exists():
            try:
                with open(output_path, 'r', encoding='utf-8') as f:
                    existing_results = json.load(f)
                if not isinstance(existing_results, list):
                    existing_results = []
            except:
                existing_results = []

        # Append new result
        existing_results.append(new_result)

        # Save updated results
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(existing_results, f, ensure_ascii=False, indent=2)

    async def run(self, use_sequential: bool = False) -> List[Dict[str, Any]]:
        """Run the full generation pipeline.

        Args:
            use_sequential: If True, process samples sequentially (for incremental generation)
        """
        # Note: dataset should already be loaded via load_dataset() or load_from_json()
        if self.dataset is None:
            self.load_dataset()
        results = await self.generate(use_sequential=use_sequential)
        if results:
            self.save_results(results)
        return results
