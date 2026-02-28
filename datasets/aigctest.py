"""
AIGCTest dataset for cross-generator evaluation.

This module provides dataset classes for evaluating models across
different AI generators using the AIGCTest structure.
"""

import os
import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from typing import List, Dict, Any, Optional, Callable
from pathlib import Path
import torchvision.transforms as T
from tqdm import tqdm

from datasets.progan import AdaptiveAIGCDataset
from strategies import BaseInputStrategy


__all__ = [
    'AIGCTestDataset',
    'CrossGeneratorEvaluator',
    'all_gather_tensor',
    'all_gather_list',
]


def all_gather_tensor(tensor: torch.Tensor, world_size: int, rank: int) -> torch.Tensor:
    """Gather tensors from all GPUs in DDP training

    Args:
        tensor: Local tensor to gather from current rank
        world_size: Total number of processes
        rank: Current process rank

    Returns:
        Concatenated tensor from all ranks
    """
    tensor_list = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(tensor_list, tensor)
    return torch.cat(tensor_list, dim=0)


def all_gather_list(data: List[Any], world_size: int, rank: int) -> List[Any]:
    """Gather lists from all GPUs in DDP training

    Args:
        data: Local list to gather from current rank
        world_size: Total number of processes
        rank: Current process rank

    Returns:
        Concatenated list from all ranks
    """
    gathered_data = [[] for _ in range(world_size)]
    dist.all_gather_object(gathered_data, data)

    # Flatten the list of lists
    result = []
    for rank_data in gathered_data:
        result.extend(rank_data)

    return result


class AIGCTestDataset(Dataset):
    """Dataset for cross-generator evaluation on AIGCTest"""

    def __init__(self,
                 root_dir: str,
                 strategy: BaseInputStrategy,
                 transform: Optional[Callable] = None,
                 test_generators: Optional[List[str]] = None,
                 balance_test: bool = False):
        """
        Initialize AIGCTest dataset

        Args:
            root_dir: Root directory of AIGCTest
            strategy: Input preprocessing strategy
            transform: Additional transforms to apply
            test_generators: List of generators to test (None for all)
            balance_test: Whether to balance samples across generators
        """
        self.root_dir = root_dir
        self.strategy = strategy

        # Default transforms for testing: Center crop and CLIP normalization
        if transform is None:
            self.transform = T.Compose([
                T.CenterCrop((224, 224)),
                T.ToTensor(),
                T.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                    inplace=True
                )
            ])
        else:
            self.transform = transform

        self.test_generators = test_generators
        self.balance_test = balance_test

        # Load dataset using AdaptiveAIGCDataset
        self.dataset = AdaptiveAIGCDataset(root_dir)

        # Filter and prepare test data
        self.test_data = self._prepare_test_data()

        print(f"AIGCTest dataset loaded: {len(self.test_data)} samples")

    def get_generator_list(self) -> List[str]:
        """Get list of all unique generators in the dataset"""
        return self.dataset.get_unique_generators()

    def _prepare_test_data(self) -> List[Dict[str, Any]]:
        """Prepare test data for evaluation"""
        test_data = []

        # Get all generator types
        all_generators = self.dataset.get_unique_generators()

        # Filter generators if specified
        if self.test_generators:
            test_generators = [g for g in all_generators if g in self.test_generators]
        else:
            test_generators = all_generators

        print(f"Evaluating on generators: {test_generators}")

        # Collect data for each generator
        for generator in test_generators:
            paths, labels = self.dataset.get_subset_by_generator(generator_name=generator)

            for path, label in zip(paths, labels):
                test_data.append({
                    'path': path,
                    'label': label,
                    'generator': generator
                })

        # Balance across generators if requested
        if self.balance_test:
            test_data = self._balance_across_generators(test_data)

        return test_data

    def _balance_across_generators(self, test_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Balance samples across different generators"""
        # Group by generator
        generator_data = {}
        for item in test_data:
            gen = item['generator']
            if gen not in generator_data:
                generator_data[gen] = []
            generator_data[gen].append(item)

        # Find minimum count
        min_count = min(len(data) for data in generator_data.values())

        # Sample equally from each generator
        balanced_data = []
        for gen, data in generator_data.items():
            # Shuffle and take min_count
            import random
            random.seed(42)  # For reproducibility
            sampled = random.sample(data, min_count)
            balanced_data.extend(sampled)

        print(f"Balanced dataset: {len(balanced_data)} samples ({min_count} per generator)")

        return balanced_data

    def __len__(self):
        return len(self.test_data)

    def __getitem__(self, idx):
        item = self.test_data[idx]

        # Load image with error handling for corrupted files
        try:
            image = Image.open(item['path']).convert('RGB')
        except (OSError, IOError, ValueError) as e:
            # Skip corrupted images - move to next valid image
            return self.__getitem__((idx + 1) % len(self.test_data))

        # Apply transform (resize, to tensor, normalize)
        image = self.transform(image)

        # Apply input strategy
        processed = self.strategy.preprocess(image)

        return {
            'image': processed,
            'label': torch.tensor(item['label'], dtype=torch.long),
            'generator': item['generator'],
            'path': item['path']
        }

    def get_generator_list(self) -> List[str]:
        """Get list of generators in test set"""
        return list(set(item['generator'] for item in self.test_data))

    def get_stats(self) -> Dict[str, Any]:
        """Get test dataset statistics"""
        generators = {}
        labels = {0: 0, 1: 0}

        for item in self.test_data:
            gen = item['generator']
            generators[gen] = generators.get(gen, 0) + 1
            labels[item['label']] += 1

        return {
            'total_samples': len(self.test_data),
            'generators': generators,
            'labels': labels,
            'real_samples': labels[0],
            'fake_samples': labels[1]
        }


def calculate_optimal_accuracy(y_true: List[int], y_probs: List[float]) -> tuple:
    """Calculate optimal accuracy by finding best threshold on ROC curve"""
    from sklearn.metrics import roc_curve

    # Get ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_probs)

    # Calculate accuracy for each threshold
    accuracies = []

    for threshold in thresholds:
        y_pred = [1 if prob >= threshold else 0 for prob in y_probs]
        accuracy = sum(1 for i in range(len(y_true)) if y_pred[i] == y_true[i]) / len(y_true)
        accuracies.append(accuracy)

    # Find optimal threshold (maximum accuracy)
    max_accuracy = max(accuracies)
    max_idx = accuracies.index(max_accuracy)
    optimal_threshold = thresholds[max_idx]

    return max_accuracy, optimal_threshold


class CrossGeneratorEvaluator:
    """Evaluate model performance across different AI generators"""

    def __init__(self,
                 model: torch.nn.Module,
                 device: torch.device,
                 output_dir: str,
                 rank: int = 0,
                 world_size: int = 1):
        """
        Initialize evaluator

        Args:
            model: Trained model
            device: Device to run evaluation on
            output_dir: Directory to save results
            rank: Process rank for DDP (default: 0)
            world_size: Total number of processes (default: 1)
        """
        self.model = model
        self.device = device
        self.output_dir = output_dir
        self.rank = rank
        self.world_size = world_size

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

    def evaluate(self, test_dataloader: DataLoader) -> Dict[str, Dict[str, float]]:
        """Evaluate model across generators with DDP support

        This method automatically gathers results from all GPUs when using DDP training,
        ensuring accurate metrics computed on the complete validation dataset.

        Args:
            test_dataloader: DataLoader for validation/test data

        Returns:
            Dictionary mapping generator names to evaluation metrics (only on rank 0)
        """
        self.model.eval()
        self.model.to(self.device)

        # 1. Create Generator String <-> ID Mapping
        # Access the underlying dataset to get all possible generator names
        if hasattr(test_dataloader.dataset, 'dataset'):
            # Handle Subset or other wrappers if necessary
            real_dataset = test_dataloader.dataset
        else:
            real_dataset = test_dataloader.dataset

        # Create mapping based on the full dataset's generator list
        unique_gens = sorted(list(set(real_dataset.get_generator_list())))
        gen_to_idx = {name: i for i, name in enumerate(unique_gens)}
        idx_to_gen = {i: name for i, name in enumerate(unique_gens)}

        # Store results per generator (per rank)
        local_predictions = []
        local_labels = []
        local_gen_indices = []  # Store INDICES, not strings
        local_probabilities = []

        with torch.no_grad():
            # Only show progress bar on rank 0 to avoid duplicate output
            dataloader = tqdm(test_dataloader, desc="Evaluating") if self.rank == 0 else test_dataloader

            for batch in dataloader:
                images = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)
                generators = batch['generator']  # These are strings

                # Forward pass
                outputs = self.model(images)
                probabilities = torch.softmax(outputs, dim=1)
                predicted_classes = torch.argmax(outputs, dim=1)

                # Store predictions and probabilities
                local_predictions.extend(predicted_classes.cpu().numpy())
                local_labels.extend(labels.cpu().numpy())
                local_probabilities.extend(probabilities[:, 1].cpu().numpy())

                # Convert strings to indices immediately
                batch_gen_indices = [gen_to_idx[g] for g in generators]
                local_gen_indices.extend(batch_gen_indices)

        # Gather results from all GPUs (if using DDP)
        if self.world_size > 1 and dist.is_available() and dist.is_initialized():
            # Convert lists to tensors for NCCL gathering (Much faster than pickling objects)
            predictions_tensor = torch.tensor(local_predictions, dtype=torch.int64, device=self.device)
            labels_tensor = torch.tensor(local_labels, dtype=torch.int64, device=self.device)
            probs_tensor = torch.tensor(local_probabilities, dtype=torch.float32, device=self.device)

            # Convert generator indices to tensor
            gen_indices_tensor = torch.tensor(local_gen_indices, dtype=torch.int64, device=self.device)

            all_predictions = all_gather_tensor(predictions_tensor, self.world_size, self.rank)
            all_labels = all_gather_tensor(labels_tensor, self.world_size, self.rank)
            all_probabilities = all_gather_tensor(probs_tensor, self.world_size, self.rank)
            all_gen_indices_tensor = all_gather_tensor(gen_indices_tensor, self.world_size, self.rank)

            # Convert back to numpy
            all_predictions = all_predictions.cpu().numpy()
            all_labels = all_labels.cpu().numpy()
            all_probabilities = all_probabilities.cpu().numpy()

            # Convert indices back to strings (Only needed on rank 0 usually)
            all_gen_indices_cpu = all_gen_indices_tensor.cpu().numpy()
            all_generators = [idx_to_gen[idx] for idx in all_gen_indices_cpu]

        else:
            # Single GPU
            all_predictions = local_predictions
            all_labels = local_labels
            all_probabilities = local_probabilities
            # Convert indices back to strings
            all_generators = [idx_to_gen[idx] for idx in local_gen_indices]

        # Only compute metrics on rank 0
        generator_results = {}
        if self.rank == 0:
            # Evaluate per generator
            unique_generators = list(set(all_generators))

            for generator in unique_generators:
                # Get indices for this generator
                indices = [i for i, g in enumerate(all_generators) if g == generator]

                gen_predictions = [all_predictions[i] for i in indices]
                gen_labels = [all_labels[i] for i in indices]
                gen_probs = [all_probabilities[i] for i in indices]

                # Calculate metrics
                from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, average_precision_score

                # Standard accuracy
                accuracy = accuracy_score(gen_labels, gen_predictions)

                # Precision, Recall, F1
                precision, recall, f1, _ = precision_recall_fscore_support(
                    gen_labels, gen_predictions, average='binary'
                )

                # AUC-ROC
                auc = roc_auc_score(gen_labels, gen_probs)

                # Average Precision (AP) - area under precision-recall curve
                ap = average_precision_score(gen_labels, gen_probs)

                # Optimal Accuracy
                optimal_acc, optimal_threshold = calculate_optimal_accuracy(gen_labels, gen_probs)

                generator_results[generator] = {
                    'accuracy': accuracy,
                    'optimal_accuracy': optimal_acc,
                    'optimal_threshold': optimal_threshold,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'auc': auc,
                    'average_precision': ap,
                    'num_samples': len(indices)
                }

        # Return results (skip barrier to avoid hanging - torchrun handles synchronization)
        return generator_results

    def save_results(self, results: Dict[str, Dict[str, float]], filename: str = "evaluation_results.txt", experiment_config: Optional[Dict[str, Any]] = None):
        """Save evaluation results to file

        Args:
            results: Dictionary of generator -> metrics
            filename: Output filename
            experiment_config: Optional experiment configuration to save
        """
        output_path = os.path.join(self.output_dir, filename)

        with open(output_path, 'w') as f:
            f.write("Cross-Generator Evaluation Results\n")
            f.write("=" * 60 + "\n\n")

            # Save experiment configuration if provided
            if experiment_config:
                f.write("Experiment Configuration\n")
                f.write("-" * 60 + "\n")
                for key, value in experiment_config.items():
                    if isinstance(value, dict):
                        f.write(f"{key}:\n")
                        for k, v in value.items():
                            f.write(f"  {k}: {v}\n")
                    else:
                        f.write(f"{key}: {value}\n")
                f.write("\n")

            for generator, metrics in results.items():
                f.write(f"Generator: {generator}\n")
                f.write(f"  Accuracy (0.5 threshold): {metrics['accuracy']:.4f}\n")
                f.write(f"  Optimal Accuracy: {metrics['optimal_accuracy']:.4f}\n")
                f.write(f"  Optimal Threshold: {metrics['optimal_threshold']:.4f}\n")
                f.write(f"  Precision: {metrics['precision']:.4f}\n")
                f.write(f"  Recall: {metrics['recall']:.4f}\n")
                f.write(f"  F1-Score: {metrics['f1']:.4f}\n")
                f.write(f"  AUC-ROC: {metrics['auc']:.4f}\n")
                f.write(f"  Average Precision (AP): {metrics['average_precision']:.4f}\n")
                f.write(f"  Num Samples: {metrics['num_samples']}\n")
                f.write("\n")

            # Overall statistics
            if results:  # Only calculate if we have results
                all_acc = [m['accuracy'] for m in results.values()]
                all_opt_acc = [m['optimal_accuracy'] for m in results.values()]
                all_f1 = [m['f1'] for m in results.values()]
                all_auc = [m['auc'] for m in results.values()]
                all_ap = [m['average_precision'] for m in results.values()]

                f.write("Overall Statistics\n")
                f.write("-" * 60 + "\n")
                f.write(f"Mean Accuracy (0.5): {sum(all_acc) / len(all_acc):.4f} ± {torch.std(torch.tensor(all_acc)):.4f}\n")
                f.write(f"Mean Optimal Accuracy: {sum(all_opt_acc) / len(all_opt_acc):.4f} ± {torch.std(torch.tensor(all_opt_acc)):.4f}\n")
                f.write(f"Mean F1-Score: {sum(all_f1) / len(all_f1):.4f} ± {torch.std(torch.tensor(all_f1)):.4f}\n")
                f.write(f"Mean AUC-ROC: {sum(all_auc) / len(all_auc):.4f} ± {torch.std(torch.tensor(all_auc)):.4f}\n")
                f.write(f"Mean Average Precision: {sum(all_ap) / len(all_ap):.4f} ± {torch.std(torch.tensor(all_ap)):.4f}\n")
                f.write("\n")

                # Best and worst performing generators
                best_ap_gen = max(results.keys(), key=lambda g: results[g]['average_precision'])
                worst_ap_gen = min(results.keys(), key=lambda g: results[g]['average_precision'])

                f.write(f"Best Generator (AP): {best_ap_gen} ({results[best_ap_gen]['average_precision']:.4f})\n")
                f.write(f"Worst Generator (AP): {worst_ap_gen} ({results[worst_ap_gen]['average_precision']:.4f})\n")
            else:
                f.write("Overall Statistics\n")
                f.write("-" * 60 + "\n")
                f.write("No results available - empty evaluation dataset or evaluation failed.\n")

        print(f"Results saved to {output_path}")


def create_evaluation_dataloader(data_root: str,
                                 strategy: BaseInputStrategy,
                                 test_generators: Optional[List[str]] = None,
                                 batch_size: int = 32,
                                 num_workers: int = 4,
                                 transform: Optional[Callable] = None) -> DataLoader:
    """
    Create evaluation dataloader for AIGCTest

    Args:
        data_root: Root directory of AIGCTest
        strategy: Input preprocessing strategy
        test_generators: Generators to test
        batch_size: Batch size
        num_workers: Number of workers
        transform: Image transforms

    Returns:
        Evaluation dataloader
    """
    dataset = AIGCTestDataset(
        root_dir=data_root,
        strategy=strategy,
        transform=transform,
        test_generators=test_generators,
        balance_test=False
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return dataloader