"""
ProGAN dataset loader for training AI detection models.

This module provides dataset classes for loading and processing ProGAN
training data with support for different input strategies using plain PyTorch.
"""

import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from typing import List, Tuple, Optional, Callable, Dict, Any
from pathlib import Path
import torchvision.transforms as T

from strategies import BaseInputStrategy


class AdaptiveAIGCDataset(Dataset):
    """Dataset loader adapted from noise_detection.py"""
    IMG_EXTS = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp')

    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.image_paths = []
        self.labels = []  # 0 for real, 1 for fake
        self.generator_types = []

        print(f"Loading dataset from {root_dir}...")

        # Find all 0_real and 1_fake directories recursively
        real_dirs, fake_dirs = [], []
        for dirpath, dirnames, _ in os.walk(root_dir):
            names = {d.strip() for d in dirnames}
            if '0_real' in names:
                real_dirs.append(os.path.join(dirpath, '0_real'))
            if '1_fake' in names:
                fake_dirs.append(os.path.join(dirpath, '1_fake'))

        # Handle case where root_dir itself is 0_real or 1_fake
        base = os.path.basename(os.path.normpath(root_dir))
        if base == '0_real':
            real_dirs.append(root_dir)
        elif base == '1_fake':
            fake_dirs.append(root_dir)

        # Collect images
        def collect_from(d, label):
            try:
                for name in os.listdir(d):
                    if name.lower().endswith(self.IMG_EXTS):
                        img_path = os.path.join(d, name)
                        self.image_paths.append(img_path)
                        self.labels.append(label)
                        # Extract generator type from path
                        path_parts = Path(img_path).parts
                        generator_type = "unknown"
                        for part in path_parts:
                            if part in ['progan', 'biggan', 'cyclegan', 'DALLE2', 'gaugan', 'Glide', 'ADM',
                                       'Midjourney', 'stable_diffusion_v_1_4', 'stable_diffusion_v_1_5',
                                       'stargan', 'stylegan', 'stylegan2', 'VQDM', 'whichfaceisreal', 'wukong']:
                                generator_type = part
                                break
                        self.generator_types.append(generator_type)
            except Exception as e:
                print(f"Warning: Cannot read directory {d}: {e}")

        for d in real_dirs:
            collect_from(d, 0)
        for d in fake_dirs:
            collect_from(d, 1)

        print(f"Dataset loaded: {len(self.image_paths)} images ({sum(self.labels)} fake, {len(self.labels)-sum(self.labels)} real)")

        # Show statistics per generator
        unique_generators = list(set(self.generator_types))
        print("Images per generator:")
        for gen in unique_generators:
            count = self.generator_types.count(gen)
            fake_count = sum(1 for i, g in enumerate(self.generator_types) if g == gen and self.labels[i] == 1)
            real_count = count - fake_count
            print(f"  {gen}: {count} total ({fake_count} fake, {real_count} real)")

    def get_unique_generators(self):
        """Get list of unique generator types"""
        return list(set(self.generator_types))

    def get_subset_by_generator(self, generator_name=None, max_samples=None, random_shuffle=True):
        """Get subset of data by generator type"""
        if generator_name is None:
            # Return all data
            indices = list(range(len(self.image_paths)))
        else:
            # Filter by exact generator name matching (like noise_detection.py)
            indices = [i for i, gen in enumerate(self.generator_types) if gen == generator_name]

        if max_samples and len(indices) > max_samples:
            if random_shuffle:
                np.random.shuffle(indices)
            indices = indices[:max_samples]

        return [self.image_paths[i] for i in indices], [self.labels[i] for i in indices]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        generator = self.generator_types[idx]

        # Load image
        image = Image.open(img_path).convert('RGB')

        return {
            'image': image,
            'label': torch.tensor(label, dtype=torch.long),
            'generator': generator,
            'path': img_path
        }


class ProGANTrainingDataset(Dataset):
    """ProGAN training dataset with input strategy support"""

    def __init__(self,
                 root_dir: str,
                 strategy: BaseInputStrategy,
                 categories: Optional[List[str]] = None,
                 transform: Optional[Callable] = None,
                 generator_filter: Optional[str] = None,
                 max_samples_per_generator: Optional[int] = None,
                 balance_classes: bool = False):
        """
        Initialize ProGAN training dataset

        Args:
            root_dir: Root directory of ProGAN data
            strategy: Input preprocessing strategy
            categories: List of categories to include (None for all)
            transform: Additional transforms to apply
            generator_filter: Filter by specific generator
            max_samples_per_generator: Max samples per generator type
            balance_classes: Whether to balance real/fake samples
        """
        self.root_dir = root_dir
        self.strategy = strategy
        self.categories = categories or os.listdir(root_dir)

        # Default transforms for training: Random crop, flip, and CLIP normalization
        if transform is None:
            self.transform = T.Compose([
                T.RandomCrop(size=[224, 224],pad_if_needed=True),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
        else:
            self.transform = transform

        self.generator_filter = generator_filter
        self.max_samples_per_generator = max_samples_per_generator
        self.samples = self._load_samples()
        self.balance_classes = balance_classes

        # Apply generator filter and max samples limit
        self._filter_and_balance()

        print(f"Loaded {len(self.samples)} samples for training")

    def _load_samples(self) -> List[Dict[str, Any]]:
        """Load all samples from dataset"""
        samples = []

        for category in self.categories:
            if not os.path.isdir(os.path.join(self.root_dir, category)):
                continue

            for label_dir in ['0_real', '1_fake']:
                full_dir = os.path.join(self.root_dir, category, label_dir)
                if not os.path.exists(full_dir):
                    continue

                label_val = int(label_dir[0])  # 0 for real, 1 for fake

                for img_name in os.listdir(full_dir):
                    if img_name.lower().endswith(AdaptiveAIGCDataset.IMG_EXTS):
                        img_path = os.path.join(full_dir, img_name)

                        # Extract generator type
                        generator = self._extract_generator(img_path)

                        samples.append({
                            'path': img_path,
                            'label': label_val,
                            'category': category,
                            'generator': generator
                        })

        return samples

    def _extract_generator(self, img_path: str) -> str:
        """Extract generator type from image path"""
        path_parts = Path(img_path).parts
        generators = ['progan','progan_train', 'biggan', 'cyclegan', 'DALLE2', 'gaugan', 'Glide', 'ADM',
                     'Midjourney', 'stable_diffusion_v_1_4', 'stable_diffusion_v_1_5',
                     'stargan', 'stylegan', 'stylegan2', 'VQDM', 'whichfaceisreal', 'wukong']

        for part in path_parts:
            if part in generators:
                return part
        return 'unknown'

    def _filter_and_balance(self):
        """Apply filters and balance dataset"""
        # Filter by generator if specified
        if self.generator_filter:
            self.samples = [s for s in self.samples if s['generator'] == self.generator_filter]

        # Limit samples per generator
        if self.max_samples_per_generator:
            generator_samples = {}
            for sample in self.samples:
                gen = sample['generator']
                if gen not in generator_samples:
                    generator_samples[gen] = []
                generator_samples[gen].append(sample)

            balanced_samples = []
            for gen, gen_samples in generator_samples.items():
                if len(gen_samples) > self.max_samples_per_generator:
                    np.random.shuffle(gen_samples)
                    balanced_samples.extend(gen_samples[:self.max_samples_per_generator])
                else:
                    balanced_samples.extend(gen_samples)

            self.samples = balanced_samples

        # Balance real/fake if requested
        if self.balance_classes:
            real_samples = [s for s in self.samples if s['label'] == 0]
            fake_samples = [s for s in self.samples if s['label'] == 1]

            min_count = min(len(real_samples), len(fake_samples))

            if len(real_samples) > min_count:
                np.random.shuffle(real_samples)
                real_samples = real_samples[:min_count]

            if len(fake_samples) > min_count:
                np.random.shuffle(fake_samples)
                fake_samples = fake_samples[:min_count]

            self.samples = real_samples + fake_samples
            np.random.shuffle(self.samples)

        print(f"After filtering/balancing: {len(self.samples)} samples "
              f"({sum(1 for s in self.samples if s['label'] == 1)} fake, "
              f"{sum(1 for s in self.samples if s['label'] == 0)} real)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Load image
        image = Image.open(sample['path']).convert('RGB')

        # Apply transform (resize, to tensor, normalize)
        image = self.transform(image)

        # Apply input strategy
        processed = self.strategy.preprocess(image)

        return {
            'image': processed,
            'label': torch.tensor(sample['label'], dtype=torch.long),
            'category': sample['category'],
            'generator': sample['generator'],
            'path': sample['path']
        }


class ProGANDataloader:
    """Plain PyTorch dataloader wrapper for ProGAN datasets with multi-GPU support"""

    def __init__(self,
                 data_root: str,
                 strategy: BaseInputStrategy,
                 batch_size: int = 32,
                 val_split: float = 0.2,
                 num_workers: int = 4,
                 transform=None,
                 pin_memory: bool = True,
                 persistent_workers: bool = True,
                 **kwargs):
        """
        Initialize ProGAN dataloader

        Args:
            data_root: Root directory of dataset
            strategy: Input preprocessing strategy
            batch_size: Batch size for dataloaders
            val_split: Validation split ratio
            num_workers: Number of workers for dataloaders
            transform: Image transforms
            pin_memory: Whether to pin memory for faster GPU transfer
            persistent_workers: Whether to keep workers alive between epochs
            **kwargs: Additional arguments for dataset
        """
        self.data_root = data_root
        self.strategy = strategy
        self.batch_size = batch_size
        self.val_split = val_split
        self.num_workers = num_workers
        self.transform = transform
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.kwargs = kwargs

    def create_dataloaders(self):
        """Create train and validation dataloaders"""
        # Load full dataset
        full_dataset = ProGANTrainingDataset(
            root_dir=self.data_root,
            strategy=self.strategy,
            transform=self.transform,
            **self.kwargs
        )

        # Split into train/val
        total_samples = len(full_dataset)
        val_size = int(total_samples * self.val_split)
        train_size = total_samples - val_size

        indices = torch.randperm(total_samples).tolist()
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]

        train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
        val_dataset = torch.utils.data.Subset(full_dataset, val_indices)

        # Create samplers for distributed training if needed
        train_sampler = None
        val_sampler = None

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                train_dataset, shuffle=True
            )
            val_sampler = torch.utils.data.distributed.DistributedSampler(
                val_dataset, shuffle=False
            )

        # Create dataloaders
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            sampler=train_sampler,
            shuffle=(train_sampler is None),
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers
        )

        val_dataloader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            sampler=val_sampler,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers
        )

        return train_dataloader, val_dataloader, self.get_dataset_stats(full_dataset)

    def get_dataset_stats(self, dataset: Dataset) -> Dict[str, Any]:
        """Get dataset statistics"""
        if isinstance(dataset, torch.utils.data.Subset):
            # Get original dataset
            full_dataset = dataset.dataset
            all_samples = full_dataset.samples
        else:
            all_samples = dataset.samples

        generators = {}
        labels = {0: 0, 1: 0}
        categories = {}

        for sample in all_samples:
            gen = sample['generator']
            generators[gen] = generators.get(gen, 0) + 1
            labels[sample['label']] += 1

            cat = sample['category']
            categories[cat] = categories.get(cat, 0) + 1

        return {
            'total_samples': len(all_samples),
            'real_samples': labels[0],
            'fake_samples': labels[1],
            'generators': generators,
            'categories': categories
        }