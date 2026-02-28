"""
Main training script for AnyAttack decoder.

Usage:
    # Single GPU
    python train.py --config anyattack_decoder_vitl16

    # Multi-GPU with torchrun
    torchrun --nproc_per_node=4 train.py --config anyattack_decoder_vitl16

Author: Implementation Plan
"""

import os
import sys
import argparse
import json
from pathlib import Path

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
import torchvision.transforms as T

# Add release root directory to path
RELEASE_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, RELEASE_ROOT)

# Add anyattack_training directory to path for local imports (trainers, models, configs)
# (not needed since we moved code to release root)

from trainers.embedding_trainer import EmbeddingTrainer, setup_ddp, cleanup_ddp
from strategies.base import BaseInputStrategy
from configs.config import get_config, get_config_with_eps, get_config_with_margin, AIGCTEST_DATA_PATH


def create_strategy_from_config(config):
    """
    Create strategy instance from configuration.

    Args:
        config: Configuration dictionary with strategy settings

    Returns:
        Strategy instance (MultiScaleRawStrategy, LocalPixelDependencyStrategy, or None)
    """
    if not config.get('use_multi_scale', False):
        return None

    strategy_type = config.get('strategy_type', 'multi_scale_raw')

    if strategy_type == 'multi_scale_raw':
        from strategies.multi_scale_raw import MultiScaleRawStrategy
        return MultiScaleRawStrategy(
            levels=config.get('strategy_levels', [0]),
            smooth_sigma=config.get('strategy_smooth_sigma', 2.0),
            noise_std=config.get('strategy_noise_std', 0.1),
            normalize=False
        )
    elif strategy_type == 'multi_scale_raw_jpeg':
        from strategies.multi_scale_raw import MultiScaleRawJpegStrategy
        return MultiScaleRawJpegStrategy(
            levels=config.get('strategy_levels', [0]),
            smooth_sigma=config.get('strategy_smooth_sigma', 2.0),
            noise_std=config.get('strategy_noise_std', 0.1),
            jpeg_qualities=config.get('strategy_jpeg_qualities', (95, 75, 50)),
            normalize=False
        )
    elif strategy_type == 'multi_scale_raw_processed':
        from strategies.multi_scale_raw import MultiScaleRawProcessedStrategy
        return MultiScaleRawProcessedStrategy(
            levels=config.get('strategy_levels', [0]),
            smooth_sigma=config.get('strategy_smooth_sigma', 2.0),
            noise_std=config.get('strategy_noise_std', 0.1),
            normalize=False
        )
    else:
        raise ValueError(f"Unknown strategy type: {strategy_type}")


def get_strategy_channels(config):
    """
    Get number of output channels from strategy configuration.

    Args:
        config: Configuration dictionary with strategy settings

    Returns:
        Number of output channels
    """
    if not config.get('use_multi_scale', False):
        return 0

    # Create strategy instance and get its output channels
    strategy = create_strategy_from_config(config)
    if strategy is not None:
        return strategy.get_output_channels()
    return 0

# Import existing dataset classes
from datasets.progan import ProGANTrainingDataset
from datasets.aigctest import AIGCTestDataset


class InputStrategy(BaseInputStrategy):
    """Simple input strategy for compatibility with ProGANTrainingDataset"""

    def preprocess(self, image):
        """Return image without modification"""
        return image

    def get_name(self) -> str:
        return "input_identity"

    def get_output_channels(self) -> int:
        return 3

    def get_model_type(self) -> str:
        return "foundation"


def create_dataloaders(config, rank, world_size, strategy=None):
    """
    Create train and validation dataloaders.

    Train: ProGAN training data with random crop/flip
    Validation: AIGCTestset with center crop (no flip)

    Args:
        config: Experiment configuration
        rank: Process rank for DDP
        world_size: Total number of processes
        strategy: Optional RFNT strategy for multi-scale inputs

    Returns:
        train_loader, val_loader
    """
    # ========================================================================
    # Training dataset: ProGAN with random augmentations
    # ========================================================================
    train_transform = T.Compose([
        T.RandomCrop(size=[224, 224], pad_if_needed=True),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet stats
    ])

    # Use simple InputStrategy for dataset (returns 3-channel images)
    # Multi-scale strategy is applied in training loop, not in dataset
    dataset_strategy = InputStrategy()

    train_dataset = ProGANTrainingDataset(
        root_dir=config['progan_train_data_path'],
        strategy=dataset_strategy,
        transform=train_transform,
        balance_classes=False
    )

    # ========================================================================
    # Validation dataset: AIGCTestset with center crop
    # ========================================================================
    # AIGCTest uses CenterCrop + ImageNet normalization (same as training)
    val_transform = T.Compose([
        T.CenterCrop((224, 224)),
        T.ToTensor(),
        T.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet stats
            std=[0.229, 0.224, 0.225]
        )
    ])

    val_dataset = AIGCTestDataset(
        root_dir=AIGCTEST_DATA_PATH,
        strategy=dataset_strategy,  # Use simple InputStrategy (returns 3-channel images)
        transform=val_transform,
        test_generators=None,  # All generators
        balance_test=False
    )

    # ========================================================================
    # Create samplers for distributed training
    # ========================================================================
    train_sampler = None
    val_sampler = None

    if world_size > 1:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True
        )
        val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False
        )

    # ========================================================================
    # Create dataloaders
    # ========================================================================
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        sampler=val_sampler,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    if rank == 0:
        print(f"\nDataset info:")
        print(f"  Train samples: {len(train_dataset)}")
        print(f"  Val samples: {len(val_dataset)}")

        # Show validation generators
        val_gens = val_dataset.get_generator_list()
        print(f"  Validation generators: {val_gens}")

        # Show stats
        stats = val_dataset.get_stats()
        print(f"  Val real samples: {stats['real_samples']}")
        print(f"  Val fake samples: {stats['fake_samples']}")
        print(f"  Generators distribution:")
        for gen, count in stats.get('generators', {}).items():
            print(f"    {gen}: {count}")

    return train_loader, val_loader


def validate_only(config, checkpoint_path):
    """Validation-only mode - uses same initialization as training but skips training loop."""
    # Setup DDP (same as train())
    rank, world_size, local_rank = setup_ddp()

    # Set device
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{local_rank}')
    else:
        device = torch.device('cpu')

    if rank == 0:
        print(f"\n{'='*60}")
        print(f"VALIDATION-ONLY MODE")
        print(f"Config: {config['name']}")
        print(f"Checkpoint: {checkpoint_path}")
        print(f"{'='*60}")
        print(f"\nConfiguration:")
        for key, value in sorted(config.items()):
            if key not in ['progan_train_data_path']:
                print(f"  {key}: {value}")
        print(f"\nWorld size: {world_size}")
        print(f"Device: {device}")
        print(f"{'='*60}\n")

    # Create strategy from config if use_multi_scale is enabled (same as train())
    strategy = None
    if config.get('use_multi_scale', False):
        strategy = create_strategy_from_config(config)
        if rank == 0:
            print(f"Created RFNT strategy: {config.get('strategy_type', 'unknown')}")
            print(f"  Strategy channels: {get_strategy_channels(config)}")

    # Create low-level feature strategy for scratch branch (same as train())
    lpd_strategy = None
    if config.get('use_lpd_strategy', False):
        lpd_type = config.get('lpd_strategy_type', 'lpd')
        if lpd_type == 'lpd':
            from strategies.median_filter import LocalPixelDependencyStrategy
            lpd_strategy = LocalPixelDependencyStrategy(
                kernel_sizes=config.get('lpd_strategy_kernel_sizes', [3]),
                normalize=False
            )
        elif lpd_type == 'gaussian_diff':
            from strategies.low_level_features import GaussianDifferenceStrategy
            lpd_strategy = GaussianDifferenceStrategy(
                sigmas=config.get('lpd_strategy_gaussian_sigmas', [1.0]),
                kernel_size=config.get('lpd_strategy_gaussian_kernel_size', 5),
                normalize=False
            )
        elif lpd_type == 'resize_diff':
            from strategies.low_level_features import ResizeDifferenceStrategy
            lpd_strategy = ResizeDifferenceStrategy(
                scales=config.get('lpd_strategy_resize_scales', [0.5]),
                mode=config.get('lpd_strategy_resize_mode', 'bilinear'),
                normalize=False
            )
        elif lpd_type == 'jpeg_diff':
            from strategies.low_level_features import JpegDifferenceStrategy
            lpd_strategy = JpegDifferenceStrategy(
                qualities=config.get('lpd_strategy_jpeg_qualities', [95, 75, 50]),
                normalize=False
            )
        elif lpd_type == 'lpd_jpeg_diff':
            from strategies.low_level_features import LpdJpegDifferenceStrategy
            lpd_strategy = LpdJpegDifferenceStrategy(
                lpd_kernel_sizes=config.get('lpd_strategy_kernel_sizes', [3]),
                jpeg_qualities=config.get('lpd_strategy_jpeg_qualities', [95, 75, 50]),
                lpd_normalize=False,
                jpeg_normalize=False,
            )
        else:
            raise ValueError(f"Unknown lpd_strategy_type: {lpd_type}")

        if rank == 0:
            print(f"Created low-level strategy: {lpd_type}")
            if lpd_type == 'lpd':
                print(f"  Kernel sizes: {config.get('lpd_strategy_kernel_sizes', [3])}")
            print(f"  Channels: {lpd_strategy.get_output_channels()}")

    # Create dataloaders (same as train())
    train_loader, val_loader = create_dataloaders(config, rank, world_size, strategy)

    # Get decoder configuration (same as train())
    decoder_type = config.get('decoder_type', 'unet')
    decoder_kwargs = config.get('decoder_kwargs', {})
    use_multi_scale_decoder = config.get('use_multi_scale_decoder', config.get('use_multi_scale', False))

    if rank == 0:
        print(f"Decoder type: {decoder_type}")
        if decoder_kwargs:
            print(f"  Decoder kwargs: {decoder_kwargs}")

    # Create trainer (SAME AS train() - this is the key!)
    trainer = EmbeddingTrainer(
        model_name=config['model_name'],
        eps=config['attack_eps'],
        lr=config['lr'],
        weight_decay=config['weight_decay'],
        device=device,
        checkpoint_dir=config['checkpoint_dir'],
        rank=rank,
        world_size=world_size,
        decoder_type=decoder_type,
        decoder_kwargs=decoder_kwargs,
        strategy=strategy,
        use_multi_scale_decoder=use_multi_scale_decoder,
        lpd_strategy=lpd_strategy,
        loss_type=config.get('loss_type', 'similarity'),
        margin=config.get('margin', 0.1),
        fusion_method=config.get('fusion_method', 'logit_weighted'),
        eps_randomization=config.get('eps_randomization', False),
        eps_min=config.get('eps_min', config['attack_eps']),
        eps_max=config.get('eps_max', config['attack_eps']),
        eps_schedule=config.get('eps_schedule', 'random'),
        normalize_loss=config.get('normalize_loss', False),
        use_four_branch_ensemble=config.get('use_four_branch_ensemble', False),
        noise_embedding_use_l2=config.get('noise_embedding_use_l2', False),
        embedding_loss_weight=config.get('embedding_loss_weight', 1.0),
    )

    # Setup training mode (SAME AS train())
    trainer.setup_training_mode(
        mode=config['training_mode'],
        lambda_classification=config.get('lambda_classification', 0.1)
    )

    # Load checkpoint
    if rank == 0:
        print(f"\nLoading checkpoint from: {checkpoint_path}")
    trainer.load_checkpoint(checkpoint_path, load_optimizers=False)

    # Run validation only
    if rank == 0:
        print(f"\nRunning validation...")

    val_metrics = trainer.validate(val_loader, epoch=0)

    if rank == 0:
        print(f"\nValidation metrics:")
        for key, value in val_metrics.items():
            print(f"  {key}: {value:.4f}")

    # Cleanup
    cleanup_ddp()


def train(config):
    """Main training function"""
    # Setup DDP
    rank, world_size, local_rank = setup_ddp()

    # Set device
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{local_rank}')
    else:
        device = torch.device('cpu')

    if rank == 0:
        print(f"\n{'='*60}")
        print(f"Training with config: {config['name']}")
        print(f"{'='*60}")
        print(f"\nConfiguration:")
        for key, value in sorted(config.items()):
            if key not in ['progan_train_data_path']:  # Skip long paths
                print(f"  {key}: {value}")
        print(f"\nWorld size: {world_size}")
        print(f"Device: {device}")
        print(f"{'='*60}\n")

    # Create strategy from config if use_multi_scale is enabled
    strategy = None
    if config.get('use_multi_scale', False):
        strategy = create_strategy_from_config(config)
        if rank == 0:
            print(f"Created RFNT strategy: {config.get('strategy_type', 'unknown')}")
            print(f"  Strategy channels: {get_strategy_channels(config)}")

    # Create low-level feature strategy for scratch branch (LPD/gaussian/resize)
    lpd_strategy = None
    if config.get('use_lpd_strategy', False):
        lpd_type = config.get('lpd_strategy_type', 'lpd')
        if lpd_type == 'lpd':
            from strategies.median_filter import LocalPixelDependencyStrategy
            lpd_strategy = LocalPixelDependencyStrategy(
                kernel_sizes=config.get('lpd_strategy_kernel_sizes', [3]),
                normalize=False
            )
        elif lpd_type == 'gaussian_diff':
            from strategies.low_level_features import GaussianDifferenceStrategy
            lpd_strategy = GaussianDifferenceStrategy(
                sigmas=config.get('lpd_strategy_gaussian_sigmas', [1.0]),
                kernel_size=config.get('lpd_strategy_gaussian_kernel_size', 5),
                normalize=False
            )
        elif lpd_type == 'resize_diff':
            from strategies.low_level_features import ResizeDifferenceStrategy
            lpd_strategy = ResizeDifferenceStrategy(
                scales=config.get('lpd_strategy_resize_scales', [0.5]),
                mode=config.get('lpd_strategy_resize_mode', 'bilinear'),
                normalize=False
            )
        elif lpd_type == 'jpeg_diff':
            from strategies.low_level_features import JpegDifferenceStrategy
            lpd_strategy = JpegDifferenceStrategy(
                qualities=config.get('lpd_strategy_jpeg_qualities', [95, 75, 50]),
                normalize=False
            )
        elif lpd_type == 'lpd_jpeg_diff':
            from strategies.low_level_features import LpdJpegDifferenceStrategy
            lpd_strategy = LpdJpegDifferenceStrategy(
                lpd_kernel_sizes=config.get('lpd_strategy_kernel_sizes', [3]),
                jpeg_qualities=config.get('lpd_strategy_jpeg_qualities', [95, 75, 50]),
                lpd_normalize=False,
                jpeg_normalize=False,
            )
        else:
            raise ValueError(f"Unknown lpd_strategy_type: {lpd_type}")

        if rank == 0:
            print(f"Created low-level strategy: {lpd_type}")
            if lpd_type == 'lpd':
                print(f"  Kernel sizes: {config.get('lpd_strategy_kernel_sizes', [3])}")
            elif lpd_type == 'gaussian_diff':
                print(f"  Sigmas: {config.get('lpd_strategy_gaussian_sigmas', [1.0])}")
                print(f"  Kernel size: {config.get('lpd_strategy_gaussian_kernel_size', 5)}")
            elif lpd_type == 'resize_diff':
                print(f"  Scales: {config.get('lpd_strategy_resize_scales', [0.5])}")
                print(f"  Mode: {config.get('lpd_strategy_resize_mode', 'bilinear')}")
            elif lpd_type == 'jpeg_diff':
                print(f"  JPEG qualities: {config.get('lpd_strategy_jpeg_qualities', [95, 75, 50])}")
            elif lpd_type == 'lpd_jpeg_diff':
                print(f"  Kernel sizes: {config.get('lpd_strategy_kernel_sizes', [3])}")
                print(f"  JPEG qualities: {config.get('lpd_strategy_jpeg_qualities', [95, 75, 50])}")
            print(f"  Channels: {lpd_strategy.get_output_channels()}")

    # Create dataloaders
    train_loader, val_loader = create_dataloaders(config, rank, world_size, strategy)

    # Get decoder configuration
    decoder_type = config.get('decoder_type', 'unet')
    decoder_kwargs = config.get('decoder_kwargs', {})
    use_multi_scale_decoder = config.get('use_multi_scale_decoder', config.get('use_multi_scale', False))

    if rank == 0:
        print(f"Decoder type: {decoder_type}")
        if decoder_kwargs:
            print(f"  Decoder kwargs: {decoder_kwargs}")

    # Create trainer
    trainer = EmbeddingTrainer(
        model_name=config['model_name'],
        eps=config['attack_eps'],
        lr=config['lr'],
        weight_decay=config['weight_decay'],
        device=device,
        checkpoint_dir=config['checkpoint_dir'],
        rank=rank,
        world_size=world_size,
        decoder_type=decoder_type,
        decoder_kwargs=decoder_kwargs,
        strategy=strategy,
        use_multi_scale_decoder=use_multi_scale_decoder,
        lpd_strategy=lpd_strategy,  # Add LPD strategy for ensemble
        loss_type=config.get('loss_type', 'similarity'),  # 'similarity' or 'discrepancy'
        margin=config.get('margin', 0.1),  # Margin for discrepancy loss
        fusion_method=config.get('fusion_method', 'logit_weighted'),  # Ensemble fusion method
        eps_randomization=config.get('eps_randomization', False),  # Epsilon randomization for domain generalization
        eps_min=config.get('eps_min', config['attack_eps']),  # Min epsilon for randomization
        eps_max=config.get('eps_max', config['attack_eps']),  # Max epsilon for randomization
        eps_schedule=config.get('eps_schedule', 'random'),  # Epsilon schedule
        normalize_loss=config.get('normalize_loss', False),  # Loss normalization for domain generalization
        use_four_branch_ensemble=config.get('use_four_branch_ensemble', False),  # Use 4-branch ensemble
        noise_embedding_use_l2=config.get('noise_embedding_use_l2', False),  # Add L2 branch to noise_embedding classifier
        embedding_loss_weight=config.get('embedding_loss_weight', 1.0),  # Weight for embedding loss
    )

    # Setup training mode
    trainer.setup_training_mode(
        mode=config['training_mode'],
        lambda_classification=config.get('lambda_classification', 0.1)
    )

    # Resume from checkpoint if specified
    start_epoch = 1
    if config.get('resume_checkpoint'):
        trainer.load_checkpoint(config['resume_checkpoint'])
        start_epoch = trainer.current_epoch + 1

    # Training loop
    niter = config['niter']
    save_every = config.get('save_every', 5)

    if rank == 0:
        print(f"\nStarting training for {niter} epochs...")
        print(f"Save checkpoint every {save_every} epochs")

    best_val_similarity = float('inf')  # Lower is better (we want to minimize similarity)
    train_metrics = {}
    val_metrics = {}

    for epoch in range(start_epoch, niter + 1):
        if rank == 0:
            print(f"\n{'='*60}")
            print(f"Epoch {epoch}/{niter}")
            print(f"{'='*60}")

        # Train
        train_metrics = trainer.train_epoch(train_loader, epoch)

        if rank == 0:
            print(f"\nTrain metrics:")
            for key, value in train_metrics.items():
                print(f"  {key}: {value:.4f}")

        # Validate
        val_metrics = trainer.validate(val_loader, epoch)

        if rank == 0:
            print(f"\nValidation metrics:")
            for key, value in val_metrics.items():
                print(f"  {key}: {value:.4f}")

        # Save checkpoint
        if epoch % save_every == 0:
            trainer.save_checkpoint(epoch)

        # Save best model
        if rank == 0:
            current_val_similarity = val_metrics.get('val/similarity', float('inf'))
            if current_val_similarity < best_val_similarity:
                best_val_similarity = current_val_similarity
                trainer.save_checkpoint(epoch, filename="checkpoint_best.pt")
                print(f"  New best model saved! (similarity: {best_val_similarity:.4f})")

    if rank == 0:
        print("\nTraining completed!")

        # Collect final training results
        final_results = {
            'total_epochs': niter,
            'best_val_similarity': float(best_val_similarity),
            'final_train_metrics': train_metrics,
            'final_val_metrics': val_metrics,
            'config': {
                'model_name': config['model_name'],
                'attack_eps': config['attack_eps'],
                'lr': config['lr'],
                'weight_decay': config['weight_decay'],
                'niter': config['niter'],
                'batch_size': config['batch_size'],
                'training_mode': config.get('training_mode', 'embedding_only'),
                'lambda_classification': config.get('lambda_classification', 0.0),
                'loss_type': config.get('loss_type', 'similarity'),
                'margin': config.get('margin', 0.1),
                'embedding_loss_weight': config.get('embedding_loss_weight', 1.0),
            },
            'checkpoint_dir': config['checkpoint_dir']
        }

        # Save results to JSON
        trainer.save_results(final_results)

    # Cleanup
    cleanup_ddp()


def main():
    parser = argparse.ArgumentParser(description='Train AnyAttack decoder')
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Configuration name (see configs/config.py)'
    )
    parser.add_argument(
        '--eps',
        type=float,
        default=None,
        help='Override epsilon value (e.g., 4/255, 8/255, etc.). Creates epsilon-specific checkpoint directory.'
    )
    parser.add_argument(
        '--margin',
        type=float,
        default=None,
        help='Override margin value for discrepancy loss (default: from config)'
    )
    parser.add_argument(
        '--niter',
        type=int,
        default=None,
        help='Override number of training iterations (default: from config, or 1 if --eps is specified)'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Resume from checkpoint path'
    )
    parser.add_argument(
        '--checkpoint-dir',
        type=str,
        default=None,
        help='Override checkpoint directory path'
    )
    parser.add_argument(
        '--fusion-method',
        type=str,
        default="logit_weighted",
        choices=['max', 'avg', 'sum', 'logit_weighted', 'learned_weight', 'product', 'attention'],
        help='Ensemble fusion method (default: logit_weighted)'
    )
    parser.add_argument(
        '--model-name',
        type=str,
        default=None,
        help='Override backbone model name (e.g., dinov3_vitl16, ViT-L-14)'
    )
    parser.add_argument(
        '--decoder-type',
        type=str,
        default=None,
        choices=['unet', 'simple'],
        help='Override decoder type (unet or simple)'
    )
    parser.add_argument(
        '--embedding-loss-weight',
        type=float,
        default=None,
        help='Override embedding loss weight (0.0 for BCE-only)'
    )
    parser.add_argument(
        '--use-lpd-strategy',
        action='store_true',
        help='Enable scratch branch low-level feature strategy'
    )
    parser.add_argument(
        '--lpd-strategy-type',
        type=str,
        default=None,
        choices=['lpd', 'gaussian_diff', 'resize_diff', 'jpeg_diff', 'lpd_jpeg_diff'],
        help='Low-level feature type for scratch branch'
    )
    parser.add_argument(
        '--lpd-kernel-sizes',
        type=str,
        default=None,
        help='Comma-separated kernel sizes for LPD (e.g., 3,5)'
    )
    parser.add_argument(
        '--lpd-gaussian-sigmas',
        type=str,
        default=None,
        help='Comma-separated sigmas for Gaussian diff (e.g., 1.0,2.0)'
    )
    parser.add_argument(
        '--lpd-resize-scales',
        type=str,
        default=None,
        help='Comma-separated scales for resize diff (e.g., 0.5,0.25)'
    )
    parser.add_argument(
        '--lpd-jpeg-qualities',
        type=str,
        default=None,
        help='Comma-separated JPEG qualities for JPEG diff (e.g., 95,75,50)'
    )
    parser.add_argument(
        '--lpd-jpeg-with-lpd',
        action='store_true',
        help='Use LPD + JPEG diff combined scratch strategy'
    )
    parser.add_argument(
        '--eps-randomization',
        action='store_true',
        help='Enable epsilon randomization for domain generalization (vary eps per batch)'
    )
    parser.add_argument(
        '--eps-min',
        type=float,
        default=4.0,
        help='Minimum epsilon value for randomization (default: 4.0, will be divided by 255)'
    )
    parser.add_argument(
        '--eps-max',
        type=float,
        default=64.0,
        help='Maximum epsilon value for randomization (default: 64.0, will be divided by 255)'
    )
    parser.add_argument(
        '--eps-schedule',
        type=str,
        default='random',
        choices=['random', 'cycle', 'inverse_cycle', 'linear_increase'],
        help='Epsilon randomization schedule (default: random per batch)'
    )
    parser.add_argument(
        '--normalize-loss',
        action='store_true',
        help='Enable loss normalization for domain generalization (scale-invariant loss)'
    )
    parser.add_argument(
        '--four-branch-ensemble',
        action='store_true',
        help='Use 4-branch ensemble (foundation + scratch + l2_distance + embedding_diff)'
    )
    parser.add_argument(
        '--validate-checkpoint',
        type=str,
        default=None,
        help='Path to checkpoint to validate (skip training, only run validation)'
    )

    args = parser.parse_args()

    # Get configuration
    config = get_config(args.config)

    # Override hyperparameters if specified
    if args.eps is not None:
        config['attack_eps'] = args.eps
    if args.margin is not None:
        config['margin'] = args.margin
    if args.niter is not None:
        config['niter'] = args.niter
    if args.checkpoint_dir is not None:
        config['checkpoint_dir'] = args.checkpoint_dir
    if args.fusion_method is not None:
        config['fusion_method'] = args.fusion_method
    if args.model_name is not None:
        config['model_name'] = args.model_name
    if args.decoder_type is not None:
        config['decoder_type'] = args.decoder_type
        if args.decoder_type != 'unet':
            config['use_multi_scale_decoder'] = False
            config['decoder_kwargs'] = {}
    if args.embedding_loss_weight is not None:
        config['embedding_loss_weight'] = args.embedding_loss_weight
    if args.use_lpd_strategy:
        config['use_lpd_strategy'] = True
    if args.lpd_strategy_type is not None:
        config['lpd_strategy_type'] = args.lpd_strategy_type
    if args.lpd_kernel_sizes is not None:
        config['lpd_strategy_kernel_sizes'] = [int(k.strip()) for k in args.lpd_kernel_sizes.split(',') if k.strip()]
    if args.lpd_gaussian_sigmas is not None:
        config['lpd_strategy_gaussian_sigmas'] = [float(s.strip()) for s in args.lpd_gaussian_sigmas.split(',') if s.strip()]
    if args.lpd_resize_scales is not None:
        config['lpd_strategy_resize_scales'] = [float(s.strip()) for s in args.lpd_resize_scales.split(',') if s.strip()]
    if args.lpd_jpeg_qualities is not None:
        config['lpd_strategy_jpeg_qualities'] = [int(q.strip()) for q in args.lpd_jpeg_qualities.split(',') if q.strip()]
    if args.lpd_jpeg_with_lpd:
        config['lpd_strategy_type'] = 'lpd_jpeg_diff'

    # Epsilon randomization settings
    if args.eps_randomization:
        config['eps_randomization'] = True
        config['eps_min'] = args.eps_min / 255.0
        config['eps_max'] = args.eps_max / 255.0
        config['eps_schedule'] = args.eps_schedule
    else:
        config['eps_randomization'] = False
        config['eps_min'] = config['attack_eps']
        config['eps_max'] = config['attack_eps']
        config['eps_schedule'] = 'random'

    # Loss normalization setting
    config['normalize_loss'] = args.normalize_loss

    # Four-branch ensemble setting
    config['use_four_branch_ensemble'] = args.four_branch_ensemble

    # Override with resume checkpoint if provided
    if args.resume:
        config['resume_checkpoint'] = args.resume

    # Validation-only mode
    if args.validate_checkpoint:
        validate_only(config, args.validate_checkpoint)
    else:
        # Train
        train(config)


if __name__ == "__main__":
    main()
