"""Evaluate FORTE (ICLR 2025) on OpenOOD benchmarks.

FORTE uses per-point PRDC features from multiple foundation model
embeddings (CLIP, ViTMSN, DINOv2) as input to a GMM classifier.

Unlike standard OpenOOD postprocessors, FORTE requires fusing ID test
and OOD test samples together before computing PRDC features. This
script handles that evaluation flow correctly.

Usage:
    python scripts/eval_ood_forte.py --id-data cifar10 --data-root ./data
"""
import os
import sys

ROOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.append(ROOT_DIR)

import argparse

import torch

from openood.evaluation_api.datasets import (DATA_INFO, data_setup,
                                              get_id_ood_dataloader)
from openood.evaluation_api.postprocessor import get_postprocessor
from openood.evaluation_api.preprocessor import get_default_preprocessor

parser = argparse.ArgumentParser(
    description='Evaluate FORTE on OpenOOD benchmarks')
parser.add_argument('--id-data', type=str, default='cifar10',
                    choices=list(DATA_INFO.keys()))
parser.add_argument('--data-root', type=str, default='./data')
parser.add_argument('--config-root', type=str, default='./configs')
parser.add_argument('--batch-size', type=int, default=200,
                    help='Batch size for OpenOOD data loaders (label reading)')
parser.add_argument('--save-csv', action='store_true')
parser.add_argument('--save-dir', type=str, default='./results/forte')
args = parser.parse_args()

# Ensure data is available
data_setup(args.data_root, args.id_data)

# OpenOOD preprocessor (needed for DataLoader construction;
# FORTE uses its own model-specific processors for feature extraction)
preprocessor = get_default_preprocessor(args.id_data)

# Build data loaders
dataloader_dict = get_id_ood_dataloader(
    args.id_data, args.data_root, preprocessor,
    batch_size=args.batch_size, shuffle=False, num_workers=4)

# Build FORTE postprocessor from config
postprocessor = get_postprocessor(args.config_root, 'forte', args.id_data)

# FORTE does not use the benchmark backbone — pass a dummy network
dummy_net = torch.nn.Linear(1, 1)

# Setup: load foundation models, extract train/test features, train GMM
postprocessor.setup(dummy_net, dataloader_dict['id'], dataloader_dict['ood'])

# Evaluate with fused ID+OOD (FORTE's mixed-batch evaluation)
metrics = postprocessor.eval_ood_forte(dataloader_dict)

# Save results
if args.save_csv:
    os.makedirs(args.save_dir, exist_ok=True)
    save_path = os.path.join(args.save_dir, f'forte_{args.id_data}.csv')
    metrics.to_csv(save_path, float_format='{:.2f}'.format)
    print(f'\nSaved results to {save_path}')
