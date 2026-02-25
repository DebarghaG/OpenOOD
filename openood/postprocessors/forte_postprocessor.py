import os
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import sklearn.metrics
import torch
import torch.nn as nn
from PIL import Image
from sklearn.mixture import GaussianMixture
from torch.utils.data import DataLoader
from tqdm import tqdm

from openood.evaluators.metrics import compute_all_metrics

from .base_postprocessor import BasePostprocessor


class FortePostprocessor(BasePostprocessor):
    """FORTE (ICLR 2025) postprocessor for OOD detection.

    Uses per-point PRDC features from multiple foundation model embeddings
    (CLIP, ViTMSN, DINOv2) as input to a GMM classifier.

    IMPORTANT: FORTE requires fusing ID test + OOD test samples together
    before computing PRDC features. The standard OpenOOD evaluator processes
    them separately, so use eval_ood_forte() instead of the standard pipeline.
    """

    def __init__(self, config):
        super().__init__(config)
        self.args = self.config.postprocessor.postprocessor_args
        self.nearest_k = self.args.nearest_k
        self.seed = self.args.seed
        self.batch_size_extract = self.args.batch_size_extract

        self.forte_models = {}
        self.ref_features = {}
        self.id_test_features = {}
        self.id_test_labels = None
        self.n_id_test = 0
        self.classifier = None
        self.setup_flag = False
        self.device = None

        self.APS_mode = False
        self.hyperparam_search_done = True

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict):
        if self.setup_flag:
            return

        if torch.cuda.is_available():
            self.device = 'cuda'
        elif torch.backends.mps.is_available():
            self.device = 'mps'
        else:
            self.device = 'cpu'

        # 1. Initialize foundation models
        print('FORTE: Initializing foundation models...', flush=True)
        self._init_models()

        # 2. Extract features from ID training data
        print('FORTE: Extracting ID train features...', flush=True)
        id_train_paths = self._get_image_paths_from_loader(
            id_loader_dict['train'])
        id_train_features = self._extract_all_features(id_train_paths)

        # 3. Split ID train into part1 (reference) and part2 (train PRDC)
        rng = np.random.RandomState(self.seed)
        n_train = len(id_train_paths)
        indices = rng.permutation(n_train)
        split_idx = n_train // 2
        part1_idx, part2_idx = indices[:split_idx], indices[split_idx:]

        for model_name in self.forte_models:
            self.ref_features[model_name] = \
                id_train_features[model_name][part1_idx]

        part2_features = {
            m: id_train_features[m][part2_idx] for m in self.forte_models
        }

        # 4. Compute PRDC features for part2 against part1
        print('FORTE: Computing training PRDC features...', flush=True)
        train_prdc = self._compute_prdc_features(part2_features)

        # 5. Train GMM classifier
        print('FORTE: Training GMM classifier...', flush=True)
        self._train_classifier(train_prdc)

        # 6. Pre-extract ID test features
        print('FORTE: Extracting ID test features...', flush=True)
        id_test_paths = self._get_image_paths_from_loader(
            id_loader_dict['test'])
        self.id_test_features = self._extract_all_features(id_test_paths)
        self.n_id_test = len(id_test_paths)

        # 7. Store ID test labels
        self.id_test_labels = self._get_labels_from_loader(
            id_loader_dict['test'])

        self.setup_flag = True
        print('FORTE: Setup complete.', flush=True)

    # ------------------------------------------------------------------
    # Model initialization
    # ------------------------------------------------------------------

    def _init_models(self):
        from transformers import (AutoFeatureExtractor, AutoImageProcessor,
                                  AutoModel, CLIPModel, CLIPProcessor,
                                  ViTMSNModel)

        # CLIP
        clip_model = CLIPModel.from_pretrained(
            'openai/clip-vit-base-patch32').to(self.device)
        clip_processor = CLIPProcessor.from_pretrained(
            'openai/clip-vit-base-patch32')
        clip_model.eval()
        self.forte_models['clip'] = (clip_model, clip_processor)

        # ViTMSN
        vitmsn_model = ViTMSNModel.from_pretrained(
            'facebook/vit-msn-base').to(self.device)
        vitmsn_processor = AutoFeatureExtractor.from_pretrained(
            'facebook/vit-msn-base')
        vitmsn_model.eval()
        self.forte_models['vitmsn'] = (vitmsn_model, vitmsn_processor)

        # DINOv2
        dinov2_model = AutoModel.from_pretrained(
            'facebook/dinov2-base').to(self.device)
        dinov2_processor = AutoImageProcessor.from_pretrained(
            'facebook/dinov2-base')
        dinov2_model.eval()
        self.forte_models['dinov2'] = (dinov2_model, dinov2_processor)

    # ------------------------------------------------------------------
    # Image path and label extraction from OpenOOD data loaders
    # ------------------------------------------------------------------

    @staticmethod
    def _get_image_paths_from_loader(data_loader: DataLoader) -> List[str]:
        dataset = data_loader.dataset
        data_dir = dataset.data_dir
        n = len(dataset)
        paths = []
        for i in range(n):
            line = dataset.imglist[i].strip('\n')
            image_name = line.split(' ', 1)[0]
            paths.append(os.path.join(data_dir, image_name))
        return paths

    @staticmethod
    def _get_labels_from_loader(data_loader: DataLoader) -> np.ndarray:
        labels = []
        for batch in data_loader:
            labels.append(batch['label'])
        return torch.cat(labels).numpy().astype(int)

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------

    def _extract_all_features(
            self, image_paths: List[str]) -> Dict[str, np.ndarray]:
        all_features = {}
        for model_name, (model, processor) in self.forte_models.items():
            features = []
            for i in tqdm(range(0, len(image_paths), self.batch_size_extract),
                          desc=f'  {model_name}'):
                batch_paths = image_paths[i:i + self.batch_size_extract]
                images = [Image.open(p).convert('RGB') for p in batch_paths]

                inputs = processor(
                    images=images, return_tensors='pt', padding=True)
                inputs = {k: v.to(self.device) for k, v in inputs.items()
                          if isinstance(v, torch.Tensor)}

                with torch.no_grad():
                    if model_name == 'clip':
                        feat = model.get_image_features(**inputs)
                    else:
                        feat = model(**inputs).last_hidden_state[:, 0, :]

                features.append(feat.cpu().numpy())

            all_features[model_name] = np.concatenate(features, axis=0)

        return all_features

    # ------------------------------------------------------------------
    # PRDC computation (matches FORTE's prdc.py and prdc_per_point.py)
    # ------------------------------------------------------------------

    @staticmethod
    def _pairwise_distances(X, Y):
        """Euclidean pairwise distances, matching FORTE's prdc.py."""
        return sklearn.metrics.pairwise_distances(
            X, Y, metric='euclidean', n_jobs=8)

    @staticmethod
    def _knn_radii(features, k):
        """Compute k-th nearest-neighbour distance for each point.

        Matches FORTE's compute_nearest_neighbour_distances:
        uses k+1 because self-distance (0) is included.
        """
        dists = sklearn.metrics.pairwise_distances(
            features, metric='euclidean', n_jobs=8)
        # k+1 smallest values include self (dist=0)
        indices = np.argpartition(dists, k + 1, axis=-1)[..., :k + 1]
        k_smallests = np.take_along_axis(dists, indices, axis=-1)
        return k_smallests.max(axis=-1)

    def _compute_prdc_features(
            self, test_features: Dict[str, np.ndarray]) -> np.ndarray:
        """Compute per-point PRDC features across all models.

        test_features may be fused (ID test + OOD). The kNN distances
        within the test set depend on this composition -- this is
        FORTE's key design: mixed-batch evaluation.

        Returns array of shape (N_test, 12).
        """
        all_prdc = []

        for model_name in self.forte_models:
            ref = self.ref_features[model_name]
            fake = test_features[model_name]
            n_ref = ref.shape[0]

            # kth NN distances within reference and within test set
            real_nn_dist = self._knn_radii(ref, self.nearest_k)
            fake_nn_dist = self._knn_radii(fake, self.nearest_k)

            # Pairwise distances: ref (rows) vs fake (cols)
            dist_matrix = self._pairwise_distances(ref, fake)

            # Per-point metrics (matching prdc_per_point.py exactly)
            precision = (
                dist_matrix < real_nn_dist[:, np.newaxis]
            ).any(axis=0).astype(np.float32)

            recall = (
                dist_matrix < fake_nn_dist[np.newaxis, :]
            ).sum(axis=0).astype(np.float32) / n_ref

            density = (
                dist_matrix < real_nn_dist[:, np.newaxis]
            ).sum(axis=0).astype(np.float32) / (self.nearest_k * n_ref)

            coverage = (
                dist_matrix.min(axis=0) < fake_nn_dist
            ).astype(np.float32)

            # FORTE's get_prdc_features stacks: recall, density, precision, coverage
            all_prdc.append(np.column_stack(
                [recall, density, precision, coverage]))

        return np.concatenate(all_prdc, axis=1)

    # ------------------------------------------------------------------
    # Classifier training and scoring
    # ------------------------------------------------------------------

    def _train_classifier(self, X_train: np.ndarray):
        """Train GMM with hyperparameter tuning (matching FORTE's eval.py)."""
        best_n = None
        best_score = -np.inf

        for n_components in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
            if n_components > X_train.shape[0]:
                break
            gmm = GaussianMixture(
                n_components=n_components, random_state=self.seed)
            gmm.fit(X_train)
            score = gmm.score(X_train)
            if score > best_score:
                best_score = score
                best_n = n_components

        self.classifier = GaussianMixture(
            n_components=best_n, random_state=self.seed)
        self.classifier.fit(X_train)
        print(f'  Best n_components={best_n} (score={best_score:.4f})')

    def _score_samples(self, X: np.ndarray) -> np.ndarray:
        """Score samples. Higher = more likely ID."""
        return self.classifier.score_samples(X)

    # ------------------------------------------------------------------
    # postprocess / inference — NOT SUPPORTED for FORTE
    # ------------------------------------------------------------------

    def postprocess(self, net: nn.Module, data: Any):
        raise NotImplementedError(
            'FORTE is a set-level method and does not support per-batch '
            'postprocessing. Use eval_ood_forte() instead.')

    def inference(self, net: nn.Module, data_loader: DataLoader,
                  progress: bool = True):
        raise NotImplementedError(
            'FORTE requires fused ID+OOD evaluation. '
            'Use eval_ood_forte() instead.')

    # ------------------------------------------------------------------
    # FORTE-specific evaluation (handles fused ID+OOD correctly)
    # ------------------------------------------------------------------

    def eval_ood_forte(self, dataloader_dict: dict) -> pd.DataFrame:
        """Evaluate FORTE with proper fused test sets.

        For each OOD dataset, fuses ID test + OOD features before computing
        PRDC, preserving FORTE's mixed-batch evaluation semantics.

        Returns a DataFrame matching OpenOOD's Evaluator.eval_ood() format.
        """
        metrics_dict = {}

        for ood_split in ['near', 'far']:
            split_metrics = []
            for dataset_name, ood_dl in \
                    dataloader_dict['ood'][ood_split].items():
                print(f'FORTE eval: {dataset_name} ({ood_split}-ood)...',
                      flush=True)

                # 1. Extract OOD features
                ood_paths = self._get_image_paths_from_loader(ood_dl)
                ood_features = self._extract_all_features(ood_paths)
                n_ood = len(ood_paths)

                # 2. Fuse ID test + OOD (FORTE's key: mixed evaluation)
                fused_features = {}
                for m in self.forte_models:
                    fused_features[m] = np.concatenate([
                        self.id_test_features[m],
                        ood_features[m],
                    ], axis=0)

                # 3. Compute PRDC on fused set
                print(f'  Computing PRDC ({self.n_id_test} ID + {n_ood} OOD '
                      f'= {self.n_id_test + n_ood} fused)...', flush=True)
                prdc_features = self._compute_prdc_features(fused_features)

                # 4. Score all samples
                scores = self._score_samples(prdc_features)

                # 5. Split scores back to ID and OOD
                id_conf = scores[:self.n_id_test]
                ood_conf = scores[self.n_id_test:]

                # 6. Compute OpenOOD metrics
                conf = np.concatenate([id_conf, ood_conf])
                label = np.concatenate([
                    self.id_test_labels,
                    -1 * np.ones(n_ood, dtype=np.int32),
                ])
                # FORTE does not produce class predictions;
                # set id_pred = id_gt so ACC column is meaningful
                pred = np.concatenate([
                    self.id_test_labels.copy(),
                    np.zeros(n_ood, dtype=np.int32),
                ])

                ood_metrics = compute_all_metrics(conf, label, pred)
                split_metrics.append(ood_metrics)
                self._print_metrics(ood_metrics, dataset_name)

            # Mean across datasets in this split
            split_metrics = np.array(split_metrics)
            mean_metrics = np.mean(split_metrics, axis=0, keepdims=True)
            print('Mean metrics:', flush=True)
            split_name = 'nearood' if ood_split == 'near' else 'farood'
            self._print_metrics(list(mean_metrics[0]), split_name)
            metrics_dict[ood_split] = np.concatenate(
                [split_metrics, mean_metrics], axis=0) * 100

        # Build DataFrame matching Evaluator.eval_ood() format
        index_names = (
            list(dataloader_dict['ood']['near'].keys()) + ['nearood'] +
            list(dataloader_dict['ood']['far'].keys()) + ['farood']
        )
        all_metrics = np.concatenate(
            [metrics_dict['near'], metrics_dict['far']], axis=0)

        df = pd.DataFrame(
            all_metrics,
            index=index_names,
            columns=['FPR@95', 'AUROC', 'AUPR_IN', 'AUPR_OUT', 'ACC'],
        )

        with pd.option_context(
                'display.max_rows', None,
                'display.max_columns', None,
                'display.float_format', '{:,.2f}'.format):
            print('\nFORTE Results:')
            print(df)

        return df

    @staticmethod
    def _print_metrics(metrics, name=''):
        [fpr, auroc, aupr_in, aupr_out, _] = metrics
        print(f'  {name}: FPR@95: {100 * fpr:.2f}, AUROC: {100 * auroc:.2f}, '
              f'AUPR_IN: {100 * aupr_in:.2f}, AUPR_OUT: {100 * aupr_out:.2f}',
              flush=True)
        print('\u2500' * 70, flush=True)
