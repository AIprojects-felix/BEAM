"""Checks for the manuscript cohort counts and training/inference conventions."""

import unittest

import numpy as np
import torch
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset

from data.samplers import MergeSingletonBatchSampler
from models import BEAM
from train import run_one_epoch, split_discovery_cohort


class ManuscriptAlignmentTests(unittest.TestCase):
    def test_discovery_counts_and_patient_isolation(self):
        labels = np.r_[np.zeros(204), np.ones(312)]
        parts = split_discovery_cohort(labels)
        self.assertEqual(tuple(map(len, parts)), (361, 52, 103))
        self.assertEqual(len(set().union(*map(set, parts))), 516)
        self.assertEqual(sum(map(len, parts)), 516)
        self.assertEqual(parts, split_discovery_cohort(labels))
        for part in parts:
            self.assertEqual(set(labels[part]), {0, 1})

    def test_batches_keep_all_patients_without_singletons(self):
        for n in (2, 4, 5, 8, 9, 360, 361):
            with self.subTest(n=n):
                sampler = MergeSingletonBatchSampler(SequentialSampler(range(n)), 4)
                batches = list(sampler)
                self.assertEqual([i for b in batches for i in b], list(range(n)))
                self.assertEqual(len(sampler), len(batches))
                self.assertTrue(all(2 <= len(b) <= 5 for b in batches))
                # Exercise the BatchNorm operation that previously failed on
                # the final single patient, without training a full network.
                bn = torch.nn.BatchNorm1d(8).train()
                for batch in batches:
                    self.assertTrue(torch.isfinite(bn(torch.randn(len(batch), 8))).all())

    def test_classifier_dropout_does_not_change_transformer_dropout(self):
        model = BEAM(d_model=16, num_heads=2, num_layers=1, dropout=0.1)
        self.assertEqual(model.classifier[2].p, 0.3)
        self.assertEqual(model.transformer_layers[0].attn.dropout, 0.1)
        self.assertTrue(all(layer.p == 0.3 for layer in model.cfdna_encoder.modules()
                            if isinstance(layer, torch.nn.Dropout)))

    def test_probability_at_threshold_is_positive(self):
        class HalfProbabilityModel(torch.nn.Module):
            def forward(self, mri, cfdna):
                return torch.zeros(mri.shape[0], device=mri.device)

        loader = DataLoader(TensorDataset(
            torch.zeros(2, 3, 8, 8, 8), torch.zeros(2, 275), torch.tensor([0., 1.]),
        ), batch_size=2)
        metrics, _, probabilities = run_one_epoch(
            HalfProbabilityModel(), loader, torch.nn.BCEWithLogitsLoss(), torch.device('cpu'),
        )
        np.testing.assert_array_equal(probabilities, [0.5, 0.5])
        self.assertEqual(metrics['sensitivity'], 1.0)
        self.assertEqual(metrics['specificity'], 0.0)


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
