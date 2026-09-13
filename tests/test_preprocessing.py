"""Regression checks for training-only scaling and the manuscript feature layout."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from data import MultiModalDataset
from data.cfdna_features import build_cfdna_feature_vector
from data.cfdna_scaling import scaler_from_checkpoint, scaler_to_state
from models import BEAM
from predict import load_patient, infer


class PreprocessingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.values = np.random.default_rng(17).normal(size=(40, 275))
        self.values[:, 0] = 5  # constant feature must retain a valid unit scale
        for i, features in enumerate(self.values):
            pdir = self.root / f'p{i}'
            pdir.mkdir()
            for name in ('T1', 'T2', 'DWI'):
                np.save(pdir / f'{name}.npy', np.ones((8, 8, 8), dtype=np.float32))
            np.save(pdir / 'cfdna_features.npy', features)
        self.labels = self.root / 'labels.csv'
        pd.DataFrame({'patient_id': [f'p{i}' for i in range(40)],
                      'label': [i % 2 for i in range(40)]}).to_csv(self.labels, index=False)

    def tearDown(self):
        self.tmp.cleanup()

    def dataset(self, scaler=None):
        return MultiModalDataset(self.root, self.labels, target_shape=(8, 8, 8),
                                 cfdna_scaler=scaler)

    def test_held_out_values_never_affect_fit(self):
        ds = self.dataset()
        self.assertIsNone(ds.cfdna_scaler)
        with self.assertRaises(ValueError):
            ds[0]
        train_idx = [0, 2, 5, 8]
        expected = StandardScaler().fit(self.values[train_idx])
        # Change an evaluation patient's distribution before fitting.
        np.save(self.root / 'p39/cfdna_features.npy', np.full(275, 1e9))
        scaler = ds.fit_cfdna_scaler(train_idx)
        np.testing.assert_allclose(scaler.mean_, expected.mean_)
        np.testing.assert_allclose(scaler.scale_, expected.scale_)
        self.assertEqual(int(scaler.n_samples_seen_), len(train_idx))
        with self.assertRaises(ValueError):
            ds.fit_cfdna_scaler([39])

    def test_checkpoint_roundtrip_and_prediction_input_parity(self):
        scaler = self.dataset().fit_cfdna_scaler([0, 1, 2, 3])
        path = self.root / 'scaler.pth'
        torch.save({'cfdna_scaler': scaler_to_state(scaler)}, path)
        frozen = scaler_from_checkpoint(torch.load(path, weights_only=True), 275)
        ds = self.dataset(frozen)
        mri, cfdna, _ = ds[10]
        pred_mri, pred_cfdna = load_patient(self.root / 'p10', (8, 8, 8), frozen)
        np.testing.assert_array_equal(cfdna.numpy(), pred_cfdna)
        np.testing.assert_array_equal(mri.numpy(), pred_mri)
        expected = scaler.transform(self.values[10:11])[0]
        np.testing.assert_allclose(pred_cfdna, expected, rtol=1e-6, atol=1e-6)
        before = pred_cfdna.copy()
        np.save(self.root / 'p39/cfdna_features.npy', np.full(275, -1e9))
        np.testing.assert_array_equal(self.dataset(frozen)[10][1].numpy(), before)

    def test_missing_or_incompatible_scaler_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'no training-fitted'):
            scaler_from_checkpoint({}, 275)
        scaler = StandardScaler().fit(self.values)
        with self.assertRaises(ValueError):
            scaler_from_checkpoint({'cfdna_scaler': scaler_to_state(scaler)}, 274)

    def test_module_boundaries_and_old_layout_rejected(self):
        dims = {'cnv': 44, 'fsr': 80, 'griffin': 100, 'mutcs': 30, 'fragma': 21}
        blocks = {name: np.full(n, i) for i, (name, n) in enumerate(dims.items())}
        result = build_cfdna_feature_vector(blocks)
        start = 0
        for i, n in enumerate(dims.values()):
            np.testing.assert_array_equal(result[start:start+n], np.full(n, i))
            start += n
        self.assertEqual(result.shape, (275,))
        old = {'cnv': np.zeros(39), 'fsr': np.zeros(30), 'griffin': np.zeros(100),
               'mutcs': np.zeros(85), 'fragma': np.zeros(21)}
        with self.assertRaises(ValueError):
            build_cfdna_feature_vector(old)

    def test_train_evaluate_and_predict_end_to_end(self):
        import yaml
        config = self.root / 'config.yaml'
        config.write_text(yaml.safe_dump({
            'model': {'d_model': 16, 'num_heads': 2, 'num_layers': 1, 'cfdna_dim': 275},
            'data': {'mri_shape': [8, 8, 8]},
            'training': {'epochs': 1, 'batch_size': 4, 'num_workers': 0}, 'seed': 42,
        }))
        env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                   MPLBACKEND='Agg', MPLCONFIGDIR=str(self.root / 'mpl'))
        def run(script, *args):
            result = subprocess.run([sys.executable, script, '--config', str(config), *map(str, args)],
                                    env=env, capture_output=True, text=True, timeout=120)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        run('train.py', '--data_path', self.root, '--labels_file', self.labels,
            '--output_dir', self.root / 'training')
        checkpoint_path = next((self.root / 'training').glob('*/checkpoints/best_model.pth'))
        ckpt = torch.load(checkpoint_path, weights_only=True)
        split = json.loads((checkpoint_path.parent.parent / 'split_patient_ids.json').read_text())
        idx = [int(pid[1:]) for pid in split['train']]
        expected = StandardScaler().fit(self.values[idx])
        frozen = scaler_from_checkpoint(ckpt, 275)
        np.testing.assert_allclose(frozen.mean_, expected.mean_)
        self.assertEqual(int(frozen.n_samples_seen_), len(idx))
        held_labels = self.root / 'held.csv'
        labels = pd.read_csv(self.labels)
        labels[labels.patient_id.isin(split['test'])].to_csv(held_labels, index=False)
        run('evaluate.py', '--model_path', checkpoint_path, '--test_data', self.root,
            '--labels_file', held_labels, '--output_dir', self.root / 'eval')
        evaluated = pd.read_csv(self.root / 'eval/predictions.csv')
        run('predict.py', '--model_path', checkpoint_path, '--batch_mode',
            '--data_dir', self.root, '--output_dir', self.root / 'pred')
        predicted = pd.read_csv(self.root / 'pred/batch_predictions.csv')
        joined = evaluated.merge(predicted, on='patient_id', suffixes=('_eval', '_pred'))
        self.assertEqual(len(joined), len(split['test']))
        np.testing.assert_allclose(joined.cancer_probability_eval, joined.cancer_probability_pred,
                                   rtol=1e-5, atol=1e-6)
        pid = split['test'][0]
        run('predict.py', '--model_path', checkpoint_path, '--mri_dir', self.root / pid,
            '--output_dir', self.root / 'single')
        model = BEAM(d_model=16, num_heads=2, num_layers=1).eval()
        model.load_state_dict(ckpt['model_state_dict'])
        mri, cfdna = load_patient(self.root / pid, (8, 8, 8), frozen)
        prob = infer(model, mri, cfdna, torch.device('cpu'))
        self.assertAlmostEqual(prob, float(evaluated.set_index('patient_id').loc[pid, 'cancer_probability']), places=5)


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
