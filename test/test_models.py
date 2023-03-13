import torch
import torch.nn as nn
import unittest
import sys

sys.path.append("../NMR_Upscale_UW_DIRECT")
from models import MLP, CNN, ConvVAE
from model_params import DotDict, name2params


class TestMLP(unittest.TestCase):
    def test_forward(self):
        cfg = name2params['mlp']
        mlp = MLP(cfg)
        x = torch.randn(32, cfg.input_dim)
        y = mlp(x)
        self.assertEqual(y.shape, (32, cfg.output_dim), f"Expected output shape (32, {cfg.output_dim}), but got {y.shape}")


class TestCNN(unittest.TestCase):
    def test_forward(self):
        cfg = name2params['cnn']
        cnn = CNN(cfg)
        x = torch.randn(32, cfg.input_dim, 10)
        y = cnn(x)
        self.assertEqual(y.shape, (32, cfg.output_dim, 10), f"Expected output shape (32, {cfg.output_dim}, 10), but got {y.shape}")


class TestConvVAE(unittest.TestCase):
    def test_forward(self):
        cfg = name2params['conv_vae']
        vae = ConvVAE(cfg)
        x = torch.randn(16, cfg.encoder.input_dim, 10)
        y, mu, log_var = vae(x)
        self.assertEqual(y.shape, (16, cfg.decoder.output_dim, 10), f"Expected output shape (16, 3, 10), but got {y.shape}")
        self.assertEqual(mu.shape , (16, 10), f"Expected mu shape (16, 2), but got {mu.shape}")
        self.assertEqual(log_var.shape,(16, 10), f"Expected log_var shape (16, 2), but got {log_var.shape}")


if __name__ == '__main__':
    unittest.main()
