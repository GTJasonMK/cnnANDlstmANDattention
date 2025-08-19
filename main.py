from __future__ import annotations

import argparse
import os
from typing import Optional

import numpy as np
import torch

from config import load_config
from data_preprocessor import load_array_from_path, create_dataloaders
from model_architecture import CNNLSTMAttentionModel
from trainer import Trainer
from visualizer import plot_losses, plot_lr, plot_predictions, plot_attention_heatmap


def build_model_with_data(cfg, input_size: int, n_targets: int):
    # Build model from cfg but override num_features and targets
    m = cfg.model
    model = CNNLSTMAttentionModel(
        num_features=input_size,
        cnn_layers=[vars(l) if not isinstance(l, dict) else l for l in m.cnn.layers],
        use_batchnorm=m.cnn.use_batchnorm,
        cnn_dropout=m.cnn.dropout,
        lstm_hidden=m.lstm.hidden_size,
        lstm_layers=m.lstm.num_layers,
        bidirectional=m.lstm.bidirectional,
        attn_enabled=m.attention.enabled,
        attn_heads=m.attention.num_heads,
        attn_dropout=m.attention.dropout,
        fc_hidden=m.fc_hidden,
        forecast_horizon=m.forecast_horizon,
        n_targets=n_targets,
        attn_add_pos_enc=m.attention.add_positional_encoding,
    )
    return model


def main(config_path: Optional[str]):
    cfg = load_config(config_path)
    # load data
    if cfg.data.data_path is None:
        raise ValueError("Please provide data.data_path in config pointing to CSV/NPZ/NPY")
    data = load_array_from_path(cfg.data.data_path)
    train_loader, val_loader, test_loader, input_size, n_targets = create_dataloaders(
        data=data,
        sequence_length=cfg.data.sequence_length,
        horizon=cfg.data.horizon,
        feature_indices=cfg.data.feature_indices,
        target_indices=cfg.data.target_indices,
        normalize=cfg.data.normalize,
        batch_size=cfg.data.batch_size,
        train_split=cfg.data.train_split,
        val_split=cfg.data.val_split,
        num_workers=cfg.data.num_workers,
        shuffle_train=cfg.data.shuffle_train,
        drop_last=cfg.data.drop_last,
    )

    model = build_model_with_data(cfg, input_size, n_targets)
    trainer = Trainer(model, cfg)
    history = trainer.fit(train_loader, val_loader)

    # visualize (only save, do not show)
    plot_losses(history)
    plot_lr(history)
    # predictions and attention
    preds, targets = trainer.predict(test_loader)
    plot_predictions(preds, targets)
    # one forward with attn
    model.eval()
    with torch.no_grad():
        for x, _ in test_loader:
            out, attn = model(x, return_attn=True)
            plot_attention_heatmap(attn)
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None, help="Path to JSON/YAML config file")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume")
    args = parser.parse_args()
    if args.resume:
        cfg = load_config(args.config)
        from data_preprocessor import load_array_from_path, create_dataloaders
        data = load_array_from_path(cfg.data.data_path)
        train_loader, val_loader, test_loader, input_size, n_targets = create_dataloaders(
            data=data,
            sequence_length=cfg.data.sequence_length,
            horizon=cfg.data.horizon,
            feature_indices=cfg.data.feature_indices,
            target_indices=cfg.data.target_indices,
            normalize=cfg.data.normalize,
            batch_size=cfg.data.batch_size,
            train_split=cfg.data.train_split,
            val_split=cfg.data.val_split,
            num_workers=cfg.data.num_workers,
            shuffle_train=cfg.data.shuffle_train,
            drop_last=cfg.data.drop_last,
        )
        model = build_model_with_data(cfg, input_size, n_targets)
        trainer = Trainer(model, cfg)
        last_epoch = trainer.load_checkpoint(args.resume)
        print(f"Resumed from epoch {last_epoch}")
        history = trainer.fit(train_loader, val_loader)
    else:
        main(args.config)

