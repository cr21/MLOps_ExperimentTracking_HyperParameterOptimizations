import os
from pathlib import Path
import sys

import rootutils

# Setup the root directory
root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
print(f"Project root: {root}")

# Rest of your imports
from typing import List, Optional
import hydra
from omegaconf import DictConfig
import pytorch_lightning as pl
import logging
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from src.utils.logging_utils import setup_logger, task_wrapper
from lightning.pytorch.loggers import Logger

# Set up logging
log = logging.getLogger(__name__)

def instantiate_callbacks(callback_cfg: DictConfig) -> List[pl.Callback]:
    callbacks: List[pl.Callback] = []
    if not callback_cfg:
        log.warning("No callback configs found! Skipping..")
        return callbacks
    i=0
    for _, cb_conf in callback_cfg.items():
        print(cb_conf)
        print(i)
        i+=1
        if "_target_" in cb_conf:
            log.info(f"Instantiating callback <{cb_conf._target_}>")
            callbacks.append(hydra.utils.instantiate(cb_conf))

    return callbacks

def instantiate_loggers(logger_cfg: DictConfig) -> List[Logger]:
    loggers: List[pl.LightningLoggerBase] = []
    if not logger_cfg:
        log.warning("No logger configs found! Skipping..")
        return loggers

    for _, lg_conf in logger_cfg.items():
        if "_target_" in lg_conf:
            log.info(f"Instantiating logger <{lg_conf._target_}>")
            loggers.append(hydra.utils.instantiate(lg_conf))

    return loggers

def plot_confusion_matrix(y_true, y_pred, class_names, title, filename):
    """Generate and save a confusion matrix plot with percentage values."""
    cm = confusion_matrix(y_true, y_pred)
    cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # Normalize to get percentages

    plt.figure(figsize=(12, 8))  # Increased figure size for better visibility
    sns.heatmap(cm_percentage, annot=True, fmt='.2%', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.xticks(rotation=45)  # Rotate x-axis labels for better visibility
    plt.yticks(rotation=0)   # Rotate y-axis labels for better visibility
    plt.savefig(filename)
    plt.close()

@task_wrapper
def train(
    cfg: DictConfig,
    trainer: pl.Trainer,
    model: pl.LightningModule,
    datamodule: pl.LightningDataModule,
):
    log.info("Starting training!")
    trainer.fit(model, datamodule)

    # Load the best model from the checkpoint
    if trainer.checkpoint_callback and trainer.checkpoint_callback.best_model_path:
        log.info(f"Loading best model from checkpoint: {trainer.checkpoint_callback.best_model_path}")
        best_model = model.__class__.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
    else:
        log.warning("No checkpoint found! Using current model weights.")
        best_model = model

    # Get training predictions and true labels
    train_loader = datamodule.train_dataloader()
    y_train_true = []
    y_train_pred = []
    
    for batch in train_loader:
        x, y = batch
        preds = best_model.predict_step(batch, 0)  # Use the predict_step method
        y_train_true.extend(y.numpy())
        y_train_pred.extend(preds.numpy())
    
    # Plot confusion matrix for training data
    plot_confusion_matrix(y_train_true, y_train_pred, 
                          class_names=datamodule.class_names,
                          title='Confusion Matrix for Training Data', 
                          filename='train_confusion_matrix.png')

    train_metrics = trainer.callback_metrics
    log.info(f"Training metrics:\n{train_metrics}")
    return train_metrics

@task_wrapper
def test(cfg: Optional[DictConfig] = None, trainer: Optional[pl.Trainer] = None, model: Optional[pl.LightningModule] = None, datamodule: Optional[pl.LightningDataModule] = None):
    log.info("Starting testing!")
    
    # Load the best model from the checkpoint
    if trainer.checkpoint_callback and trainer.checkpoint_callback.best_model_path:
        log.info(f"Loading best checkpoint: {trainer.checkpoint_callback.best_model_path}")
        best_model = model.__class__.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
    else:
        log.warning("No checkpoint found! Using current model weights.")
        best_model = model
    
    # Call trainer.test to get test metrics
    test_metrics = trainer.test(best_model, datamodule)
    
    # Get test predictions and true labels
    test_loader = datamodule.test_dataloader()
    y_test_true = []
    y_test_pred = []
    
    for batch in test_loader:
        x, y = batch
        preds = best_model.predict_step(batch, 0)  # Use the predict_step method
        y_test_true.extend(y.numpy())
        y_test_pred.extend(preds.numpy())
    
    # Plot confusion matrix for test data
    plot_confusion_matrix(y_test_true, y_test_pred, 
                          class_names=datamodule.class_names,
                          title='Confusion Matrix for Test Data', 
                          filename='test_confusion_matrix.png')

    log.info(f"Test metrics:\n{test_metrics}")  
    return test_metrics

@hydra.main(version_base=None, config_path="../configs", config_name="train")
def main(cfg: DictConfig):
    # Set up the root directory (if needed)
    log_dir = Path(cfg.paths.log_dir)
    print(log_dir)
    # set up logger
    setup_logger(log_dir/"train.log")
    #root.set_root_dir()

    # Create data module
    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: pl.LightningDataModule = hydra.utils.instantiate(cfg.data)

    # Create model
    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: pl.LightningModule = hydra.utils.instantiate(cfg.model)

    callbacks: List[pl.Callback] = instantiate_callbacks(cfg.get("callbacks"))
    
    loggers: List[Logger] = instantiate_loggers(cfg.get("logger"))
    # Create trainer
    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: pl.Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=loggers)
    # Train the model
    if cfg.get("train"):
        train(cfg, trainer, model, datamodule)

    # Test the model
    if cfg.get("test"):
        test(cfg, trainer, model, datamodule)

    # # Return metric score for hyperparameter optimization
    # optimized_metric = cfg.get("optimized_metric")
    # if optimized_metric and optimized_metric in trainer.callback_metrics:
    #     return trainer.callback_metrics[optimized_metric]

if __name__ == "__main__":
    main()
