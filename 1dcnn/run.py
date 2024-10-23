import pychemauth
import pickle
import os
import keras
import sklearn

import numpy as np

from pychemauth import utils
from pychemauth.datasets import load_pgaa
from sklearn.preprocessing import OrdinalEncoder

os.environ["CUDA_VISIBLE_DEVICES"]="-1" # Run on CPU only

def load(
    limit = 2631 # From EDA - cutoff at E ~ 7650 keV
):
    X, y = load_pgaa(return_X_y=True, as_frame=True)
    X_data = X.values
    y_data = y.values
    centers = np.array(X.columns[:limit])

    # Exclude these classes from training
    mask = ~np.array([y_ in ['Carbon Powder', 'Phosphate Rock', 'Zircaloy'] for y_ in y_data])
    X_data = X_data[mask, :]
    y_data = y_data[mask]
    indices = np.arange(np.sum(mask))

    # Normalize sum after clipping, then take log
    X_clipped = X_data[:,:limit]
    X_normed = (X_clipped.T/np.sum(X_clipped, axis=1)).T
    X_normed = np.log(np.clip(X_normed, a_min=1.0e-7, a_max=None))

    X_train, X_test, y_train, y_test, idx_train, idx_test = sklearn.model_selection.train_test_split(
        np.expand_dims(X_normed, axis=-1), # Add axis (1 channel)
        y_data, 
        indices,
        test_size=0.2, random_state=42, shuffle=True, stratify=y_data)
    
    n_classes = len(np.unique(y_train))
    image_size = X_train.shape[1:]

    # For sparse categorical encoding
    enc = OrdinalEncoder(dtype=int)
    y_train = enc.fit_transform(y_train.reshape(-1,1)).reshape(-1)
    y_test = enc.transform(y_test.reshape(-1,1)).reshape(-1)

    return X_train, X_test, y_train, y_test, idx_train, idx_test, centers, n_classes, image_size, enc, X_normed

def build(
    model_idx, 
    image_size,
    n_classes,
    activation='elu'
):
    input_ = keras.layers.Input(shape=image_size)
    if model_idx == 1: # 16
        conv1 = keras.layers.Conv1D(filters=16, kernel_size=16, activation=activation, strides=8, padding='same', use_bias=True)(input_)
        pool1 = keras.layers.MaxPool1D(2)(conv1)
        conv2 = keras.layers.Conv1D(filters=16*8, kernel_size=8, activation=activation, strides=4, padding='same', use_bias=True)(pool1)
        pool2 = keras.layers.MaxPool1D(2)(conv2)
        conv3 = keras.layers.Conv1D(filters=16*8*4, kernel_size=8, activation=activation, strides=2, padding='same', use_bias=True)(pool2)
        gap = keras.layers.GlobalAveragePooling1D()(conv3)
        output = keras.layers.Dense(n_classes, activation='softmax')(gap)
    elif model_idx == 2: # 32
        conv1 = keras.layers.Conv1D(filters=16, kernel_size=16, activation=activation, strides=8, padding='same', use_bias=True)(input_)
        conv2 = keras.layers.Conv1D(filters=16, kernel_size=8, activation=activation, strides=4, padding='same', use_bias=True)(conv1)
        pool1 = keras.layers.MaxPool1D(8)(conv2)
        conv3 = keras.layers.Conv1D(filters=16*2, kernel_size=8, activation=activation, strides=1, padding='same', use_bias=True)(pool1)
        conv4 = keras.layers.Conv1D(filters=16*2, kernel_size=8, activation=activation, strides=1, padding='same', use_bias=True)(conv3)
        gap = keras.layers.GlobalAveragePooling1D()(conv4)
        output = keras.layers.Dense(n_classes, activation='softmax')(gap)
    elif model_idx == 3: # 17
        conv1 = keras.layers.Conv1D(filters=16, kernel_size=16, activation=activation, strides=8, padding='same', use_bias=True)(input_)
        conv2 = keras.layers.Conv1D(filters=16, kernel_size=8, activation=activation, strides=4, padding='same', use_bias=True)(conv1)
        pool1 = keras.layers.MaxPool1D(4)(conv2)
        conv3 = keras.layers.Conv1D(filters=16*2, kernel_size=8, activation=activation, strides=1, padding='same', use_bias=True)(pool1)
        conv4 = keras.layers.Conv1D(filters=16*2, kernel_size=8, activation=activation, strides=1, padding='same', use_bias=True)(conv3)
        gap = keras.layers.GlobalAveragePooling1D()(conv4)
        output = keras.layers.Dense(n_classes, activation='softmax')(gap)
    else:
        raise ValueError('Unknown model index')
        
    model = keras.Model(inputs=[input_], outputs=[output])
    
    return model

def run(
    model_idx,
    base_lr,
    max_lr,
    mode='triangular2', 
    epochs=5000,
    batch_size=32,
    step_size=50, # 1/2 cycle every this many epochs
    seed=42,
    wandb_project='pgaa-imaging-1d'
):
    # Get dataset
    X_train, X_test, y_train, y_test, idx_train, idx_test, centers, n_classes, image_size, enc, X_normed = load(limit=2631)

    prefix = f"model{model_idx}"
    
    # Load CLR finder
    clr = pychemauth.utils.NNTools.CyclicalLearningRate(
        base_lr=base_lr,
        max_lr=max_lr,
        step_size=step_size, # 1/2 cycle every this many epochs
        mode=mode,
    )
    
    # Checkpointing callback
    ckpt = keras.callbacks.ModelCheckpoint(
        filepath=f"./{prefix}_"+"checkpoints/model.{epoch:05d}",
        save_weights_only=True,
        monitor="val_sparse_categorical_accuracy",
        save_freq="epoch",
        mode="max",
        save_best_only=False,
    )
    
    model = utils.NNTools.train(
        model=build(
            model_idx=model_idx, 
            image_size=image_size,
            n_classes=n_classes,
            activation='elu'
        ),
        data=(X_train, y_train), 
        fit_kwargs={
            'epochs': epochs,
            'batch_size': batch_size,
            'shuffle': True,
            'validation_split': 0.2,
            'callbacks': [clr, ckpt]
        },
        model_filename=prefix+'.keras',
        history_filename=prefix+'_history.pkl',
        wandb_project=wandb_project,
        seed=seed,
        class_weight=None # Automatically weights by frequency in the dataset
    )