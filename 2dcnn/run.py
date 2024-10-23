import pickle
import os

from pychemauth import utils
from pychemauth.classifier.cnn import CNNFactory
from datasets import load_dataset

os.environ["CUDA_VISIBLE_DEVICES"]="-1" # Run on CPU only

def run(
    prefix, 
    frac, 
    skip,
    cnn_name,
    mode='triangular2', 
    epochs=200,
    batch_size=10,
    step_size=10, # 1/2 cycle every this many epochs
    seed=42,
    wandb_project='pgaa-imaging-2d',
    wandb_restart=None,
    restart_lr=None,
    build_from_scratch=False
):
    # Fixed parameters
    image_size = (2631, 2631, 1)
    n_classes = 10
    HF_TOKEN = "hf_*" # Enter your own token here

    if build_from_scratch:
        # Build data loader using local dataset
        data_loc = './2d-dataset/'
        train_loader = utils.NNTools.build_loader(data_loc+'/train', batch_size=batch_size, shuffle=True)
        val_loader = utils.NNTools.build_loader(data_loc+'/test', batch_size=batch_size)
    else:
        # Use dataset from HuggingFace
        dataset_fnames = load_dataset(
            "mahynski/pgaa-sample-gadf-images", 
            split="train",
            token=HF_TOKEN,
            trust_remote_code=True, # This is important to include
            name='filenames' # This is the default
        )
        train_loader = NNTools.XLoader(
            x_files = [entry['filename'] for entry in dataset_fnames],
            y = [entry['label'] for entry in dataset_fnames],
            batch_size=batch_size,
            exclude=[10, 11, 12], # -> ['Carbon Powder', 'Phosphate Rock', 'Zircaloy']
            shuffle=True
        )

        dataset_fnames = load_dataset(
            "mahynski/pgaa-sample-gadf-images", 
            split="test",
            token=HF_TOKEN,
            trust_remote_code=True, # This is important to include
            name='filenames' # This is the default
        )
        val_loader = NNTools.XLoader(
            x_files = [entry['filename'] for entry in dataset_fnames],
            y = [entry['label'] for entry in dataset_fnames],
            batch_size=batch_size,
            exclude=[10, 11, 12], # -> ['Carbon Powder', 'Phosphate Rock', 'Zircaloy']
            shuffle=False
        )

    # Load CLR finder - restarts end with '_r'
    if prefix.endswith('_r'):
        prefix_ = prefix.split('_r')[0]
    else:
        prefix_ = prefix

    lrf = pickle.load(open(prefix_+'_lrf.pkl', 'rb'))
    if wandb_restart is None:
        clr = utils.NNTools.CyclicalLearningRate(
            base_lr=lrf.estimate_clr(frac=frac, skip=skip)[0],
            max_lr=lrf.estimate_clr(frac=frac, skip=skip)[1],
            step_size=step_size, # 1/2 cycle every this many epochs
            mode=mode,
        )
    else:
        if restart_lr is None: # Restarting with
            restart_lr = lrf.estimate_clr(frac=frac, skip=skip)[0] # Use lower bound by default
        clr = utils.NNTools.CyclicalLearningRate(
            base_lr=restart_lr,
            max_lr=restart_lr,
            step_size=step_size, # 1/2 cycle every this many epochs
            mode=mode,
        )

    compiler_kwargs={
        "optimizer": "adam", 
        "loss": "sparse_categorical_crossentropy",
        "weighted_metrics": ["accuracy", "sparse_categorical_accuracy"],
    }
    
    # Build model
    cnn_builder = CNNFactory(
        name=cnn_name,
        input_size=image_size,
        n_classes=n_classes,
        pixel_range=(-1, 1),
        cam=True,
        dropout=0.2,
    ) 

    # Train
    model = utils.NNTools.train(
        model=cnn_builder.build(),
        data=train_loader, 
        compiler_kwargs=compiler_kwargs,
        fit_kwargs={
            'epochs': epochs,
            'validation_data': val_loader,
            'callbacks': [clr],
        },
        model_filename=prefix+'.keras',
        history_filename=prefix+'_history.pkl',
        wandb_project=wandb_project,
        seed=seed,
        restart=None if wandb_restart is None else {"from_wandb":True, "filepath":wandb_restart, "weights_only":True},
        class_weight=None # Automatically weights by frequency in the dataset
    )
