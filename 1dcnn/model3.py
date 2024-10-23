import run

if __name__ == '__main__':
    # Model basics
    run.run(
        model_idx = 3,
        base_lr = 5.0e-5,
        max_lr = 1.0e-2,
        mode = 'triangular2',
        epochs = 5000,
        batch_size = 32,
        step_size = 50,
        wandb_project = 'pgaa-imaging-1d',
    )