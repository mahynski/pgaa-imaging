import run

if __name__ == '__main__':
    # Model basics
    run.run(
        model_idx = 2,
        base_lr = 1.0e-4,
        max_lr = 1.0e-2,
        mode = 'triangular2',
        epochs = 5000,
        batch_size = 32,
        step_size = 50,
        wandb_project = 'pgaa-imaging-1d',
    )