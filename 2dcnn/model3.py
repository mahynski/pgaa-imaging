import run

if __name__ == '__main__':
    # Model basics
    run.run(
        prefix = 'model3',
        frac = 0.75,
        skip = 20,
        cnn_name = 'inceptionv3',
        mode = 'triangular2',
        epochs = 200,
        batch_size = 30,
        step_size = 20,
        wandb_project = 'pgaa-imaging-2d',
    )