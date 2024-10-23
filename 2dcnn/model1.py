import run

if __name__ == '__main__':
    # Model basics
    run.run(
        prefix = 'model1',
        frac = 0.6,
        skip = 0,
        cnn_name = 'mobilenetv3small',
        mode = 'triangular2',
        epochs = 200,
        batch_size = 30,
        step_size = 20,
        wandb_project = 'pgaa-imaging-2d',
    )
