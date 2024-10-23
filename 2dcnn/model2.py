import run

if __name__ == '__main__':
    # Model basics
    run.run(
        prefix = 'model2',
        frac = 0.75,
        skip = 5,
        cnn_name = 'nasnetmobile',
        mode = 'triangular2',
        epochs = 200,
        batch_size = 30,
        step_size = 20,
        wandb_project = 'pgaa-imaging-2d',
    )