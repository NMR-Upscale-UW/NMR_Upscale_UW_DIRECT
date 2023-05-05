import optuna

def objective(trial):
    '''
    Objective function to be optimized by Optuna.
    
    Dictionary:
        cnn

    Hyperparameters(accessing by the above dictionary):
        input_dim: input dimension
        output_dim: output dimension
        hidden_dims: hidden dimensions
        kernel_size: kernel size
        padding: padding
        p_drop: drop out values
        lr: learning rate
    '''
    # Define range of values to be tested for the hyperparameters
    input_dim = trial.suggest_int('input_dim', 1, 3)
    output_dim = trial.suggest_int('output_dim', 1, 3)
    # hidden_dims = trial.suggest_int('hidden_dims', 1, 3)
    kernel_size = trial.suggest_int('kernel_size', 1, 10)
    padding = trial.suggest_int('padding', 1, 5)
    p_drop = trial.suggest_float('p_drop',0.1, 0.5)


    #Generate the model
    model = CNN(cfg)
    # score = train_evaluate_hyperparameters(input_dim,output_dim, kernel_size, padding, p_drop)

study = optuna.create_study(sampler=optuna.samplers.TPESampler(), direction='maximize')
study.optimize(objective, n_trials=50)
print('Best params: ', study.best_params)