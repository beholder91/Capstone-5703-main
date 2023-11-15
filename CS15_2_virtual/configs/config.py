memo_config = {
    'max_len': 31,  
    'train_size': 0.8,
    # testing 
    # 'batch_size': 32, 
    # 'dropout_rates': [0],
    # 'learning_rates': [1e-3],
    # 'gamma_values': [2],
    # 'num_epochs': 2,
    # uncommand following 4 for client
    'batch_size': 256, 
    'dropout_rates': [0, 0.2, 0.5],
    'learning_rates': [1e-3, 1e-4, 1e-5],
    'gamma_values': [2, 3, 4],
    'num_epochs': 100,
    'momentum': 0.9, 
    'patience': 5, 
}


mvsa_config = {
    'max_len': 140,  
    'train_size': 0.8,
    # testing
    # 'batch_size': 32, 
    # 'dropout_rates': [0.2],
    # 'learning_rates': [1e-4],
    # 'num_epochs': 2,
    # uncommand following 3 for client
    'batch_size': 256, 
    'dropout_rates': [0, 0.2, 0.5],
    'learning_rates': [1e-2, 1e-3, 1e-4, 1e-5],
    'num_epochs': 100,
    'gamma': 2, 
    'momentum': 0.9, 
    'patience': 5, 
}
