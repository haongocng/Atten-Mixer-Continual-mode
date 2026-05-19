Dataset_setting = {
    'bundle':{
        'num_node':14240
    },
    'games':{
        'num_node':17389
    },
    'ml-1m':{
        'num_node':3416
    }
}

Model_setting = {
    'AttenMixer': {
        'model_dir': 'attenMixer',
        'dataloader':'AttMixerDataset',
        'norm': True,
        'scale': True,
        'use_lp_pool': True,
        'softmax':True,
        'lr_dc': 0.1,
        'lr_dc_step': 3,
        'l2': 1e-5,
        'n_layers': 1,
        'dropout': 0.1,
        'alpha': 0.2,
        'patience':3 ,
        'description': 'HIDE',
        'session_len': 50,
        'dot':True,
        'last_k':7, # need to be fixed
        # need to be tuned
        'epochs':100,
        'item_embedding_dim': 32,
        'learning_rate': 0.001,
        'batch_size':64,
        'l_p':3,
        'heads':8
    }

}


HyperParameter_setting = {
     'AttenMixer': {
        'int': {
            'l_p': {'min':1, 'max': 10, 'step': 1}
        },
        'categorical': {
            'item_embedding_dim': [32, 64, 128],
            'learning_rate': [0.0001, 0.001, 0.01],
            'batch_size': [64, 128, 256],
            'heads': [1,2,4,8]
        }
    }
}

Best_setting = {
    'AttenMixer': {
        'bundle':{
            'epochs':100,
            'item_embedding_dim': 32,
            'learning_rate': 0.0001,
            'batch_size':256,
            'l_p':7,
            'heads':1
        },
        'games':{
            'epochs':100,
            'item_embedding_dim': 128,
            'learning_rate': 0.001,
            'batch_size':256,
            'l_p':3,
            'heads':4
        },
        'ml-1m':{
            'epochs':100,
            'item_embedding_dim': 32,
            'learning_rate': 0.001,
            'batch_size':64,
            'l_p':10,
            'heads':2
        }
    }

}
