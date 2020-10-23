from sweep import *

# prepare sweep config, update also train function default config in sweep after adding new parameters type
# change local_path in sweep.py train function before running sweep

sweep_config = {
    'name': 'classification sweep',
    'method': 'grid',
    'parameters': {
        'data': {
            'values': ['classification/data.three_gauss.test.100.csv']
        },
        'layers': {
            'values': ['[2,3]','[2,2,3]']
        },            
        'loss_function':{
            'values': ['MSE', 'SE']
        },
        'activation_function':{
            'values': ['sigmoid', 'relu']
        },
        'seed':{
            'values': [1]
        },
        'lr':{
            'values': [1,2]
        },
    }
}

sweep(sweep_config)