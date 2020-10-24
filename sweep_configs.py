

sweep_configs = [
{
    'name': 'classification gauss - loss function and activation comparision',
    'method': 'grid',
    'parameters': {
        'data': {
            'values': ['classification/data.three_gauss.train.100.csv']
        },
        'layers': {
            'values': ['[2,3]', '[2,2,3]', '[2,2,2,3]', '[2,4,3]', '[2,4,4,3]']
        },            
        'loss_function':{
            'values': ['SE', 'cross_entropy']
        },
        'activation_function':{
            'values': ['sigmoid', 'ReLU', 'LeakyReLU']
        },
        'seed':{
            'values': [1]
        },
        'lr':{
            'values': [10, 1, 0.1, 0.01, 0.001]
        },
        'epochs':{
            'values': [50]
        }
    }
},

{
    'name': 'classification gauss architecture influence',
    'method': 'grid',
    'parameters': {
        'data': {
            'values': ['classification/data.three_gauss.train.10000.csv']
        },
        'layers': {
            'values': ['[2,3]','[2,2,3]','[2,2,2,3]','[2,2,2,2,3]','[2,2,2,2,2,3]',
                               '[2,4,3]','[2,4,4,3]','[2,4,4,4,3]','[2,4,4,4,4,3]',
                               '[2,8,3]','[2,8,8,3]','[2,8,8,8,3]','[2,8,8,8,8,3]']
        },            
        'loss_function':{
            'values': ['SE', 'cross_entropy']
        },
        'activation_function':{
            'values': ['sigmoid', 'ReLU', 'LeakyReLU']
        },
        'seed':{
            'values': [1, 2, 3, 4, 5]
        },
        'lr':{
            'values': [10, 1, 0.1, 0.01, 0.001]
        },
        'epochs':{
            'values': [50]
        }
    }
}]