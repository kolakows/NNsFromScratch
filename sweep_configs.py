import numpy as np

report_sweep_configs = [
{
    'name': 'report classification gauss - loss function and activation comparision',
    'method': 'grid',
    'parameters': {
        'data': {
            'values': ['classification/data.three_gauss.train.1000.csv']
        },

        'layers': {
            'values': ['[2,4,3]']
        },            
        'loss_function':{
            'values': ['MSE', 'cross_entropy']
        },
        'activation_function':{
            'values': ['sigmoid', 'ReLU', 'LeakyReLU']
        },
        'seed':{
            'values': [1,2,3,4,5]
        },
        'lr':{
            'values': [1]
        },
        'epochs':{
            'values': [50]
        }
    }
},

{  
    'name': 'report classification gauss - architecture influence',
    'method': 'grid',
    'parameters': {
        'data': {
            'values': ['classification/data.three_gauss.train.1000.csv']
        },
        'layers': {
            'values': ['[2,3]','[2,2,3]','[2,2,2,3]','[2,2,2,2,3]','[2,2,2,2,2,3]',
                               '[2,4,3]','[2,4,4,3]','[2,4,4,4,3]','[2,4,4,4,4,3]',
                               '[2,8,3]','[2,8,8,3]','[2,8,8,8,3]','[2,8,8,8,8,3]']
        },            
        'loss_function':{
            'values': ['cross_entropy']
        },
        'activation_function':{
            'values': ['sigmoid']
        },
        'seed':{
            'values': [1, 2, 3, 4, 5]
        },
        'lr':{
            'values': [1]
        },
        'epochs':{
            'values': [50]
        }
    }
},
{  
    'name': 'report classification gauss - architecture influence - relu',
    'method': 'grid',
    'parameters': {
        'data': {
            'values': ['classification/data.three_gauss.train.100.csv']
        },
        'layers': {
            'values': ['[2,3]','[2,2,3]','[2,2,2,3]','[2,2,2,2,3]','[2,2,2,2,2,3]',
                               '[2,4,3]','[2,4,4,3]','[2,4,4,4,3]','[2,4,4,4,4,3]',
                               '[2,8,3]','[2,8,8,3]','[2,8,8,8,3]','[2,8,8,8,8,3]']
        },            
        'loss_function':{
            'values': ['MSE']
        },
        'activation_function':{
            'values': ['ReLU']
        },
        'seed':{
            'values': [1, 2, 3, 4, 5]
        },
        'lr':{
            'values': [1]
        },
        'epochs':{
            'values': [50]
        }
    }
}]

report_regression_sweep_configs = [
{
    'name': 'report regression cube corrected - loss function and activation comparision',
    'method': 'grid',
    'parameters': {
        'data': {
            'values': ['regression/data.cube.train.1000.csv']
        },

        'layers': {
            'values': ['[1,4,4,1]']
        },            
        'loss_function':{
            'values': ['MSE', 'MAE']
        },
        'activation_function':{
            'values': ['sigmoid', 'ReLU', 'LeakyReLU']
        },
        'seed':{
            'values': [1,2,3,4,5]
        },
        'lr':{
            'values': [1]
        },
        'epochs':{
            'values': [100]
        }
    }
},

{  
    'name': 'report regression cube corrected - architecture influence',
    'method': 'grid',
    'parameters': {
        'data': {
            'values': ['regression/data.cube.train.1000.csv']
        },
        'layers': {
            'values': ['[1,1]','[1,2,1]','[1,2,2,1]','[1,2,2,2,1]','[1,2,2,2,2,1]',
                               '[1,4,1]','[1,4,4,1]','[1,4,4,4,1]','[1,4,4,4,4,1]',
                               '[1,8,1]','[1,8,8,1]','[1,8,8,8,1]','[1,8,8,8,8,1]']
        },            
        'loss_function':{
            'values': ['MSE']
        },
        'activation_function':{
            'values': ['LeakyReLU']
        },
        'seed':{
            'values': [1, 2, 3, 4, 5]
        },
        'lr':{
            'values': [1]
        },
        'epochs':{
            'values': [100]
        }
    }
}]




preparation_sweep_configs = [
{
    #0
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
            'values': ['MSE', 'cross_entropy']
        },
        'activation_function':{
            'values': ['sigmoid', 'ReLU', 'LeakyReLU']
        },
        'seed':{
            'values': [1]
        },
        #doświadczalnie lr = 1 dawal ok wyniki
        'lr':{
            'values': [10, 1, 0.1, 0.01, 0.001]
        },
        'epochs':{
            'values': [50]
        }
    }
},

{
    #1
    'name': 'regression cube - loss function and activation comparision',
    'method': 'grid',
    'parameters': {
        'data': {
            'values': ['regression/data.cube.train.100.csv']
        },
        'layers': {
            'values': ['[1,1]', '[1,2,1]', '[1,2,2,1]', '[1,4,1]', '[1,4,4,1]']
        },            
        'loss_function':{
            'values': ['MSE', 'MAE']
        },
        'activation_function':{
            'values': ['sigmoid', 'ReLU', 'LeakyReLU']
        },
        'seed':{
            'values': [1]
        },
        'lr':{
            'values': [1]
        },
        'epochs':{
            'values': [50]
        }
    }
},

{   #2
    'name': 'classification gauss architecture influence',
    'method': 'grid',
    'parameters': {
        'data': {
            'values': ['classification/data.three_gauss.train.1000.csv']
        },
        'layers': {
            'values': ['[2,3]','[2,2,3]','[2,2,2,3]','[2,2,2,2,3]','[2,2,2,2,2,3]',
                               '[2,4,3]','[2,4,4,3]','[2,4,4,4,3]','[2,4,4,4,4,3]',
                               '[2,8,3]','[2,8,8,3]','[2,8,8,8,3]','[2,8,8,8,8,3]']
        },            
        'loss_function':{
            'values': ['MSE']
        },
        'activation_function':{
            'values': ['ReLU']
        },
        'seed':{
            'values': [1, 2, 3, 4, 5]
        },
        'lr':{
            'values': [10]
        },
        'epochs':{
            'values': [50]
        }
    }
},

{#3
    'name': 'regression cube - find lr',
    'method': 'grid',
    'parameters': {
        'data': {
            'values': ['regression/data.cube.train.100.csv']
        },
        'layers': {
            'values': ['[1,2,1]']
        },            
        'loss_function':{
            'values': ['MSE']
        },
        'activation_function':{
            'values': ['ReLU']
        },
        'seed':{
            'values': [1,2,3,4,5]
        },
        # lr = 1 is ok
        'lr':{
            'values': [1,10,100,0.1,0.01]
        },
        'epochs':{
            'values': [50]
        }
    }
},

{#4
    'name': 'regression cube - fit the curve',
    'method': 'grid',
    'parameters': {
        'data': {
            'values': ['regression/data.cube.train.100.csv']
        },
        'layers': {
            'values': ['[1,4,1]']
        },            
        'loss_function':{
            'values': ['cross_entropy']
        },
        'activation_function':{
            'values': ['sigmoid']
        },
        'seed':{
            'values': [1]
        },
        'lr':{
            'values': [1]
        },
        'epochs':{
            'values': [200]
        }
    }
},
]