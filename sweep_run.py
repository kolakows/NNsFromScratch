from sweep import *
from sweep_configs import *

# prepare sweep config, update also train function default config in sweep after adding new parameters type
# change local_path in sweep.py train function before running sweep

#sweep(preparation_sweep_configs[4])
#sweep(report_sweep_configs[2])


for sweep_config in report_sweep_configs:
    sweep(sweep_config)