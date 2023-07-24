
####
import warnings

import basic
import os
warnings.filterwarnings("ignore", category=DeprecationWarning)

import postprocess
import pandas as pd
import numpy as np
import run_parallel
warnings.filterwarnings("ignore", category=DeprecationWarning)

# path = "RR_2022\Results"
remove_ponds = True
max_reach = 76
datestart = '1/1/2013'
post_process_only = False
copyfiles_from_base = True

runs = ['SWR_long_2lays_med_SWRK',
"Base_model_long_07142023_constant_swrk",
'SWR_long_2lays_low_SWRK']
# runs = ['SWR_short_2lays_med_SWRK']
# runs = [
# "Base_model_07142023"]
model_dir = None
skip_setup = False



for run in runs:

 run_parallel.par_run(run,  numdays = None,
 post_process_only = post_process_only,
 skip_setup = skip_setup,
 check_for_success = True,
 copyfiles_from_base = copyfiles_from_base,
 do_pre_run = False,
 model_dir=model_dir)

# ml = basic.load_model(path = os.path.join('temp', run))
# ml.run_model()
# basic.plot_maps(ml, run)
# m = basic.load_model(path = "temp\SWR_short_2lays_low_SWRK")
# out_folder = os.path.join('versions', run)
# postprocess.plot_rds_stage(m, datestart, out_folder, numdays = numdays, ISWRPQAQ = None, max_reach = max_reach)
# m.run_model()