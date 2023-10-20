
####
import warnings
import matplotlib as mpl
mpl.use("TkAgg")  # or can use 'TkAgg', whatever you have/prefer
import basic
import os
warnings.filterwarnings("ignore", category=DeprecationWarning)

import postprocess
import pandas as pd
import numpy as np
import run_parallel
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# path = "RR_2022\Results"
remove_ponds = True
max_reach = 76
datestart = '1/1/2013'
post_process_only = True
copyfiles_from_base = True
# runs = [
# "Base_model_long_07142023_constant_swrk",
# 'SWR_long_2lays_low_SWRK']
# runs = ['SWR_long_2lays_med_SWRK',
# "Base_model_long_07142023_constant_swrk",
# 'SWR_long_2lays_low_SWRK']
runs = [
'RIVnoSWR_v4']
# runs = ['SWR_long_2lays_med_SWRK']
# runs = [
# "Base_model_07142023"]
model_dir = None
skip_setup = False
write_SFRtoSWR = False
make_wells_input = True
write_inflows_input = True

copy_mod_files_for_website = False

post_process_SWR = False


for run in runs:

 run_parallel.par_run(run,  numdays = None,
 post_process_only = post_process_only,
 skip_setup = skip_setup,
 check_for_success = True,
 copyfiles_from_base = copyfiles_from_base,
 write_SFRtoSWR = write_SFRtoSWR,
 make_wells_input = make_wells_input,
 write_inflows_input = write_inflows_input,
 copy_mod_files_for_website = copy_mod_files_for_website,
 do_pre_run = False,
 model_dir=model_dir,
 post_process_SWR = post_process_SWR)

# ml = basic.load_model(path = os.path.join('temp', run))
# ml.run_model()
# basic.plot_maps(ml, run)
# m = basic.load_model(path = "temp\SWR_short_2lays_low_SWRK")
# out_folder = os.path.join('versions', run)
# postprocess.plot_rds_stage(m, datestart, out_folder, numdays = numdays, ISWRPQAQ = None, max_reach = max_reach)
# m.run_model()