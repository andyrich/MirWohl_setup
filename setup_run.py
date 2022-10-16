import basic
import SFRtoSWR
import SFR_calibrate
import postprocess
import write_inflows
import Hydrographs
import post_process_heads
import matplotlib.pyplot as plt
import write_pond_inflows_rch
import make_wells
import GHB
import initial_conditions
import zone_bud
import os


# for run in [ 'June2019', 'June2020']:

# for run in ['June2012', 'June2013', 'June2014', 'June2015',
#             'June2016', 'June2017', 'June2018', 'June2019',
#             'June2020']:

allruns = ["June2016",'June2018','June2019', 'June2017',
'June2012', 'June2013', 'June2014','June2020', 'June2015']

runs = basic.check_runs(allruns)
numdays = 365
path = 'RR_2022'

run = 'June2016'

print('\n\n\n\n\n----------------------')
print(run)

# basic.reset_model_files(path = path)
basic.write_run_name_to_file(run, 'started')
m = basic.load_model(path=path)
GHB.run(run)
basic.plot_all_aquifer_props(m, run)




SFRtoSWR.run(run)



make_wells.run(name=run, m = m)
write_inflows.run(model_name=run, m = m)
write_pond_inflows_rch.run(run, m = m)



