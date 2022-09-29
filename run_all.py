import basic
import SFRtoSWR
import postprocess
import write_inflows
import Hydrographs
import post_process_heads
import matplotlib.pyplot as plt
import write_pond_inflows_rch
import make_wells
import initial_conditions
import os


# for run in [ 'June2019',
#             'June2020']:
# #
# for run in ['June2012', 'June2013', 'June2014', 'June2015',
#             'June2016', 'June2017', 'June2018', 'June2019',
#             'June2020']:
for run in ['June2013']:

    # print(run)
    # basic.setup_folder(run)
    # # basic.plot_all_aquifer_props(m, run)
    # basic.copy_mod_files(run)
    # make_wells.run(name=run)
    # write_inflows.run(model_name=run)
    # SFRtoSWR.run(run)
    # SFRtoSWR.plot_start_stage(None, os.path.join('versions', run))
    # write_pond_inflows_rch.run(run)
    #
    m = basic.load_model()
    basic.plot_all_aquifer_props(m, run)
    basic.plot_maps(m, run)

    success = initial_conditions.rerun_for_initial_cond(m, 3)

    success, buffer = m.run_model(silent = False,)

    if success:
        Hydrographs.run(run_name=run, reload = False)
        postprocess.run(run, riv_only = True)

        post_process_heads.run(run_name=run, head_frequency=5)


    plt.close()
    plt.close(plt.gcf())
    plt.close('all')
    plt.clf()

