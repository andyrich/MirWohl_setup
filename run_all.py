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
numdays = 109
path = 'RR_2022'

for run in runs:
    # for run in ['June2012','June2013', 'June2014', 'June2015', 'June2020'
    #             ]:
    # for run in ['ghbpt01', 'ghbpt001']:
    # for run in ['sfrpt07','sfrpt04']:
    print('\n\n\n\n\n----------------------')
    print(run)
    basic.setup_folder(run)
    basic.reset_model_files(path = path)
    basic.write_run_name_to_file(run, 'started')
    m = basic.load_model(path=path)
    # GHB.run(run)
    # basic.plot_all_aquifer_props(m, run)




    # SFRtoSWR.run(run)
    # SFR_calibrate.run(run)
    write_pond_inflows_rch.run(run, m = m)

    datestart_initial = basic.offset_start_date(run, 60)

    write_inflows.run(model_name=run, m = m, numdays=numdays, datestart=datestart_initial)
    make_wells.run(name=run, m = m, datestart=datestart_initial, numdays=numdays)

    success = initial_conditions.rerun_for_initial_cond(m, 1)
    m = basic.load_model(path = path)
    make_wells.run(name=run, m = m)
    write_inflows.run(model_name=run, m = m)
    write_pond_inflows_rch.run(run, m = m)
    SFRtoSWR.plot_start_stage(None, os.path.join('versions', run))

    success, buffer = m.run_model(silent = False)

    basic.copy_mod_files(run)

    if success:
        Hydrographs.run(run_name=run, reload = True)
        postprocess.run(run, riv_only = True)

        post_process_heads.run(run_name=run, head_frequency=5, add_basemap=False)
        # zone_bud.run(run)
        basic.write_run_name_to_file(run, 'ended')

    plt.close()
    plt.close(plt.gcf())
    plt.close('all')
    plt.clf()

