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
import shutil
import sys

def copyfiles(run):

    EXE_DIR = 'RR_2022'

    new_folder = os.path.join('temp', run)
    print(f'copying files from {EXE_DIR} to {new_folder}\n')

    if os.path.exists(new_folder):
        try:
            shutil.rmtree(new_folder, ignore_errors=True)
        except:
            print('could not remove folder')

    try:
        shutil.copytree(EXE_DIR, new_folder)
    except:
        shutil.rmtree(os.path.join(new_folder))

    shutil.rmtree(os.path.join(new_folder, 'Results'))
    os.mkdir(os.path.join(new_folder, 'Results'))

    return new_folder

def par_run(run, path, numdays = 365, do_pre_run = False):
    '''
    run single model with name 'run' in the path.
    :param run:
    :param path:
    :param numdays:
    :return:
    '''


    print('\n\n\n\n\n----------------------')
    print(run)
    basic.setup_folder(run)
    basic.reset_model_files(path = path)
    basic.write_run_name_to_file(run, 'started')
    m = basic.load_model(path=path)
    print(f"model ws {m.model_ws}")
    print(f"model exe {m.exe_name}")

    # GHB.run(run)
    # basic.plot_all_aquifer_props(m, run)



    if do_pre_run:
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
        Hydrographs.run(run_name=run, reload = True, ml = m)
        postprocess.run(run, riv_only = True, m= m)

        post_process_heads.run(run_name=run, head_frequency=10, add_basemap=False, m = m)
        # zone_bud.run(run, ml = m)
        basic.write_run_name_to_file(run, 'ended')

        try:
            p = os.path.join('initial_heads', run)
            os.mkdir(p)
            initial_conditions.set_starting_heads(m, plot = False, alt_outpath=p)
        except:
            print('did not copy final heads')

    plt.close()
    plt.close(plt.gcf())
    plt.close('all')
    plt.clf()


if __name__ =='__main__':
    name = sys.argv[1]
    print(f'running script for {name}')
    # name = 'June2016'
    path = copyfiles(name)
    par_run(name, path, numdays=109)
    # print(path)