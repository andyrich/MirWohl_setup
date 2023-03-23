import warnings

import pandas as pd

warnings.filterwarnings("ignore", category=DeprecationWarning)
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
import hydro_context
from shutil import copytree, ignore_patterns
import Streamflows

def copyfiles(run, starting_heads_from_previous = True, date_start = None):
    '''
    copyfiles from source (RR_2022) to new run
    :param run: run_name from run_names.json
    :param starting_heads_from_previous: copy heads from previous
    :param date_start: if left as None will be sourced from run_names.json
    :return: new_folder
    '''

    EXE_DIR = 'RR_2022'

    new_folder = os.path.join('temp', run)
    print(f'copying files from {EXE_DIR} to {new_folder}\n')

    if os.path.exists(new_folder):
        try:
            shutil.rmtree(new_folder, ignore_errors=True)
            print(f'removed folder {new_folder}')
        except:
            print(f'could not remove folder {new_folder}')

    try:
        shutil.copytree(EXE_DIR, new_folder, ignore = ignore_patterns('*.git'))
    except:
        shutil.rmtree(os.path.join(new_folder))

    shutil.rmtree(os.path.join(new_folder, 'Results'))
    os.mkdir(os.path.join(new_folder, 'Results'))

    if starting_heads_from_previous:
        # p = int(run.strip('June'))
        if date_start is None:
            info, swr_info, sfr_info, riv_keys_info = basic.load_params(run)
            date_start = info['start_date']
            # raise ValueError('if you want to use start_heads_from_previous, you need to provide starting date')

        if pd.to_datetime(date_start).year > 2012:
            # src = os.path.join('temp',f'June{p-1}')
            year = pd.to_datetime(date_start).year
            src = os.path.join('initial_heads', f'June{year-1}')
            print(f'copying initial heads from {src} to {new_folder}')
            initial_conditions.set_start_from_path(src, new_folder)

    return new_folder

def par_run(run,  numdays = 365,
            post_process_only = False,
            skip_setup = False,
            check_for_success = True,
            copyfiles_from_base = True,
            do_pre_run = False):

    '''
    run single model with name 'run' in the path.
    :param do_pre_run:
    :param copyfiles_from_base:
    :param check_for_success:
    :param skip_setup:
    :param post_process_only:
    :param run:
    :param numdays:
    :return:
    '''

    if post_process_only:
        skip_setup = True
        check_for_success = False
        copyfiles_from_base = False
        do_pre_run = False

    print(f"\n\n\n\nnumdays = {numdays},\n\
            post_process_only = {post_process_only},\n\
            skip_setup = {skip_setup},\n\
            check_for_success = {check_for_success},\n\
            copyfiles_from_base = {copyfiles_from_base},\n\
            do_pre_run = {do_pre_run}\n\n\n")


    if copyfiles_from_base:
        path = copyfiles(run, starting_heads_from_previous=True)
    else:
        path = os.path.join('temp', run)

    print('\n\n\n\n\n----------------------')
    print(run)

    if skip_setup or post_process_only:
        print('not creating files for new run')
    else:
        try:
            basic.setup_folder(run)
            basic.reset_model_files(path = path)
            basic.write_run_name_to_file(run, 'started', mode = 'a')
            hydro_context.run_current_context(run)
            # SFRtoSWR.run(run_name = run)
        except Exception as e:
            basic.write_run_name_to_file(run, 'failed set up   ' + e, mode='a')

    m = basic.load_model(path=path)
    print(f"model ws {m.model_ws}")
    print(f"model exe {m.exe_name}")

    # GHB.run(run)
    # basic.plot_all_aquifer_props(m, run)



    if do_pre_run and (not post_process_only):
        # SFRtoSWR.run(run)
        # SFR_calibrate.run(run)
        write_pond_inflows_rch.run(run, m = m)

        datestart_initial = basic.offset_start_date(run, 60)

        write_inflows.run(model_name=run, m = m, numdays=numdays, datestart=datestart_initial)
        make_wells.run(name=run, m = m, datestart=datestart_initial, numdays=numdays)

        success = initial_conditions.rerun_for_initial_cond(m, 1)

        m = basic.load_model(path = path)
    else:
        print('not doing pre-run')

    if skip_setup:
        print('not creating input files')
    else:
        try:
            make_wells.run(name=run, m = m)

            write_inflows.run(model_name=run, m = m)
            write_pond_inflows_rch.run(run, m = m)
            SFRtoSWR.plot_start_stage(None, os.path.join('versions', run))
            basic.copy_mod_files(run)
        except Exception as e:
            basic.write_run_name_to_file(run, 'failed set up p2.  ' + e, mode='a')

    if post_process_only:
        print('not doing run')
        success = True
    else:
        success, buffer = m.run_model(silent = False)


    if success or (not check_for_success):
        try:
            basic.write_run_name_to_file(run, 'successful', mode = 'a')
            basic.setup_folder(run)
            Hydrographs.run(run_name=run, reload = True, ml = m)
            postprocess.run(run, riv_only = True, m= m)
            Streamflows.run(out_folder=os.path.join('versions', run), ml=m)
            post_process_heads.run(run_name=run, head_frequency=10, add_basemap=False, m = m)
            # zone_bud.run(run, ml = m)

            p = os.path.join('initial_heads', run)
            initial_conditions.set_starting_heads(m, plot = False, alt_outpath=p)
            basic.write_run_name_to_file(run, 'completed post processing', mode='a')

        except Exception as e:
            basic.write_run_name_to_file(run, str(e), mode='a')

    else:
        basic.write_run_name_to_file(run, 'failed', mode='a')

    plt.close()
    plt.close(plt.gcf())
    plt.close('all')
    plt.clf()


if __name__ =='__main__':

    name = sys.argv[1]
    print(f'running script for {name}')

    if len(sys.argv) > 2:
        post_process_only = 'True' == sys.argv[2]
    else:
        post_process_only = False

    # name = 'June2016'
    # path = copyfiles(name)
    par_run(name,  numdays=365, post_process_only=post_process_only)
    # print(path)