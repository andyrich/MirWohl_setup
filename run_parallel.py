import warnings

import pandas as pd

warnings.filterwarnings("ignore", category=DeprecationWarning)
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
import zone_bud
import os
import shutil
import sys
import hydro_context
from shutil import copytree, ignore_patterns
import Streamflows
import traceback


def copyfiles(run, starting_heads_from_previous = True, date_start = None, main_folder = None):
    '''
    copyfiles from source (RR_2022) to new run
    :param run: run_name from run_names.json
    :param starting_heads_from_previous: copy heads from previous
    :param date_start: if left as None will be sourced from run_names.json
    :return: new_folder
    '''


    if main_folder is None:
        main_folder = r'C:\modeling\MirabelWohler'

    EXE_DIR = os.path.join(main_folder,'RR_2022')

    new_folder = os.path.join(main_folder, 'temp', run)
    print(f'copying files from {EXE_DIR} to {new_folder}\n')

    if os.path.exists(new_folder):
        try:
            shutil.rmtree(new_folder, ignore_errors=True)
            print(f'removed folder {new_folder}')
        except:
            print(f'could not remove folder {new_folder}')

    try:
        shutil.copytree(EXE_DIR, new_folder, ignore = ignore_patterns('*.git', "reg_pest*", "pp_pest*"))
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

def par_run(run,  numdays = None,
            post_process_only = False,
            skip_setup = False,
            check_for_success = True,
            copyfiles_from_base = True,
            copy_mod_files_for_website = True,
            do_pre_run = False,
            model_dir = None,
            main_folder = None,
            starting_heads_from_previous = False,
            set_starting_heads_after = False,
            make_wells_input = True,
            write_inflows_input = True,
            write_SFRtoSWR = True,
            write_pond_inflows = True,
            do_zone_bud = False,
            post_process_hydrographs = True,
            post_process_streamflow = True,
            post_process_head_maps = True,
            post_process_SWR = True):

    '''
    run single model with name 'run' in the path.
    :param write_inflows_input:
    :param make_wells_input:
    :param write_pond_inflows:
    :param write_SFRtoSWR:
    :param main_folder: main folder all model runs are - defaults to C:\modeling\MirabelWohler
    :param do_pre_run:
    :param copyfiles_from_base:
    :param check_for_success:
    :param skip_setup:
    :param post_process_only:
    :param post_process_hydrographs:
    :param post_process_streamflow:
    :param post_process_head_maps:
    :param post_process_SWR:
    :param run:
    :param numdays:
    :param model_dir: model_dir to model_dir (if None will be assumed to be temp/{run})
    :param starting_heads_from_previous: use the end of the previous year to set as staring condition
    :param set_starting_heads_after: after run is complete, set starting heads file.
                not implemented for multiyear runs correctly
    :param do_zone_bud: do zone bud
    :return: None
    '''

    if post_process_only:
        skip_setup = True
        check_for_success = False
        copyfiles_from_base = False
        do_pre_run = False
        starting_heads_from_previous = False


    if main_folder is None:
        main_folder = r'C:\modeling\MirabelWohler'

    if copyfiles_from_base:
        model_dir = copyfiles(run, starting_heads_from_previous=starting_heads_from_previous,
                              main_folder = main_folder)
    elif model_dir is None:
        model_dir = os.path.join(main_folder, 'temp', run)
    else:
        if os.path.exists(model_dir):
            print('using user defined model_dir')
        else:
            raise AssertionError(f'folder does not exist for model_dir\n{model_dir}')

    if numdays is None:
        info, swr_info, sfr_info, riv_keys_info = basic.load_params(run)
        numdays = info['numdays']

    print(f"\n\n\n\nnumdays = {numdays},\n\
            post_process_only = {post_process_only},\n\
            skip_setup = {skip_setup},\n\
            check_for_success = {check_for_success},\n\
            copyfiles_from_base = {copyfiles_from_base},\n\
            do_pre_run = {do_pre_run}\n\n\n")

    print('\n\n\n\n\n----------------------')
    print(run)

    if skip_setup or post_process_only:
        print('not creating files for new run')
    else:
        try:
            basic.setup_folder(run)
            # basic.reset_model_files(path = model_dir)
            basic.write_run_name_to_file(run, 'started', mode = 'a')
            hydro_context.run_current_context(run)
            # SFRtoSWR.run(run_name = run)
        except Exception as e:
            warnings.warn(f"failed bc of\n{e}")
            print(traceback.format_exc())

            basic.write_run_name_to_file(run, 'failed set up   ' + e, mode='a')

    m = basic.load_model(path=model_dir)
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

        if not success:
            raise ValueError('Model did not run successfully on pre-run')

        m = basic.load_model(path = model_dir)
    else:
        print('not doing pre-run')

    if skip_setup:
        print('not creating input files')
    else:
        try:
            if make_wells_input:
                make_wells.run(name=run, m = m)

            if write_inflows_input:
                print('writing inflows')
                write_inflows.run(model_name=run, m = m)
            if write_pond_inflows:
                print('writing pond inflows')
                write_pond_inflows_rch.run(run, m = m)

            if write_SFRtoSWR:
                print('writing SFRtoSWR')
                SFRtoSWR.run(run_name=run, model_dir=model_dir)
                SFRtoSWR.plot_start_stage(None, os.path.join('versions', run),model_dir= model_dir)

            if copy_mod_files_for_website:
                print('copying model files to webiste folder')
                basic.copy_mod_files(run, path = model_dir)

        except Exception as e:
            warnings.warn(f"failed bc of\n{e}")
            print(traceback.format_exc())
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
            if post_process_hydrographs:
                Hydrographs.run(run_name=run, reload = True, ml = m)
            if post_process_SWR:
                postprocess.run(run, riv_only = True, m= m)
            if post_process_streamflow:
                Streamflows.run(out_folder=os.path.join('versions', run), ml=m)
            if post_process_head_maps:
                post_process_heads.run(run_name=run, head_frequency=14, ml = m, add_basemap=True)

            if do_zone_bud:
                zone_bud.run(run, ml = m)

            if set_starting_heads_after:
                p = os.path.join('initial_heads', run)
                initial_conditions.set_starting_heads(m, plot = False, alt_outpath=p)
            basic.write_run_name_to_file(run, 'completed post processing', mode='a')

        except Exception as e:
            warnings.warn(f"failed bc of\n{e}")
            print(traceback.format_exc())
            basic.write_run_name_to_file(run, str(e), mode='a')

    else:
        basic.write_run_name_to_file(run, 'failed', mode='a')

    plt.close()
    plt.close(plt.gcf())
    plt.close('all')
    plt.clf()

    print(f"Done with Run: {run}")

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