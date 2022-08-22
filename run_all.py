import basic
import SFRtoSWR
import postprocess
import write_inflows
import Hydrographs
import post_process_heads
import matplotlib.pyplot as plt
import pond_inflows


# for run in ['June2012', 'June2013', 'June2014', 'June2015', 'June2016', 'June2017', 'June2018', 'June2019', 'June2020', 'calibration']:
# for run in ['June2018', 'June2019',  'June2020',]:
for run in [ 'June2020',]:

    # print(run)
    # basic.setup_folder(run)
    # write_inflows.run(model_name=run)
    # # SFRtoSWR.run(run)
    # pond_inflows.run(run)
    # m = basic.load_model()
    #
    # success, buffer = m.run_model(silent = False,)
    #
    # if success:
    #     Hydrographs.run(run_name=run, reload = False)
    #     postprocess.run(run)
    #
    #     post_process_heads.run(run_name=run, head_frequency=5)
    #
    # plt.close('all')


    Hydrographs.run(run_name=run, reload = False)
    postprocess.run(run)

    post_process_heads.run(run_name=run, head_frequency=5)
