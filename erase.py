import basic
import SFRtoSWR
import postprocess
import write_inflows
import Hydrographs
import post_process_heads
import matplotlib.pyplot as plt
import pond_inflows


for run in ['June2012']:





    post_process_heads.run(run_name=run, head_frequency=5)

    plt.close('all')
