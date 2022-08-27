#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import conda_scripts.arich_functions as af
import pathlib
import basic


def flo_dict():
    flow = dict()

    flow['Russian'] = dict(station='11464000', title='Russian River', file='RRinflow.dat',
                           figurename='russian_river.png',)
    flow['Dry'] = dict(station='11465350', title='Dry Creek', file='Dry_creek.dat', figurename='dry_creek.png', )
    flow['Mark'] = dict(station='11466800', title='Mark West Creek', file='MarkWest.dat', figurename='mark_west.png', )

    return flow

def run(model_name, minvalue = 29.54,
    max_value = 38,
    mid_cutoff = 32,
    flow_max = 300,
    numdays = 109):

    m = basic.load_model()

    info, swr_info, sfr_info, riv_keys_info = basic.load_params(model_name)
    
    datestart = info['start_date']

    name = info['name']

    out_folder = basic.out_folder(model_name)
    print(datestart)
    print(out_folder)

    numdays = info['numdays']

    basic.setup_folder(model_name)
    start_year = pd.to_datetime(datestart).year

    rr = load_riv(station='11464000', title='Russian River', file='RRinflow.dat', figurename='russian_river.png',
                  datestart = datestart, out_folder = out_folder, m = m, numdays=109, save_fig=True, write_output=True)

    dry = load_riv(station='11465350', title='Dry Creek', file='Dry_creek.dat', figurename='dry_creek.png',
                   datestart=datestart, out_folder=out_folder, m=m, numdays=109, save_fig=True, write_output=True)

    mw = load_riv(station='11466800', title='Mark West Creek', file='MarkWest.dat', figurename='mark_west.png',
                   datestart=datestart, out_folder=out_folder, m=m, numdays=109, save_fig=True, write_output=True)

    total = dry.loc[:, 'Q'] + rr.loc[:, 'Q']
    total = total.to_frame('rrtotal')

    stg = load_dam(total, datestart=datestart, minvalue=minvalue, max_value=max_value, numdays=109, flow_max=flow_max,
                   mid_cutoff=mid_cutoff)

    plot_dam(stg, minvalue=minvalue, max_value=max_value,
             mid_cutoff=mid_cutoff, out_folder = out_folder)

    f =    "1         1       0      11         51      9\n \
    116       1       0        6          0.61   0.5  {:}          200.00       0.1       1     56    1 #{:}\n"
    f.format(1,2)

    cnt = 0
    for ind, d in stg.iterrows():
        # print(d['Value'])
        # print(ind)
        name = f"RR_2022/ref/dam_stage/day{cnt}.dat"
        # print(name)
        with open(name,'w') as out:
            out.write(f.format(d['Value'], ind.strftime("%y %b %d")))

        cnt = cnt+1

    # cnt = 1
    # for ind, d in stg.iterrows():
    #
    #     print(
    #     r"# ITMP IRDBND IRDRAI IRDEVP IRDLIN IRDGEO IRDSTR IRDSTG IPTFLG [IRDAUX]   #\
    #     1   -1    0      0       0     -1      2      0       1      # 5              #\
    #     open/close 'ref\nstruct.dat'\n")
    #     print(f"    open/close 'ref\dam_stage\day{cnt}.dat")
    #
    #     cnt = cnt+1


def plot_dam(stg,minvalue, max_value,
             mid_cutoff, out_folder, save_fig = True):

    fig, (ax, ax1) = plt.subplots(2, 1, sharex=True, figsize=(8, 8))
    ax = stg.loc[:,['Value']].plot(marker='.', ax=ax)
    stg.loc[:,['Original_Value']].rename(columns = {'Original_Value': 'Observed RDS'}).plot(marker = '.', color = 'g', ax = ax)

    ax.axhline(minvalue, c='c', ls='-.', label='Min Dam Stage')
    ax.axhline(max_value, c='r', ls='-.', label='Max Dam Stage')
    # ax.axhline(mid_cutoff, c='pink', ls='--',
    #            label='mid_cutoff (dam values below\nbecome minvalue, above become max)')
    ax.legend(bbox_to_anchor=(1, 1), loc='upper left')
    ax1.set_title('Russian River Inflows')
    ax.set_title('Dam Elevations')
    stg.loc[:, 'rrtotal'].plot(ax=ax1, c='r')
    ax.grid(True)
    ax1.grid(True)
    #
    # lims = ax.get_xlim()
    #
    # landsat = pd.read_csv(r"T:\arich\Russian_River\RDS\Diversions\Pond_yearly_acres 2021.csv",index_col = [0])
    # landsat.index = pd.to_datetime(landsat.index)
    #
    # landsat = landsat.loc[:,['Dam Area']]
    # landsat.plot(ax = ax2, marker = 'o')
    # ax2.set_ylabel('Ponded area from landasy')
    #
    # ax2.set_xlim(lims)
    # ax2.set_ylim([0,20])

    if save_fig:
        plt.savefig(os.path.join(out_folder, 'dam_elevation.png'), bbox_inches='tight', dpi=250)


def load_dam(total, datestart, minvalue=29.54, max_value=38, numdays=109, flow_max=500, mid_cutoff=32):
    p = pathlib.Path(
        r"T:\arich\Russian_River\MirabelWohler_2022\Waterlevel_Data\MWs_Caissons - AvailableDailyAverages\DailyData")

    rds = 'RiverStageDaily.csv'

    stg = pd.read_csv(p.joinpath(rds), parse_dates=[0]).set_index('StartDateTime')

    stg.loc[stg.loc[:, 'Value'] > 50, 'Value'] = 50.
    stg.loc[:, 'Original_Value'] = stg.loc[:, 'Value'].copy()
    c = stg.loc[:, 'FillValue'].notnull()
    stg.loc[c, 'Value'] = stg.loc[c, 'FillValue']

    stg.loc[:, 'Value'] = stg.loc[:, 'Value'].interpolate()


    stg.loc[:, 'INTERP'] = stg.loc[:, 'INTERP'].replace({'UP': max_value, 'DOWN': minvalue})
    stg.loc[:, 'INTERP'] = stg.loc[:, 'INTERP'].fillna(method='ffill')
    stg.loc[:, 'INTERP'] = stg.loc[:, 'INTERP'].rolling(7).mean()

    stg.loc[:, "Value"] = stg.loc[:, 'INTERP']

    stg = stg.resample('1D').mean()

    stg = stg.loc[datestart:, :].iloc[:numdays]

    # stg.loc[stg.loc[:, 'Value'] < minvalue, 'Value'] = minvalue
    # stg.loc[stg.loc[:, 'Value'] > max_value, 'Value'] = max_value
    #
    stg = stg.join(total)
    #
    # stg.loc[stg.loc[:, 'rrtotal'] > flow_max, "Value"] = minvalue
    # stg.loc[stg.loc[:, 'Value'] > mid_cutoff, 'Value'] = max_value
    # stg.loc[stg.loc[:, 'Value'] < mid_cutoff, 'Value'] = minvalue

    # stg.loc[:,'Value'] = stg.loc[:,'Value'].rolling(5, min_periods=0).mean()

    assert stg.loc[:, 'Value'].isnull().sum() == 0, 'has nans'

    return stg


def load_riv(station, title, file , figurename,datestart, out_folder, m, numdays = 109, save_fig = True, write_output = True):
    start_year = pd.to_datetime(datestart).year

    flow, info = af.download_daily(station, start_year, begin_month=1)

    flow = flow.loc[datestart:, ].head(numdays)
    flow.plot(title= title)
    flow.loc[:, 'time'] = np.arange(flow.shape[0])
    flow = flow.loc[:, ['time', 'Q']]

    flow.loc[:, 'time'] = flow.loc[:, 'time'] * 86400

    assert flow.isnull().sum().sum() == 0, 'has nans'

    if write_output:
        file = os.path.join(m.model_ws, 'ref', file)

        with open(file, 'w', newline='') as wr:
            # wr.write('OFFSET 0.0 SCALE 1.0\n')
            flow.rename(columns={flow.columns[0]: '#' + flow.columns[0]}).to_csv(wr, sep='\t', index=False,
                                                                                 header=False,
                                                                                 mode='a', )
    if save_fig:
        plt.savefig(os.path.join(out_folder, figurename), dpi=250)

    return flow

#
# def dry_flow(datestart, out_folder, m, numdays = 109, save_fig = True, write_output = True):
#     start_year = pd.to_datetime(datestart).year
#
#     flow, info = af.download_daily('11465350', start_year, begin_month=1)
#     flow = flow.loc[datestart:, ].head(numdays)
#
#     flow.plot(title='Dry Creek')
#     flow.loc[:, 'time'] = np.arange(flow.shape[0])
#     flow = flow.loc[:, ['time', 'Q']]
#
#     flow.loc[:, 'time'] = flow.loc[:, 'time'] * 86400
#
#     assert flow.isnull().sum().sum() == 0, 'has nans'
#
#     if write_output:
#         file = os.path.join(m.model_ws,'ref','Dry_creek.dat')
#
#         with open(file,'w',newline='') as wr:
#             # wr.write('OFFSET 0.0 SCALE 1.0\n')
#             flow.rename(columns = {flow.columns[0]:'#'+flow.columns[0]}).to_csv(wr, sep = '\t',  index = False, header = False, mode='a', )
#
#     if save_fig:
#         plt.savefig(os.path.join(out_folder, 'dry_creek.png'), dpi = 250)
#
#     return flow


# def rrflow(datestart, out_folder, m, numdays = 109, save_fig = True, write_output = True):
#     start_year = pd.to_datetime(datestart).year
#
#     flow, info = af.download_daily('11464000', start_year, begin_month=1)
#
#     flow = flow.loc[datestart:, ].head(numdays)
#     flow.plot(title='Russian River')
#     flow.loc[:, 'time'] = np.arange(flow.shape[0])
#     flow = flow.loc[:, ['time', 'Q']]
#
#     flow.loc[:, 'time'] = flow.loc[:, 'time'] * 86400
#
#     assert flow.isnull().sum().sum() == 0, 'has nans'
#
#     if write_output:
#         file = os.path.join(m.model_ws, 'ref', 'RRinflow.dat')
#
#         with open(file, 'w', newline='') as wr:
#             # wr.write('OFFSET 0.0 SCALE 1.0\n')
#             flow.rename(columns={flow.columns[0]: '#' + flow.columns[0]}).to_csv(wr, sep='\t', index=False, header=False,
#                                                                              mode='a', )
#     if save_fig:
#         plt.savefig(os.path.join(out_folder, 'russian_river.png'), dpi=250)
#
#     return flow
#
#
# def markwest(datestart, out_folder, m, numdays = 109, save_fig = True, write_output = True):
#     start_year = pd.to_datetime(datestart).year
#
#     flow, info = af.download_daily('11466800', start_year, begin_month=1)
#     flow = flow.loc[datestart:, ].head(numdays)
#     flow.plot(title='Mark West Creek')
#     flow.loc[:, 'time'] = np.arange(flow.shape[0])
#     flow = flow.loc[:, ['time', 'Q']]
#
#     flow.loc[:, 'time'] = flow.loc[:, 'time'] * 86400
#
#     assert flow.isnull().sum().sum() == 0, 'has nans'
#
#     if write_output:
#         file = os.path.join(m.model_ws,'ref','MarkWest.dat')
#
#         with open(file,'w',newline='') as wr:
#             # wr.write('OFFSET 0.0 SCALE 1.0\n')
#             flow.rename(columns = {flow.columns[0]:'#'+flow.columns[0]}).to_csv(wr, sep = '\t',  index = False, header = False, mode='a', )
#
#     if save_fig:
#         plt.savefig(os.path.join(out_folder, 'mark_west.png'), dpi = 250)
#
#     return flow