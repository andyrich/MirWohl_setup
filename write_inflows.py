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
import warnings

def flo_dict():
    flow = dict()

    flow['Russian'] = dict(station='11464000', title='Russian River', file='RRinflow.dat',
                           figurename='russian_river.png',)
    flow['Dry'] = dict(station='11465350', title='Dry Creek', file='Dry_creek.dat', figurename='dry_creek.png', )
    flow['Mark'] = dict(station='11466800', title='Mark West Creek', file='MarkWest.dat', figurename='mark_west.png', )

    return flow

def run(model_name, m = None, minvalue = 29.54,
    max_value = 38,    numdays = None, datestart = None, cleandamdata = True, inflow_limit = 2000):
    '''
    run processing of river inflows, dam stage
    :param inflow_limit:
    :param model_name:
    :param m:
    :param minvalue:
    :param max_value:
    :param numdays:
    :param datestart:
    :param cleandamdata:
    :return: rr, dry, mw, total, stg
    '''

    if m is None:
        m = basic.load_model()

    info, swr_info, sfr_info, riv_keys_info = basic.load_params(model_name)

    nswr_cells = swr_info['SWR_ncells']
    if datestart is None:
        datestart = info['start_date']
    else:
        warnings.warn(f"Using supplied datestart ({datestart}), not that which is listed in the run_names.txt")

    if numdays is None:
        numdays = info['numdays']
    else:
        warnings.warn(f"Using supplied numdays ({numdays}), not that which is listed in the run_names.txt")

    name = info['name']

    out_folder = basic.out_folder(model_name)
    print(datestart)
    print(out_folder)

    basic.setup_folder(model_name)
    start_year = pd.to_datetime(datestart).year

    rr = load_riv(station='11464000', title='Russian River', file='RRinflow.dat', figurename='russian_river.png',
                  datestart = datestart, out_folder = out_folder, m = m, numdays=numdays, save_fig=True, write_output=True,
                  inflow_limit = inflow_limit)

    dry = load_riv(station='11465350', title='Dry Creek', file='Dry_creek.dat', figurename='dry_creek.png',
                   datestart=datestart, out_folder=out_folder, m=m, numdays=numdays, save_fig=True, write_output=True,
                   inflow_limit=inflow_limit)

    mw = load_riv(station='11466800', title='Mark West Creek', file='MarkWest.dat', figurename='mark_west.png',
                   datestart=datestart, out_folder=out_folder, m=m, numdays=numdays, save_fig=True, write_output=True,
                  inflow_limit=inflow_limit)

    total = dry.loc[:, 'Q'] + rr.loc[:, 'Q']
    total = total.to_frame('rrtotal')

    stg = load_dam(total, datestart=datestart, minvalue=minvalue, max_value=max_value, numdays=numdays, clean = cleandamdata)

    plot_dam(stg, minvalue=minvalue, max_value=max_value,
              out_folder = out_folder)

    f =    "1         1       0      11         51      25\n \
    {:}       1       0        6          0.5   0.5  {:}          200.00       0.5       1     55    1 #{:}\n"
    # f.format(1,2)

    cnt = 0
    for ind, d in stg.iterrows():
        # print(d['Value'])
        # print(ind)
        name = os.path.join(m.model_ws, f"ref/dam_stage/day{cnt}.dat")
        # print(name)
        with open(name,'w') as out:
            out.write(f.format(nswr_cells, d['Value'], ind.strftime("%y %b %d")))

        cnt = cnt+1

    return rr, dry, mw, total, stg


def plot_dam(stg,minvalue, max_value,
            out_folder, save_fig = True):

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

    if save_fig:
        plt.savefig(os.path.join(out_folder, 'dam_elevation.png'), bbox_inches='tight', dpi=250)


def load_dam(total, datestart, minvalue=29.54, max_value=38, numdays=109, clean = True):
    '''
    load the dam stage
    :param total: total russain flow from'run'
    :param datestart:
    :param minvalue: what to set open dam elevation at
    :param max_value: closed dam elevation
    :param numdays:
    :param clean: remove bad values >50 or <20
    :return: stg
    '''
    p = pathlib.Path(
        r"Waterlevel_Data\MWs_Caissons - AvailableDailyAverages\DailyData")

    rds = 'RiverStageDaily.csv'

    stg = pd.read_csv(p.joinpath(rds), parse_dates=[0]).set_index('StartDateTime')

    if clean:
        stg.loc[stg.loc[:, 'Value'] > 50, 'Value'] = 50.
        stg.loc[stg.loc[:, 'Value'] < 20, 'Value'] = 20.

    # print(stg.head())
    stg = stg[~stg.index.duplicated(keep='first')]
    # # stg = stg.resample('1D').mean()
    # # stg = stg.drop_duplicates()
    # print(stg.head())
    # stg = stg.reindex(pd.date_range(datestart, periods = numdays, freq = 'D'))
    # print(stg.head())
    # print(stg.head())

    stg.loc[:, 'Original_Value'] = stg.loc[:, 'Value'].copy()
    c = stg.loc[:, 'FillValue'].notnull()
    stg.loc[c, 'Value'] = stg.loc[c, 'FillValue']

    stg.loc[:, 'Value'] = stg.loc[:, 'Value'].interpolate()

    stg.loc[:, 'INTERP'] = stg.loc[:, 'INTERP'].replace({'UP': max_value, 'DOWN': minvalue})
    stg = stg.reindex(pd.date_range(datestart, periods=numdays, freq='D'))
    stg.loc[:, 'INTERP'] = stg.loc[:, 'INTERP'].fillna(method='ffill').fillna(method='bfill')
    stg.loc[:, 'INTERP'] = stg.loc[:, 'INTERP'].rolling(7, min_periods=0).mean()
    # print(stg.head(10))
    assert stg.loc[:, 'INTERP'].isnull().sum()== 0, f"there are nans in the INTERP value,\n{stg[stg.loc[:, 'INTERP'].isnull()].index}"

    stg.loc[:, "Value"] = stg.loc[:, 'INTERP']

    stg = stg.resample('1D').mean()

    end_days = pd.to_datetime(datestart) + pd.to_timedelta(numdays, unit="D")

    if pd.to_datetime(datestart) < stg.index.min():
        print('interpolating data points for dam records')
        # in special cases need to create dummy data for 2012 data
        stg = stg.reindex(index = pd.date_range(datestart, periods = numdays, freq = 'D'), method = 'nearest')
        stg = stg.bfill().ffill()

    stg = stg.loc[datestart:end_days, :]
    stg = stg.join(total)

    assert stg.loc[:, 'Value'].isnull().sum() == 0, 'stage has nans\n'\
                                                    f'shape of nans\n{stg.loc[:,"Value"].isnull().sum()}\n'\
                                                    f'stage nans are\n{stg.loc[stg.loc[:,"Value"].isnull()]}'

    return stg


def load_riv(station, title, file, figurename, datestart, out_folder, m, numdays = 109, save_fig = True,
             write_output = True, inflow_limit = 2000):
    '''
    load discharge from gage
    :param station: USGS number
    :param title: name of station
    :param file: model input filename
    :param figurename: figname
    :param datestart: srting like '1/1/2000'
    :param out_folder: folder name of plot
    :param m: model instance
    :param numdays: number of days of model run
    :param save_fig: save the figure
    :param write_output:
    :param inflow_limit: to limit stream inflows for model run time
    :return: df of flow values
    '''
    start_year = pd.to_datetime(datestart).year

    flow, info = af.download_daily(station, start_year, begin_month=1)

    flow_all = flow.copy()
    end_days = pd.to_datetime(datestart) + pd.to_timedelta(numdays, unit="D")
    flow = flow.loc[datestart:end_days,]
    flow = flow.reindex(index=pd.date_range(datestart, periods=numdays, freq='D'), method='nearest')

    if save_fig:
        ax = flow_all.iloc[0:365, :].rename(columns = {'Q': title + ' (all year)'}).plot(title = title)
        flow.rename(columns = {'Q': title + ' (Model Period)'}).plot(lw = 3, ax = ax)
        ax.legend()
        plt.savefig(os.path.join(out_folder, figurename), dpi=250)

    flow.loc[:, 'time'] = np.arange(flow.shape[0])
    flow = flow.loc[:, ['time', 'Q']]

    flow.loc[:, 'time'] = flow.loc[:, 'time'] * 86400

    assert flow.shape[0] == numdays, f'flow has incorrect shape\nshape is\n{flow.shape}\n{flow.head()}\n{flow.tail()}'
    assert flow.isnull().sum().sum() == 0, 'has nans'

    if write_output:
        file = os.path.join(m.model_ws, 'ref', file)

        with open(file, 'w', newline='') as wr:
            # wr.write('OFFSET 0.0 SCALE 1.0\n')
            ftemp = flow.copy()
            ftemp.loc[ftemp.loc[:,'Q']>inflow_limit,'Q'] = inflow_limit
            ftemp.rename(columns={ftemp.columns[0]: '#' + ftemp.columns[0]}).to_csv(wr, sep='\t', index=False,
                                                                                 header=False,
                                                                                 mode='a', )

    return flow
