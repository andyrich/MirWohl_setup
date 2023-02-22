#!/usr/bin/env python
# coding: utf-8

# In[1]:


import flopy
import basic
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import geopandas as gpd
import flopy.utils.mflistfile as mflist
import pathlib


def run(model_name, ponds_only = False, riv_only = False, plot_buds = True, m = None, numdays = 365):
    '''

    :param model_name:
    :param ponds_only: plot only ponds
    :param riv_only: plot only SWR reaches
    :param plot_buds: plot the SFR/MODFLOW/SWR budgets
    :return:
    '''

    if m is None:
        m = basic.load_model(verbose = False)

    info, swr_info, sfr_info, riv_keys_info = basic.load_params(model_name)

    datestart = info['start_date']

    numdays = info['numdays']

    out_folder = basic.out_folder(model_name)

    print(datestart)
    print(out_folder)

    if ponds_only:
        r = [False]
    elif riv_only:
        r = [True]
    else:
        r = [True, False]

    for remove_ponds in r:
        ISWRPQAQ, ISWRPRGF, ISWRPSTG, ISWRPSTR, ISWRPQM = SWR(m, datestart,
                                                              remove_ponds = remove_ponds)
        show_stats(ISWRPQAQ, out_folder, remove_ponds=remove_ponds)

        plot_stages(ISWRPQAQ, out_folder, remove_ponds = remove_ponds)

        show_stages_indiv(ISWRPSTG,out_folder, remove_ponds = remove_ponds)

        plot_swr_reach_all(ISWRPRGF, out_folder, remove_ponds = remove_ponds)

        plot_inflow_outflows(ISWRPQM, out_folder, remove_ponds = remove_ponds)

        if not remove_ponds:
            a,b = plot_ponds(m, datestart, out_folder, numdays=numdays, ISWRPQAQ = ISWRPQAQ)
        if remove_ponds:
            plot_rds_stage(m, datestart, out_folder, numdays=numdays, ISWRPQAQ = ISWRPQAQ)

    show_structure(ISWRPSTR, out_folder)

    if plot_buds:
        plot_budgets(datestart, m, out_folder)

    print('done with budget post-processing')
    return ISWRPQAQ, ISWRPRGF, ISWRPSTG, ISWRPSTR, ISWRPQM

def plot_rds_stage(m, datestart, out_folder, numdays = 365, ISWRPQAQ = None):
    p = pathlib.Path(
        r"Waterlevel_Data\MWs_Caissons - AvailableDailyAverages\DailyData")

    rds = 'RiverStageDaily.csv'

    stg = pd.read_csv(p.joinpath(rds), parse_dates=[0]).set_index('StartDateTime')
    stg = stg.loc[:,['Value']].rename(columns = {'Value':'Observed'})

    stg = stg.resample('1D').mean()

    stg = stg.loc[datestart:, :].iloc[:numdays]

    stg.loc[:, 'Observed'] = stg.loc[:, 'Observed'].interpolate()

    if ISWRPQAQ is None:
        ISWRPQAQ, ISWRPRGF, ISWRPSTG, ISWRPSTR, ISWRPQM = SWR(m, datestart, remove_ponds=True)

    ISWRPQAQ = ISWRPQAQ.loc[:, 'STAGE'].droplevel([1, 2, 3, 4, 6]).groupby(level = [0,1]).mean().unstack()
    ISWRPQAQ = ISWRPQAQ.loc[:, ISWRPQAQ.columns.isin([116])].rename(columns = {116: "Reach 116, simulated"})

    ax = ISWRPQAQ.plot(figsize=(9, 6), title='RDS Water Level', color='b')

    stg[stg<100].plot(ax = ax, color = 'k')

    # add manually observed river stage
    year = pd.to_datetime(datestart).year
    rds_phist = basic.load_river_report(year)
    rds_phist = rds_phist.loc[:, 'River Diversion'].loc[:, 'River Level']
    rds_phist = rds_phist.replace({'N/D': np.nan}).astype(float)
    rds_phist.plot(ax = ax, c = 'g', label = 'RDS (manually observed)')

    plt.savefig(os.path.join(out_folder, 'rds_stage.png'), dpi=250, bbox_inches='tight')

def plot_ponds(m, datestart, out_folder, numdays = 365, ISWRPQAQ = None):
    if ISWRPQAQ is None:
        ISWRPQAQ, ISWRPRGF, ISWRPSTG, ISWRPSTR, ISWRPQM = SWR(m, datestart, remove_ponds=False)

    ponds = gpd.read_file('ponds/ponds.geojson')

    ponds = ponds.drop_duplicates('name').loc[:, ['rno', 'name']]

    ISWRPQAQ = ISWRPQAQ.loc[:, 'DEPTH'].droplevel([1, 2, 3, 4, 6]).unstack()

    ISWRPQAQ = ISWRPQAQ.loc[:, ISWRPQAQ.columns.isin(ponds.loc[:,'rno'])]

    ponds.loc[:,'name'] = ponds.loc[:,'name'].replace({'One':1, 'Two':2, 'Three':3, 'Four':4})
    rename = ponds.drop_duplicates('name').loc[:, ['rno', 'name']].set_index('rno').to_dict()['name']

    ISWRPQAQ = ISWRPQAQ.rename(columns = rename)
    ISWRPQAQ = ISWRPQAQ.loc[:,[1,2,3,4]].rename(columns = lambda x:'Pond'+ str(x) + ' (Mod)')
    # ISWRPQAQ = ISWRPQAQ.droplevel([1, 2, 3, 4])
    ax = ISWRPQAQ.plot(figsize = (15, 10), subplots = True, title = 'Pond Water Levels', color = 'b')

    # get water levels
    wl = pd.DataFrame()
    for pond in [1, 2, 3, 4]:
        p = pathlib.Path(
            r"Waterlevel_Data\MWs_Caissons - AvailableDailyAverages\DailyData\InfiltrationPonds")

        p = p.joinpath(f"Pond{pond}WaterLevel.csv")

        c = pd.read_csv(p)
        c.loc[:, 'Value'] = c.loc[:, 'Value'].apply(isnumber)
        c = c.astype({'Value': np.float64})
        c = c.set_index(pd.to_datetime(c.loc[:, 'DateTime']))
        c = c.loc[:, ['Value']].resample('1D').mean()
        c = c.rename(columns={"Value": f"Pond{pond}"})
        wl = wl.join(c, how='outer')

    wl[wl.abs() > 100] = np.nan
    wl[wl < 0] = 0.

    wl = wl.interpolate('spline', order=2)
    stop = pd.to_datetime(datestart) + pd.to_timedelta(numdays, unit = 'D')
    wl = wl.loc[datestart:stop]
    wl.plot(ax = ax, subplots = True, color = 'k')

    plt.savefig(os.path.join(out_folder, 'pond_wl.png'), dpi=250, bbox_inches='tight')

    return ISWRPQAQ, wl

def isnumber(x):
    try:
        float(x)
        return x
    except:
        return np.nan

def plot_budgets(datestart, m, out_folder):
    print('extracting budgets, errors, etc.')
    plot_error(datestart, m, out_folder)
    plot_swr_bud(m, datestart, out_folder)

    sfrsum, sfr = plot_sfr_bud(datestart,m,out_folder)

    sfrsum, sfr = plot_sfr_bud(datestart,m, out_folder, [51, 56])


def shorten_legend(ax, n=20, hand = None, lab = None):

    if hand is None:
        hand, lab = ax.get_legend_handles_labels()

    hand, lab = np.array(hand)[np.linspace(0,len(hand)-1,n,dtype = int)], np.array(lab)[np.linspace(0,len(hand)-1,n,dtype = int)]

    return hand, lab


# In[11]:


def plot_error(datestart, m, out_folder):
    plist = os.path.join(m.model_ws,  'Results','RRlist.lst')

    mf_list = mflist.MfListBudget(plist)
    incremental, cumulative = mf_list.get_budget()
    df_in, df_out = mf_list.get_dataframes(start_datetime=datestart)
    ax = df_out.loc[:,['PERCENT_DISCREPANCY']].plot(grid = True)
    ax.set_title('Cumulative Water Budget Error')

    plt.savefig(os.path.join(out_folder, 'bud_cum_err.png'),dpi = 250, bbox_inches = 'tight')

    ax = df_in.loc[:,['PERCENT_DISCREPANCY']].plot(grid = True)
    ax.set_title('Instantaneous Water Budget Error')
    plt.savefig(os.path.join(out_folder, 'bud_inst_err.png'),dpi = 250, bbox_inches = 'tight')

    ax = df_in.loc[:,['SWR_LEAKAGE_IN']].join(df_out.loc[:,['SWR_LEAKAGE_OUT']]).plot()

    ax = df_in.filter(regex = 'STORAGE').plot()

    plt.savefig(os.path.join(out_folder, 'bud_leakage.png'),dpi = 250, bbox_inches = 'tight')

    plt.figure()
    dftot = df_in.drop(columns = df_in.filter(regex = 'STORAGE|TOTAL').columns).filter(regex = '_IN|_OUT')

    ax = dftot.sum(axis=1).plot()
    ax.set_title('Total Water Balance')

    plt.savefig(os.path.join(out_folder, 'bud_basic.png'),dpi = 250, bbox_inches = 'tight')

    # w budget
    df_filt = df_in.drop(columns=df_in.filter(regex='TOTAL|STORAGE|PERCENT|IN-OUT')).drop(
        columns=df_in.loc[:, df_in.abs().sum() == 0].columns)
    df_filt.filter(regex='_IN').plot(subplots=True, figsize=(10, 10), title='GW budget outflows')
    plt.savefig(os.path.join(out_folder, 'bud_gw_basic_inflows.png'), dpi=250, bbox_inches='tight')

    ax = df_filt.filter(regex='_OUT').plot(subplots=True, figsize=(10, 10), title='GW budget outflows')
    plt.savefig(os.path.join(out_folder, 'bud_gw_basic_outflows.png'), dpi=250, bbox_inches='tight')

    return df_in, df_out



def plot_swr_bud(m, datestart, out_folder):
    plist = os.path.join(m.model_ws,   'Results','RRlist.lst')
    swrlist = mflist.SwrListBudget(plist,)

    incremental , cum = swrlist.get_dataframes(start_datetime=datestart)

    incremental.loc[:, cum.sum().abs() > 0].resample("1D").sum().filter(regex='FLOW').plot(subplots=True,
                                                                                           figsize=(8, 8),
                                                                                           title='SWR Budget')

    plt.savefig(os.path.join(out_folder, 'bud_swr_basic.png'),dpi = 250, bbox_inches = 'tight')




def plot_sfr_bud(date_start, m,out_folder,  seg_filter = None, ):
    plist = os.path.join(m.model_ws,  'Results','sfr_output.cbc')

    if not os.path.exists(plist):
        return None, None

    sfrbud = flopy.utils.SfrFile(plist)

    sfr = sfrbud.df
    sfr.loc[:,'timestep'] = sfr.kstpkper.apply(lambda x:x[0])
    sfr.loc[:,'stressperiod'] = sfr.kstpkper.apply(lambda x:x[1])

    sfr.loc[:,'date'] = pd.to_datetime(date_start) + pd.to_timedelta(sfr.loc[:,'stressperiod'], unit = 'D')

    sfrsum = sfr.groupby(['segment', 'date']).sum().filter(regex = 'Q').unstack(['segment'])

    if seg_filter is None:
        print('showing all segments')
    else:
        print(f"filtering to {seg_filter}")
        sfrsum = sfrsum.iloc[:, sfrsum.columns.get_level_values(1).isin(seg_filter)]

    fig, axes = plt.subplots(2,3, sharex = True, figsize = (10,10))
    axes = axes.ravel()
    for cnt, col in enumerate(sfrsum.columns.get_level_values(0).unique()):
        sfrsum.loc[:,col].plot(title = col, marker = 'o', ax = axes[cnt])
        if seg_filter is None:
            axes[cnt].legend().remove()


    plt.tight_layout()

    print(f" the SFR flow totals are:\n{sfrsum.stack(1).sum()}")

    reaches =  '_'.join([str(x) for x in seg_filter]) if seg_filter is not None else 'all'
    filename = f'sfr_flows_reaches_{reaches}.png'
    print(filename)
    plt.savefig(os.path.join(out_folder, filename),dpi = 250, bbox_inches = 'tight')

    return sfrsum, sfr

def set_time(df, start_date):
    col = df.filter(regex = 'TIME').columns[0]
    df.loc[:,col] = pd.to_timedelta(df.loc[:,col].astype(int), unit = 's') + pd.to_datetime(start_date)

    return df


def check_len(file):
    file_size = os.path.getsize(file)

    return file_size>100.

def SWR(m, start_date, max_reach = 116, remove_ponds = True  ):
    path = os.path.join(m.model_ws, 'Results')


    if check_len(os.path.join(path,'ISWRPQM.dat')):
        ISWRPQM = pd.read_csv(os.path.join(path,'ISWRPQM.dat')).rename(columns = lambda x: x.strip())
        print(ISWRPQM.columns)
        ISWRPQM = filter_ponds(ISWRPQM, remove_ponds = remove_ponds, max_reach = max_reach)
        ISWRPQM = set_time(ISWRPQM, start_date)
        ISWRPQM = ISWRPQM.set_index(['TOTIME','SWRDT','KPER','KSTP','KSWR', 'RCHGRP'])
    else:
        ISWRPQM = None

    if check_len(os.path.join(path,'ISWRPQAQ.dat')):
        ISWRPQAQ = pd.read_csv(os.path.join(path,'ISWRPQAQ.dat'),header = None, skiprows  =1).drop([0])
        c = ['TOTIME','SWRDT','KPER','KSTP','KSWR','REACH','LAYER','GBELEV','STAGE','DEPTH','HEAD','WETPERM','CONDUCT','HEADDIFF','QAQFLOW','na']
        ISWRPQAQ.columns = c
        ISWRPQAQ = ISWRPQAQ.drop(columns = 'na')
        ISWRPQAQ = filter_ponds(ISWRPQAQ, remove_ponds=remove_ponds, max_reach=max_reach)
        ISWRPQAQ = set_time(ISWRPQAQ, start_date)
        ISWRPQAQ = ISWRPQAQ.set_index(["TOTIME","SWRDT","KPER","KSTP","KSWR","REACH","LAYER"])
    else:
        ISWRPQAQ = None

    if check_len(os.path.join(path,'ISWRPRGF.dat')):
        ISWRPRGF = pd.read_csv(os.path.join(path,'ISWRPRGF.dat'),index_col =False).rename(columns = lambda x: x.strip())
        ISWRPRGF = set_time(ISWRPRGF, start_date)
        ISWRPRGF = filter_ponds(ISWRPRGF, remove_ponds=remove_ponds, max_reach=max_reach)
        ISWRPRGF = ISWRPRGF.set_index(['TOTTIME','SWRDT','KPER','KSTP','KSWR', 'RCHGRP'])
        ISWRPRGF = ISWRPRGF.dropna(axis=1, how = 'any')
    else:
        ISWRPRGF = None

    if check_len(os.path.join(path,'ISWRPSTG.dat')):
        ISWRPSTG = pd.read_csv(os.path.join(path,'ISWRPSTG.dat')).rename(columns = lambda x: x.strip())
        ISWRPSTG = ISWRPSTG.dropna(axis=1, how='all')
        ISWRPSTG = set_time(ISWRPSTG, start_date)
        ISWRPSTG = ISWRPSTG.set_index(['TOTIME', 'SWRDT', 'KPER', 'KSTP', 'KSWR'])
        c = [int(x.replace('STAGE', '').lstrip('0')) for x in ISWRPSTG.columns]
        ISWRPSTG.columns = c

        if remove_ponds:
            c = np.array(c) <= max_reach
            ISWRPSTG = ISWRPSTG.loc[:,c]
        else:
            c = np.array(c) > max_reach
            ISWRPSTG = ISWRPSTG.loc[:,c]
    else:
        ISWRPSTG = None

    if check_len(os.path.join(path,'ISWRPSTR.dat')):
        ISWRPSTR = pd.read_csv(os.path.join(path,'ISWRPSTR.dat')).rename(columns = lambda x: x.strip())
        ISWRPSTR = set_time(ISWRPSTR, start_date)
        ISWRPSTR = filter_ponds(ISWRPSTR, remove_ponds=remove_ponds, max_reach=max_reach)
        ISWRPSTR = ISWRPSTR.set_index(['TOTIME','SWRDT','KPER','KSTP','KSWR', 'REACH'])
    else:
        ISWRPSTR = None

    return ISWRPQAQ , ISWRPRGF, ISWRPSTG, ISWRPSTR, ISWRPQM

def filter_ponds(df, max_reach, remove_ponds = True):
    if remove_ponds:
        if 'REACH' in df.columns:
            df = df.query(f"REACH<={max_reach}")
        else:
            df = df.query(f"RCHGRP<={max_reach}")
    else:
        if 'REACH' in df.columns:
            df = df.query(f"REACH>{max_reach}")
        else:
            df = df.query(f"RCHGRP>{max_reach}")

    return df


def show_structure(ISWRPSTR, out_folder):

    swrZ_m = ISWRPSTR.groupby(['TOTIME','REACH']).mean()

    for reach in swrZ_m.index.get_level_values("REACH").unique():
        fz = swrZ_m.loc[swrZ_m.index.get_level_values('REACH')==reach,:].droplevel(1,0)
        fz.plot(subplots = True, figsize = (8,8), title = f"Stream Stages ISWRPSTR\nReach {reach}")


        plt.savefig(os.path.join(out_folder,f'SWR_ISWRPSTR_Structures_{reach}.png'), bbox_inches = 'tight')


def rename_output(name, remove_ponds = True):
    '''
    function to rename outputs depenind on if ponds are in outputs

    :param name:
    :param remove_ponds:
    :return:
    '''
    if not remove_ponds:
        name = name.split('.')[0]+'_ponds.'+name.split('.')[1]
        name = name.replace('..','.')
        print(f'renaming to: {name}')


    return name

def show_stats(ISWRPQAQ, out_folder, remove_ponds = True):
    f = ISWRPQAQ.groupby(['KPER']).describe()
    name = rename_output(f'ISWRPQAQ_stats.xlsx', remove_ponds=remove_ponds)
    f.to_excel(os.path.join(out_folder, name))
    name = rename_output(f'ISWRPQAQ_stats.html', remove_ponds=remove_ponds)
    f.to_html(os.path.join(out_folder, name))



def plot_stages(ISWRPQAQ, out_folder, remove_ponds = True):
    fig, ax = plt.subplots(4,2,figsize = (20,20), sharex = True)
    from matplotlib.pyplot import cm

    n = len(ISWRPQAQ.index.get_level_values('REACH').unique())
    color = cm.rainbow(np.linspace(0, 1, n))


    for  reach, c in zip(ISWRPQAQ.index.get_level_values('REACH').unique(), color):
        ISWRPQAQ.query(f"REACH=={reach}").groupby(['TOTIME']).mean().plot(subplots = True, ax =ax.ravel(), color = c, legend = False)

    [(axis.set_title(x),axis.set_facecolor("lightgrey")) for axis,x in zip(ax.ravel(), ISWRPQAQ.columns.tolist())]

    axi = ax[0,1]
    axi.legend()
    handles, lab = axi.get_legend_handles_labels()

    handles, lab = shorten_legend(ax, hand = handles, lab = list(ISWRPQAQ.index.get_level_values('REACH').unique()))

    axi.legend(handles, lab,
               loc = 'upper left', bbox_to_anchor = (1,1))
    name = rename_output(f'ISWRPQAQ.png', remove_ponds=remove_ponds)
    plt.savefig(os.path.join(out_folder,name), bbox_inches = 'tight')

    return

def show_stages_indiv(ISWRPSTG, out_folder, remove_ponds = True):

    import cycler

    if ISWRPSTG.groupby(['TOTIME']).mean().shape[1]>10:
        subplots = False
    else:
        subplots = True

    n = len(ISWRPSTG.columns.unique())
    color = plt.cm.viridis(np.linspace(0, 1,n))

    with  plt.rc_context({"axes.prop_cycle": cycler.cycler('color', color)}):

        ax = ISWRPSTG.groupby(['TOTIME']).mean().plot(subplots = subplots, figsize = (20,20))

    ax.set_facecolor("lightgrey")
    ax.legend(loc = 'upper left', bbox_to_anchor = (1,1))

    hand, lab = shorten_legend(ax, n = 20)

    ax.legend(hand, lab, loc = 'upper left', bbox_to_anchor = (1,1))
    plt.suptitle(f"Stream Stages")

    name = rename_output(f'SWR_STREAM STAGES.png', remove_ponds=remove_ponds)

    plt.savefig(os.path.join(out_folder,name), bbox_inches = 'tight')




def plot_swr_reach_all(ISWRPRGF, out_folder, remove_ponds = True):


    fig, ax = plt.subplots(int(np.ceil(ISWRPRGF.shape[1]/2)),2,figsize = (20,20), sharex = True)
    ax = ax.ravel()[:ISWRPRGF.shape[1]]
    from matplotlib.pyplot import cm

    n = len(ISWRPRGF.index.get_level_values('RCHGRP').unique())

    color = cm.rainbow(np.linspace(0, 1, n))


    for  reach, c in zip(ISWRPRGF.index.get_level_values('RCHGRP').unique(), color):
        ISWRPRGF.query(f"RCHGRP=={reach}").groupby(['TOTTIME']).mean().plot(subplots = True, ax =ax.ravel(), color = c, legend = False)

    [(axis.set_title(x),axis.set_facecolor("lightgrey")) for axis,x in zip(ax.ravel(), ISWRPRGF.columns.tolist())]

    axi = ax[1]
    axi.legend()
    hand, lab = axi.get_legend_handles_labels()
    lab = list(ISWRPRGF.index.get_level_values('RCHGRP').unique())
    # axi.legend(handles, ,
    #            loc = 'upper left', bbox_to_anchor = (1,1))

    hand, lab = shorten_legend(axi, n = 40, hand = hand, lab = lab )

    axi.legend(hand, lab,
               loc = 'upper left', bbox_to_anchor = (1,1))

    name = rename_output(f'SWR_reach_all STAGES.png', remove_ponds=remove_ponds)
    plt.savefig(os.path.join(out_folder,name), bbox_inches = 'tight')


def plot_inflow_outflows(ISWRPQM, out_folder, remove_ponds = True):


    fig, ax = plt.subplots(6,6, sharex = True, figsize = (20,20))

    count = 0
    for reach_time, gdf in ISWRPQM.query("RCHGRP<37 ").groupby(['RCHGRP']):
        g = gdf.set_index('CONNREACH', drop = True, append = True).drop(
            columns = ['REACHC', 'VLATFLOW']).unstack('CONNREACH').groupby('TOTIME').mean().droplevel(0,1)

        g.plot(ax = ax.ravel()[count])
        ax.ravel()[count].legend(title = None)

        ax.ravel()[count].set_title(f"Reach {reach_time}")
        ax.ravel()[count].set_facecolor("lightgrey")
        count = count+1

    name = rename_output(f'reach_flows.png', remove_ponds=remove_ponds)
    plt.savefig(os.path.join(out_folder,name), bbox_inches = 'tight')

    #### next
    fig, ax = plt.subplots(6,8, sharex = True, figsize = (20,20))

    count = 0
    for reach_time, gdf in ISWRPQM.query("RCHGRP>=37 & RCHGRP<=80 ").groupby(['RCHGRP']):
        g = gdf.set_index('CONNREACH', drop = True, append = True).drop(columns = ['REACHC', 'VLATFLOW']).unstack('CONNREACH').groupby('TOTIME').mean().droplevel(0,1)

        g.plot(ax = ax.ravel()[count])
        ax.ravel()[count].legend(title = None)

        ax.ravel()[count].set_title(f"Reach {reach_time}")
        ax.ravel()[count].set_facecolor("lightgrey")
        count = count+1

    name = rename_output(f'reach_flows_mid.png', remove_ponds=remove_ponds)
    plt.savefig(os.path.join(out_folder,name), bbox_inches = 'tight')

    fig, ax = plt.subplots(6,6, sharex = True, figsize = (20,20))

    count = 0
    for reach_time, gdf in ISWRPQM.query("RCHGRP>80").groupby(['RCHGRP']):
        g = gdf.set_index('CONNREACH', drop = True, append = True).drop(columns = ['REACHC', 'VLATFLOW']).unstack('CONNREACH').groupby('TOTIME').mean().droplevel(0,1)

        g.plot(ax = ax.ravel()[count])
        ax.ravel()[count].legend(title = None)

        ax.ravel()[count].set_title(f"Reach {reach_time}")
        ax.ravel()[count].set_facecolor("lightgrey")
        count = count+1

    name = rename_output(f'reach_flows_bot.png', remove_ponds=remove_ponds)
    plt.savefig(os.path.join(out_folder,name), bbox_inches = 'tight')



def plot_too_many(ISWRPRGF):
    from matplotlib.pyplot import cm
    print(f"write toomany to proceed with plotting {ISWRPRGF.shape[1]} plots")
    f = input()=='toomany'

    if f:
        print('doing plots... hold tight\n')
        print(ISWRPRGF.columns)
        cols = ISWRPRGF.columns

        allcols = []
        for c in cols:
            print(f"write 1 if {c} should be plotted")
            ccc = input()=='1'
            if ccc:
                allcols.extend([c])

        print(f'will plot {allcols}')

        for col in allcols:
            print(f"plotting {col}")
            fig, ax = plt.subplots(11,11, sharex = True, sharey = True, figsize = (20,20))
            count = 0
            fig.suptitle(col)


            # count = 0
            for reach_time, gdf in ISWRPRGF.groupby(['RCHGRP']):
                g = gdf.groupby('TOTTIME').sum().loc[:,col]

                g.plot(ax = ax.ravel()[count])
                ax.ravel()[count].legend(title = None).remove()

                ax.ravel()[count].set_title(f"Reach {reach_time}")
                ax.ravel()[count].set_facecolor("lightgrey")
                # ax.ravel()[count].legend().
                count = count+1
            plt.savefig(os.path.join(f'SWR_Processing/indiv_plots/SWR_reach_{col}.png'), bbox_inches = 'tight')
    else:
        print('not plotting')

