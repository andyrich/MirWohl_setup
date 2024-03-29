import basic
import geopandas as gpd
import pandas as pd
import numpy as np
import flopy
import pathlib
import matplotlib.pyplot as plt
import os
import warnings

def run(name, m = None, numdays = None, datestart = None, write_output = True):
    '''
    create new mf wel file
    :param name:
    :param m:
    :param numdays:
    :param datestart:
    :return: df of wel timeseries
    '''

    if m is None:
        m = basic.load_model()

    info, swr_info, sfr_info, riv_keys_info = basic.load_params(name)

    if datestart is None:
        datestart = info['start_date']
    else:
        warnings.warn(f"Using supplied datestart ({datestart}), not that which is listed in the run_names.txt")

    if numdays is None:
        numdays = info['numdays']
    else:
        warnings.warn(f"Using supplied numdays ({numdays}), not that which is listed in the run_names.txt")


    print('running wel creation package')


    out_folder = basic.out_folder(name)
    df = load_caissons()

    df = get_period(df, datestart,numdays)
    
    wells = load_wells()
    
    ts_data = get_well_info(wells, df)

    if write_output:
        wel = mf_wel(m, ts_data)

        wel.write_file()

    plot_pumping(df, out_folder)

    return df


def load_wells():
    # data_folder = os.environ['data_folder']
    data_folder = basic.get_data_folder()
    wells = gpd.read_file(os.path.join(data_folder,'GIS/wells.shp'))
    wells.loc[wells.loc[:,'flux']==0,'flux'] = -1 # some wells have no flow info, so setting to 1.
    
    smf = wells.drop(columns = 'geometry').groupby('wellname').sum().loc[:,'flux'].rename('sumflux')
    wells = pd.merge(wells, smf, on = 'wellname')

    wells.loc[:,'frac'] = wells.loc[:,'flux']/wells.loc[:,'sumflux']
    
    return wells
    
def loadcaisson(path, caisson = 'Caisson1Flow.csv', name = 'Well1'):
    '''
    load the caissonflow timeseries
    :param path:
    :param caisson:
    :param name:
    :return: dataframe with series [name]
    '''
    c = pd.read_csv(path.joinpath(caisson))
    c = c.set_index(pd.to_datetime(c.loc[:,'DateTime'])).loc[:,['Value']].rename(columns = {'Value':name})
    c.index = pd.DatetimeIndex(c.index.date)
    c = c.applymap(basic.isnumber)
    c[c.abs().values > 1e10] = np.nan
    # interpolate up to 5 days
    c = c.interpolate(limit = 5)
    # for gaps bigger than 5 days use the mean for that month
    c = c.groupby(c.index.to_period('M').to_timestamp('M')).transform(lambda x: x.fillna(x.mean()))
    # for the remaing gaps, fill with interpolate
    c = c.interpolate()
    # c = c.resample('1D').sum().applymap(basic.isnumber)
    
    return c

def load_pumps(path, name, pumps):
    '''
    load given pump data [pumpp] and combine to [name]
    :param path:
    :param name:
    :param pumps:
    :return:
    '''
    p1 = loadcaisson(path,caisson = f'Pump{pumps[0]}Flow.csv', name = name)
    p2 = loadcaisson(path,caisson = f'Pump{pumps[1]}Flow.csv', name = name)
    
    p1 = p1.applymap(basic.isnumber)
    p2 = p2.applymap(basic.isnumber)
    
    ptot = p1+p2
    
    return ptot
    
    
def load_caissons():
    '''
`
    get well pumping in feet^3/day

    :return: df of all pumping records
    '''
    

    path = pathlib.Path(r"Waterlevel_Data\MWs_Caissons - AvailableDailyAverages\DailyData\Caissons")
    
    c1 = loadcaisson(path, caisson = 'Caisson1Flow.csv', name = "well1")
    c2 = loadcaisson(path, caisson = 'Caisson2Flow.csv', name = "well2")
    c6 = loadcaisson(path, caisson = 'Caisson6Flow.csv', name = "well6")

    c3 = load_pumps(path, 'well3', [5, 6])
    c4 = load_pumps(path, 'well4', [7, 8])
    c5 = load_pumps(path, 'well5', [9, 10])
    
    df = pd.concat([c1, c2, c3, c4, c5, c6], axis = 1).fillna(0.)
    
    df = df.astype({ic: np.float32 for ic in df.columns})
    
    df[df.abs().values> 1e10] = 0
    
    df = df*1e6 / 7.480543 #from MGD to feet^3/day
    
    return df

def get_period(df, start_date, numdays, assign_per = True):
    '''

    :param df:
    :param start_date:
    :param numdays:
    :param assign_per:
    :return:
    '''

    # end_days = pd.to_datetime(start_date) + pd.to_timedelta(numdays, unit="D")
    df = df.resample("1D").mean()
    df = df.reindex(index = pd.date_range(start_date, periods = numdays, freq = 'D'))
    df = df.bfill().ffill()

    # there shouldn't be any gaps, but if there are, fill with monthly mean for that month from entire record for that well
    c = df.sum(axis=1) == 0
    df.loc[c, :] = np.nan
    df = df.groupby(df.index.month).transform(lambda x: x.fillna(x.mean()))
    # c = df.sum(axis=1) == 0
    #
    # df.loc[c,:] = df.mean().mean()

    assert (df.sum(axis=1) == 0).sum() == 0, f"there are {(df.sum(axis=1) == 0).sum()} days with zero Q values\n" \
                                           f"the df looks like:\n{df.head()}\n{df.tail()}\n"
    assert df.index.nunique()==numdays, f'repeating index values in df. nunique=={df.index.nunique()}'
    assert df.shape[0]==numdays, f'shape of df is wrong, should be {numdays} but is {df.shape[0]}'
    assert df.isnull().sum().sum() == 0, 'has nans'
    assert df.index.to_series().diff().nunique()==1, 'index has missing days'

    if assign_per:
        df.loc[:,'per'] = np.arange(numdays)
    else:
        df.loc[:,'per'] = np.arange(df.shape[0])

    df = df.set_index('per', append = True)
    
    return df

def get_well_info(wells, timeseries):
    
    wells_info = pd.merge(wells, timeseries.droplevel(0,0).T.stack().to_frame('Q').reset_index(),
                          left_on = 'wellname', right_on = 'level_0').sort_values('per')
    
    wells_info.loc[:,'qcell'] = -1 * wells_info.loc[:,'Q'].abs() * wells_info.loc[:,'frac'] / (24*60*60)
    # wells_info = wells_info.astype({'qcell':'<f4'})

    # dtypes = flopy.modflow.ModflowWel.get_default_dtype()
    # wells_info = wells_info.to_records(index = False, column_dtypes = dtypes)

    # print(wells_info.dtypes)
    
    return wells_info


def _plot_bar(df):
    '''
    plot well pumping df
    :param df:
    :return:
    '''
    fig, ax = plt.subplots(figsize=(9, 6))

    bottom = df.copy()
    bottom.insert(0, 0, 0)
    # modify the base of each columns, can do with a for loop
    for col in df.columns:
        bot = bottom.cumsum(axis=1).shift(axis=1).loc[:, col]

        ax.bar(df.droplevel(1, 0).index, df.droplevel(1, 0).cumsum(axis=1).loc[:, col].values, bottom=bot, label=col)

    ax.grid(True)
    ax.set(ylabel='acre-feet')
    basic.set_dates_xtick(ax)

    ax.legend()

    return fig, ax

def plot_pumping(df, out_folder):

    fig, ax = _plot_bar(df.mul(1 / 43560))
    ax.set_title("Total Caisson Pumping, per Well")
    plt.savefig(os.path.join(out_folder, 'pumping.png'), dpi=250, bbox_inches = 'tight')

    fig, ax = _plot_bar(df.cumsum().mul(1 / 43560))
    ax.set_title("Total Cumulative Caisson Pumping, per Well")
    plt.savefig(os.path.join(out_folder, 'pumping_cum.png'), dpi=250, bbox_inches='tight')

def mf_wel(m, ts_data):
    
    stress_period_data = {}
    dtypes = flopy.modflow.ModflowWel.get_default_dtype()
    ts_data = ts_data.astype({'qcell':np.float32})

    for per, group in ts_data.loc[:,['per','k','i', 'j', 'qcell']].groupby('per'):
        group = group.loc[group.loc[:, 'qcell'].abs() > 0, :]
        group = group.loc[group.loc[:,'qcell'].abs() > 1e-4,:] # without this threshold flopy makes a broken wel file for some reason
        sp = flopy.modflow.ModflowWel.get_empty(group.shape[0])
        # sp = np.zeros(group.shape[0], dtype=dtypes)
        # sp = sp.view(np.recarray)
        sp['i'] = group.loc[:,'i'].astype(int).values.tolist()
        sp['j'] = group.loc[:, 'j'].astype(int).values.tolist()
        sp['k'] = group.loc[:, 'k'].astype(int).values.tolist()
        sp['flux'] = group.loc[:, 'qcell'].astype(np.float32).values.tolist()
        stress_period_data[per] = sp

    wel = flopy.modflow.ModflowWel(m, ipakcb = 1, filenames ='RR.wel',
                               stress_period_data = stress_period_data)
    
    return wel
    
if __name__ == '__main__':
    print('running make wells')
    run('June2017')