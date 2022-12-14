import basic
import geopandas as gpd
import pandas as pd
import numpy as np
import flopy
import pathlib
import matplotlib.pyplot as plt
import os

def run(name):
    print('running wel creation package')
    m = basic.load_model(name)

    info, swr_info, sfr_info, riv_keys_info = basic.load_params(name)

    datestart = info['start_date']
    numdays = info['numdays']

    out_folder = basic.out_folder(name)
    df = load_caissons()

    df = get_period(df, datestart,numdays)
    
    wells = load_wells()
    
    ts_data = get_well_info(wells, df)

    wel = mf_wel(m, ts_data)

    wel.write_file()

    plot_pumping(df, out_folder)


def load_wells():
    wells = gpd.read_file('GIS/wells.shp')
    wells.loc[wells.loc[:,'flux']==0,'flux'] = -1 # some wells have no flow info, so setting to 1.
    
    smf = wells.groupby('wellname').sum().loc[:,'flux'].rename('sumflux')
    wells = pd.merge(wells, smf, on = 'wellname')


    wells.loc[:,'frac'] = wells.loc[:,'flux']/wells.loc[:,'sumflux']
    
    return wells
    
def loadcaisson(path, caisson = 'Caisson1Flow.csv', name = 'Well1'):
    c = pd.read_csv(path.joinpath(caisson))
    c = c.set_index(pd.to_datetime(c.loc[:,'DateTime'])).loc[:,['Value']].rename(columns = {'Value':name})
    c = c.resample('1D').sum().applymap(basic.isnumber)
    
    return c

def load_pumps(path, name, pumps):
    p1 = loadcaisson(path,caisson = f'Pump{pumps[0]}Flow.csv', name = name)
    p2 = loadcaisson(path,caisson = f'Pump{pumps[1]}Flow.csv', name = name)
    
    p1 = p1.applymap(basic.isnumber)
    p2 = p2.applymap(basic.isnumber)
    
    ptot = p1+p2
    
    return ptot
    
    
def load_caissons():
    
    '''
    get well pumping in feet^3/day
    '''
    path = pathlib.Path(r"T:\arich\Russian_River\MirabelWohler_2022\Waterlevel_Data\MWs_Caissons - AvailableDailyAverages\DailyData\Caissons")
    
    c1 = loadcaisson(path, caisson = 'Caisson1Flow.csv', name = "well1")
    c2 = loadcaisson(path, caisson = 'Caisson2Flow.csv', name = "well2")
    c6 = loadcaisson(path, caisson = 'Caisson6Flow.csv', name = "well6")

    c3 = load_pumps(path, 'well3', [5,6])
    c4 = load_pumps(path, 'well4', [7,8])
    c5 = load_pumps(path, 'well5', [9,10])
    
    df = pd.concat([c1, c2, c3, c4, c5, c6], axis = 1).fillna(0.)
    
    df = df.astype({ic:np.float32 for ic in df.columns})
    
    df[df.abs().values>1e10] = 0
    
    df = df*1e6 / 7.480543 #from MGD to feet^3/day
    
    return df

def get_period(df, start_date, numdays, assign_per = True):


    df = df.loc[start_date:, :].iloc[:numdays]

    # df.loc[:, 'Value'] = df.loc[:, 'Value'].interpolate()

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
    
def plot_pumping(df, out_folder):
    ax = df.droplevel(1,0).rename(lambda x: pd.to_datetime(x).strftime("%b\n%d\n%Y")).mul(1/43560).plot.bar(
        stacked = True, figsize = (10,10), ylabel = 'acre-feet', grid = True, title = "Total Caisson Pumping, per Well")

    ax.set_xticks(ax.get_xticks()[::7])
    ax.tick_params(axis="x", rotation=0)
    
    plt.savefig(os.path.join(out_folder, 'pumping.png'), dpi=250, bbox_inches = 'tight')
    
def mf_wel(m, ts_data):
    
    stress_period_data = {}
    dtypes = flopy.modflow.ModflowWel.get_default_dtype()
    ts_data = ts_data.astype({'qcell':np.float32})
    # print(dtypes)
    # dtypes = dict(dtypes)
    # print(dtypes)
    for per, group in ts_data.loc[:,['per','k','i', 'j', 'qcell']].groupby('per'):
        # stress_period_data[per] = group.drop(columns = 'per').rename(
        #     columns = {'qcell':'flux'}).to_records(index = False, column_dtypes= dtypes)
        # stress_period_data[per] = np.rec.array(group.drop(columns = 'per').rename(
        #      columns = {'qcell':'flux'}).values, dtype = [('k', '<i4'), ('i', '<i4'), ('j', '<i4'), ('flux', '<f4')])
        # dfcur = group.drop(columns = 'per').rename(
        #      columns = {'qcell':'flux'})
        # stress_period_data[per] = np.rec.array([tuple(x) for x in dfcur.values.tolist()],
        #              dtype=[('k', '<i4'), ('i', '<i4'), ('j', '<i4'), ('flux', '<f4'), ])
        # print(stress_period_data[per].dtype)
        # sdf
        group = group.loc[group.loc[:, 'qcell'].abs() > 0, :]
        group = group.loc[group.loc[:,'qcell'].abs() > 1e-4,:] # without this threshold flopy makes a broken wel file for some reason
        sp = np.zeros(group.shape[0], dtype=dtypes)
        sp = sp.view(np.recarray)
        sp['i'] = group.loc[:,'i'].astype(int).values.tolist()
        sp['j'] = group.loc[:, 'j'].astype(int).values.tolist()
        sp['k'] = group.loc[:, 'k'].astype(int).values.tolist()
        sp['flux'] = group.loc[:, 'qcell'].astype(np.float32).values.tolist()
        stress_period_data[per] = sp



    wel = flopy.modflow.ModflowWel(m, ipakcb = 1,filenames ='RR.wel',
                               stress_period_data = stress_period_data)
    
    return wel
    
if __name__ == '__main__':
    print('running make wells')
    run('June2017')