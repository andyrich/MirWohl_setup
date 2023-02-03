import flopy
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

import geopandas as gpd
import conda_scripts.plot_help as ph
import basic
import conda_scripts.arich_functions as af
from matplotlib import ticker

import pathlib


def run(name, m = None, draw_maps = True, add_overland = True, ovr_flux = 0.5):

    '''
    create pond recharge package
    :param name: name of run
    :param m: model instance
    :param draw_maps: make the maps
    :param add_overland: add overland flow
    :param ovr_flux: ft/day
    :return:
    '''

    print('running pond inflows recharge package')
    if m is None:
        m = basic.load_model(name)

    info, swr_info, sfr_info, riv_keys_info = basic.load_params(name)

    datestart = info['start_date']
    numdays = info['numdays']

    out_folder = basic.out_folder(name)

    pond_grid = gpd.read_file('ponds/ponds.geojson')

    # get counts of pond cells
    p = 1 / pond_grid.groupby('name').count().loc[:, ['row']].rename(columns={'row': 'pond_frac'})

    pond_grid = pd.merge(pond_grid, p, left_on='name', right_index=True)

    if draw_maps:
        draw_map_do(pond_grid, out_folder)
        draw_ponds_map(pond_grid, out_folder)

    df, wl = load_pond(datestart)

    df_cur = get_period(df, datestart, numdays)
    inflow_fraction = {'One': 0, 'Two': .5, 'Three': .5, "Four": 0}

    ax = df_cur.plot(ylabel='feet$^3$/s', figsize=(7, 7))
    df_cur_roll = df_cur.rolling(5, min_periods=0).mean(center=False)

    df_cur_roll.rename(columns={'Value': 'Value, Rolled'}).plot.area(ax=ax)
    ax.set_title('Pond Inflows, Split Between 2 and 3')

    plt.savefig(os.path.join(out_folder, 'pondQ.png'), dpi=250)

    cnt = 0
    # df_cur_roll.to_csv(f"RR_2022/pond_inflows/sum.csv")

    if add_overland:
        ovr = read_overland(m)
        ovr.loc[:,'flow_depth'] = ovr_flux/86400
        plot_ovr(ovr, datestart, out_folder, ovr_flux, numdays=numdays)
    else:
        ovr = pd.DataFrame()

    rech = {}

    for ind, d in df_cur_roll.iterrows():
        # pinf = pond_grid.query("inflow==True").set_index('rno')

        pond = pond_grid.copy(deep=True)

        fraction = pd.DataFrame.from_dict(inflow_fraction, orient='index', columns=['fraction'])

        pond = pd.merge(pond, fraction, left_on='name', right_index=True)
        # pinf = pinf.loc[:, ['fraction']]

        pond.loc[:,'flow'] = pond.loc[:,'pond_frac'] * pond.loc[:,'fraction'] * d['Value']

        pond.loc[:,'flow_depth'] = pond.loc[:,'flow']/(m.dis.delc[0] * m.dis.delr[0])


        if pond.loc[:,'flow_depth'].sum()>0:
            array = make_array(m, pond, col = 'flow_depth')
        else:
            array = 0.000

        array_ovr = 0.00
        if add_overland:
            if ind in ovr.index:
                ovr_cur = ovr.loc[ovr.index == ind,:]
                array_ovr = make_array(m, ovr_cur, col='flow_depth')


        rech[cnt] = array + array_ovr

        cnt = cnt + 1
        print(cnt, end =' ', flush = True)

    rch = make_rch(m, rech = rech)
    rch.write_file()

def make_rch(m, rech):
    # rech = {cnt: f"pond_inflows/day{cnt}.dat" for cnt in range(m.nper)}
    rch = flopy.modflow.ModflowRch(m, ipakcb = 1,  nrchop = 1, rech=rech)

    return rch


def read_overland(m):
    ovr = pd.read_csv('Overland_Flow/overland_flow_ts.csv', index_col=[0], parse_dates=True)
    ovr.loc[:, ['k', 'i', 'j']] = m.dis.get_lrc(list(ovr.loc[:, 'node_grid'].values))

    return ovr


def plot_ovr(ovr, datestart, folder, recharge_rate, numdays=365):
    q = ovr.groupby(ovr.index).count().loc[:, ['WSE']] * 200 * 200 * recharge_rate / 43560
    end_date = (pd.to_datetime(datestart) + pd.to_timedelta(numdays + 5, 'D')).strftime('%m/%d/%Y')
    q = q.resample('1D').sum()
    qm = q.max()
    q = q.loc[datestart:end_date, :].rename(columns={'WSE': "Overland Recharge"})
    plt.figure(figsize=(6, 6), dpi=300)
    ax = q.plot(drawstyle="steps-post", linewidth=2, ylabel='recharge (acre-feet)', c='b')
    ax.set_ylim([0, qm.values[0]]);
    ax.grid(True);
    ax.yaxis.get_label().set_color('b')
    # ax.text(1,1, f'Recharge rate = {recharge_rate}ft.', transform = ax.transAxes, ha = 'right', va = 'bottom')
    ax.legend().remove()
    ax.set_title(f'Daily Recharge from Overland Flow.\nRecharge rate = {recharge_rate}ft/d')

    ax2 = ax.twinx()

    p2 = q.cumsum().plot(ax=ax2, label='Cumulative', c='r', ylabel='cumulative recharge (acre-feet)')
    ax.get_yaxis().set_major_formatter(
        ticker.FuncFormatter(lambda x, p: format(int(x), ',')))

    ax2.get_yaxis().set_major_formatter(
        ticker.FuncFormatter(lambda x, p: format(int(x), ',')))

    ax2.yaxis.get_label().set_color('r')

    plt.savefig(os.path.join(folder, 'ovr_total.png'), bbox_inches='tight', dpi=250, figsize=(6, 6))

    return ax


def draw_map_do(pond_grid, out_folder):
    fig, ax = basic.basic_map(maptype=None)
    basic.set_bounds_to_shape(ax, pond_grid.buffer(1000))
    pond_grid.sort_values('name').plot('name', ax=ax, legend=False, edgecolor='k')
    ax.set_title('Mirabel-Wohler Infiltration Ponds Inflow Locations')

    ax = pond_grid.plot('top', ax=ax, legend=True, legend_kwds={'shrink': .2})

    pond_grid.set_geometry(pond_grid.geometry.centroid).loc[pond_grid.loc[:, 'inflow']].plot(ax=ax, color='k')

    pond_grid.plot(ax=ax, facecolor="None")

    ph.label_points(ax, pond_grid.set_geometry(pond_grid.geometry.centroid).loc[pond_grid.loc[:, 'inflow']],
                    'name', basin_name=None, fmt='s', text_color='k')

    af.add_basemaps(ax, maptype="ctx.Esri.NatGeoWorldMap")

    plt.savefig(os.path.join(out_folder, 'pondloc_inflow.png'), dpi=250)

def draw_ponds_map(pond_grid, out_folder):
    fig, ax = basic.basic_map()
    basic.set_bounds_to_shape(ax, pond_grid.buffer(5000))
    pond_grid.sort_values('name').plot('name', ax=ax, legend=False, edgecolor='k')
    ax.set_title('Mirabel-Wohler Infiltration Ponds')

    _gdf = pond_grid.dissolve('name').reset_index()
    _gdf = _gdf.set_geometry(_gdf.geometry.representative_point())

    ph.label_points(ax, _gdf,
                    'name', basin_name=None, fmt='s', text_color='y')
    plt.savefig(os.path.join(out_folder, 'pondloc.png'), dpi=250)


def make_array(m, df, col = 'flow_depth'):

    array = np.zeros((m.nrow, m.ncol))

    if 'row' in df.columns:
        array[df.loc[:,'row']-1, df.loc[:,'column']-1] = df.loc[:,col]
    else:
        array[df.loc[:, 'i'], df.loc[:, 'j']] = df.loc[:, col]

    return array

def assign_inflow(df_pond):
    '''

    :param df_pond:
    :return:
    '''
    dfall = pd.DataFrame()

    for _, pond in df_pond.groupby('name'):
        pond.loc[:, 'inflow'] = False

        pond.loc[pond.loc[:, 'top'].idxmin(), 'inflow'] = True

        pond.loc[:, 'mean_elev'] = pond.loc[:, 'top'].mean()

        dfall = dfall.append(pond)

    return dfall


def load_phist(year=2020):
    p = pathlib.Path(r"S:\Ops\RiverReport\Production_and_Demand_Report_PHIST01.xlsm")

    sheet = f"Data_{year}"

    tab = pd.read_excel(p, sheet_name=sheet, header=[3, 4, 5, 6, 7, 8], skiprows=[9, 10, 11, 12], index_col=[0])

    return tab


def isnumber(x):
    try:
        float(x)
        return x
    except:
        return np.nan


def load_pond(datestart):
    year = pd.to_datetime(datestart).year

    df = load_phist(year)

    df = df.loc[:, 'River Diversion'].iloc[:, :3].droplevel([0, 1, 3, 4], axis=1)

    df = df.applymap(isnumber).fillna(0)

    # df = df.astype({'Pump 1':np.float64})
    # print(df.dtypes)
    df.loc[:, 'Pump 1'] = df.loc[:, 'Pump 1'] * 18000 * 60 / (7.48 * 60 * 60 * 24)
    df.loc[:, 'Pump 2'] = df.loc[:, 'Pump 2'] * 9000 * 60 / (7.48 * 60 * 60 * 24)
    df.loc[:, 'Pump 3'] = df.loc[:, 'Pump 3'] * 18000 * 60 / (7.48 * 60 * 60 * 24)

    # get water levels
    wl = pd.DataFrame()
    for pond in [1, 2, 3, 4]:
        p = pathlib.Path(
            r"T:\arich\Russian_River\MirabelWohler_2022\Waterlevel_Data\MWs_Caissons - AvailableDailyAverages\DailyData\InfiltrationPonds")

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

    return df, wl


def get_period(df, start_date, numdays):
    df = df.sum(axis=1).to_frame("Value")

    df = df.loc[start_date:, :].iloc[:numdays]

    df.loc[:, 'Value'] = df.loc[:, 'Value'].interpolate()

    assert df.loc[:, 'Value'].isnull().sum() == 0, 'has nans'

    return df



if __name__ == "__main__":
    print('running')
    run('June2015')
    print("Executed when invoked directly")
else:
    print("Executed when imported")