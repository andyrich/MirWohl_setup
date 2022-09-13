import geopandas as gpd
import basic
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import pathlib
import warnings
import conda_scripts.plot_help as ph
import basic
import conda_scripts.arich_functions as af

def run(run_name):
    info, swr_info, sfr_info, riv_keys_info = basic.load_params(run_name)

    datestart = info['start_date']

    name = info['name']
    m = basic.load_model()
    out_folder = basic.out_folder(run_name)
    print(datestart)
    print(out_folder)
    basic.map_river(m)
    plt.savefig(os.path.join(out_folder, 'modelmap.png'),dpi = 250, bbox_inches = 'tight')

    print('done with map')

    pond_grid = gpd.read_file('ponds/ponds.geojson')

    ####
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
    ####


    df, wl = load_pond(datestart)

    df_cur = get_period(df, datestart, numdays)
    # inflow_fraction = {'One': 0, 'Two': .5, 'Three': .5, "Four": 0}
    inflow_fraction = {'One': 0, 'Two': 0., 'Three': 0., "Four": 0}

    fraction = pd.DataFrame.from_dict(inflow_fraction, orient='index', columns=['fraction'])

    ax = df_cur.plot(ylabel='feet$^3$/s', figsize=(7, 7))
    df_cur_roll = df_cur.rolling(5, min_periods=0).mean(center=False)

    df_cur_roll.rename(columns={'Value': 'Value, Rolled'}).plot.area(ax=ax)
    ax.set_title('Pond Inflows, Split Between 2 and 3')

    plt.savefig(os.path.join(out_folder, 'pondQ.png'), dpi=250)

    cnt = 0
    for ind, d in df_cur_roll.iterrows():
        pinf = pond_grid.query("inflow==True").set_index('rno')

        fraction = pd.DataFrame.from_dict(inflow_fraction, orient='index', columns=['fraction'])

        pinf = pd.merge(pinf, fraction, left_on='name', right_index=True)
        pinf = pinf.loc[:, ['fraction']]

        pinf = pinf * d['Value']

        # print(d['Value'])
        # print(ind)
        name = f"RR_2022/ref/pond/day{cnt}.dat"
        # print(name)
        with open(name, 'w') as out:
            pinf.to_csv(name, header=False)
            # out.write(f.format(d['Value'], ind.strftime("%y %b %d")))

        cnt = cnt + 1


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
