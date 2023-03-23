#!/usr/bin/env python
# coding: utf-8

# In[1]:


import flopy
import os
import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import conda_scripts.utils as utils
import conda_scripts.wiski as wiski
import pathlib
import basic
import matplotlib.dates as mdates
import conda_scripts.gwplot_fancy as gwp
import warnings
from flopy.utils import ZoneBudget
import conda_scripts.plot_help as ph
import re

def run(run_name, reload=False, ml=None, plot_well_locs = True, plot_hydros = True, skip_fancy = False, add_temp = True ):
    '''

    :param ml:
    :param run_name: to load from run_names.txt
    :param reload: if true will load from spreadsheets, wiski, etc and create file. False will just re-load old file.
    :return: None
    '''
    info, swr_info, sfr_info, riv_keys_info = basic.load_params(run_name)

    datestart = info['start_date']
    numdays = info['numdays']
    name = info['name']

    out_folder = basic.out_folder(run_name)

    print(datestart)
    print(out_folder)

    if ml is None:
        ml = basic.load_model()

    if reload:
        print('loading data from wiski, spreadsheets, etc')
        wells_mod = load_wells_mod(ml)
    else:
        print('loading wells_mod from file')
        wells_mod = gpd.read_file('GIS/wells_mod.geojson')

    if plot_well_locs:
        plot_model_wells(wells_mod = wells_mod, out_folder = out_folder)

    if plot_hydros:
        obsall = do_hydros(ml, wells_mod, out_folder, datestart, numdays, skip_plotting = skip_fancy, add_temp = add_temp)
        plot_one_to_one(obsall, out_folder)
        plot_residual(obsall, out_folder=out_folder, ml = ml)


def plot_residual(allobs, out_folder, ml):
    '''
    plot residuals from all observations
    :param allobs:
    :param out_folder:
    :param ml:
    :return:
    '''

    locs = allobs.drop_duplicates('well').loc[:, ['geometry', 'well']]

    allobs.loc[:, 'residual'] = allobs.loc[:, 'Observed'] - allobs.loc[:, 'Simulated']

    allobs = allobs.drop(columns='geometry').groupby('well').mean().reset_index()

    merged = pd.merge(allobs, locs, on='well')
    merged = gpd.GeoDataFrame(merged, geometry='geometry', crs=2226)

    fig, ax = basic.map_river(m=ml, add_basemap=True)
    ph.remove_ctx_annotations(ax)
    basic.set_bounds(ax)

    merged.plot('residual', markersize=15, edgecolor='k', ax=ax, legend=True,
                cmap='jet', legend_kwds={'title': 'Residual (Obs. - Sim.)', 'loc': 'upper left',
                                         'bbox_to_anchor': (1, 1)},
                classification_kwds={'bins': np.arange(-25, 26, 5)},
                scheme='UserDefined', zorder=105)

    ph.label_points(ax=ax, gdf=merged, fmt=".1f", colname='residual', basin_name=None)

    ax.text(1, 0, 'Labels are residuals', ha='right', transform=ax.transAxes)

    plt.savefig(os.path.join(out_folder, 'residuals.png'), dpi=250, bbox_inches='tight')

    return fig, ax

def do_hydros(ml, wells_mod, out_folder, datestart, numdays, skip_plotting = False, add_temp = False):
    '''
    plot all hydrographs
    :param skip_plotting:
    :param ml:
    :param wells_mod:
    :param out_folder:
    :param datestart:
    :param numdays:
    :return: obsall (predicted versus observed)
    '''
    _, hdsobj = basic.get_heads(ml, return_final= False)
    partics = os.path.join(out_folder, 'hydrographs')
    obsall = pd.DataFrame()

    nwells = 0
    passed_wells = []
    plotted_wells = []

    print(f'saving hydrographs to:\n\t{out_folder}')

    for _, wel in wells_mod.sort_values('station_no').iterrows():
        station_name = wel['station_name']
        print(f"plotting {station_name}")
        idx = (wel.loc['Model Layer'] - 1, wel.loc['i_r'], wel.loc['j_c'])
        head = get_ts(idx, hdsobj, datestart)
        obs = load_obs(wel.loc['Filename'], datestart, numdays=numdays)

        ymin, ymax = basic.isnumber(wel['ymin']), basic.isnumber(wel['ymax'])
        ymin = ymin if (not np.isnan(ymin)) else None
        ymax = ymax if (not np.isnan(ymax)) else None
        # obs = load_obs(wel.loc['Well Name'], datestart,numdays=numdays)

        if obs.shape[0] == 0:
            skip_gw_data = False
        else:
            skip_gw_data = True
            predvobs = head.join(obs.rename(columns={'Value': 'Observed'}), how = 'left')

            predvobs.loc[:, 'zone'] = wel['zone']
            predvobs.loc[:, 'geometry'] = wel['geometry']
            predvobs.loc[:, 'well'] = station_name
            predvobs.loc[:,'station_no'] = wel['station_no']
            predvobs.loc[:,'label'] = f"{wel['station_no']}, {wel['station_name']}"
            predvobs.loc[:,'Model Layer'] = idx[0]
            predvobs.loc[:,  'i_r'] = idx[1]
            predvobs.loc[:,  'j_c'] = idx[2]

            if predvobs.shape[0] == obs.shape[0]:
                warnings.warn(f"shapes not matching in setup for 1 to 1 in Hydrographs. missing values are" \
                              f"\n{obs.index[~obs.index.isin(head.index)]}" \
                              f"\nshape of simulated {head.shape}\n" \
                              f"shape of observed {obs.shape}\n" \
                              f"shape of predvobs {predvobs.shape}\n" \
                              f"index of simulated\n{head.index}\n" \
                              f"index of observed\n{obs.index}\n" \
                              f"index of predvobs\n{predvobs.index}\n")

            obsall = obsall.append(predvobs)


        if skip_plotting:
            print('not plotting hydrographs')
        else:
            f = wel['station_no']
            stno = wel['station_name']

            filename = os.path.join(partics, f'{f}.png')
            # if not os.path.exists(filename):
            nwp = gwp.fancy_plot(station_name, group=None,
                                 filename=None,
                                 allinfo=None,
                                 do_regress=False)

            nwp.do_plot(False, skip_gw_data=skip_gw_data,
                        map_buffer_size=2500, seasonal=False,
                        plot_dry=False, plot_wet=False,
                        maptype='ctx.USGS.USTopo')

            head.plot(ax=nwp.upleft)
            minya, maxya = head.loc[:, 'Simulated'].min(), head.loc[:, 'Simulated'].max()
            nwp.upleft.set_ylim([minya - 10, maxya + 10])
            nwp.upleft.set_ylim([ymin, ymax])
            if np.any([ymin, ymax] == None ):
                ymin, ymax = nwp.upleft.get_ylim()

            nwp.upleft.set_yticks(np.arange(ymin, ymax+1, 10))
            nwp.upleft.set_yticks(np.arange(ymin, ymax + 1, 2), minor=True)
            nwp.upleft.set_title(f"{f}, {stno}")
            nwp.upleft.set_xlabel('')
            nwp.ax_map.set_title('')

            if obs.shape[0] > 1:
                obs.reindex(pd.date_range(datestart, freq = 'D', periods = numdays))\
                    .rename(columns={'Value': 'Observed'}).plot(ax=nwp.upleft)
                # miny, maxy = obs.loc[:, 'Value'].min(), obs.loc[:, 'Value'].max()
                # nwp.upleft.set_ylim([np.nanmin([miny, minya]) - 10, np.nanmax([maxy, maxya]) + 10])


            # if not skip_gw_data:
            # re-set xlimits because limits fancy plot are set to 1980
            nwp.upleft.set_xlabel('')
            nwp.upleft.set_xlim(left=pd.to_datetime(datestart) - pd.to_timedelta('1 w'),
                                right=pd.to_datetime(datestart) + pd.to_timedelta(numdays + 7, unit = 'D'))
            basic.set_dates_xtick(nwp.upleft)
            # locator = mdates.AutoDateLocator(minticks=6, maxticks=18)
            # formatter = mdates.ConciseDateFormatter(locator)
            # nwp.upleft.xaxis.set_major_locator(locator)
            # nwp.upleft.xaxis.set_major_formatter(formatter)
            # nwp.upleft.set_xlim(left=head.index.min() - pd.to_timedelta('1 w'),
            #                     right=head.index.max() + pd.to_timedelta('1 w'))
            # freq = int(2*np.ceil(numdays/365))
            # nwp.upleft.xaxis.set_major_locator(mdates.MonthLocator(interval = freq))
            # nwp.upleft.xaxis.set_minor_locator(mdates.AutoDateLocator())
            # nwp.upleft.xaxis.set_major_formatter(
            #     mdates.ConciseDateFormatter(mdates.MonthLocator()))

            if add_temp:
                temp = load_temp(wel.loc['Filename'], datestart=datestart, numdays=numdays)
                if temp.shape[0]>0:
                    ax = plot_temp(nwp.upleft, temp)

            nwp.upleft.grid(True, which='minor', linewidth=.2, axis = 'y', c='grey')
            nwp.upleft.grid(True, which='major', linewidth=.5, axis='y', c='black')
            nwp.upleft.grid(False, which='major', linewidth=.5, axis='x', c='black')



            # nwp.upleft.grid(True, which='major', linewidth= .5, c = 'black')
            plt.savefig(filename, dpi=250, bbox_inches='tight')

            plt.close()
            plt.close(plt.gcf())
            plt.close('all')
            plt.clf()
            print(os.path.abspath(filename))
            assert os.path.exists(filename), f'file does not exist\n{filename}'

            del nwp

            nwells = nwells + 1
            plotted_wells.extend([station_name])

    obsall.to_csv(os.path.join(partics, 'all_meas.csv'))

    print(f"\n\nNumber of wells plotted:\n{nwells}\n")
    print(f"wells plotted:\n{plotted_wells}\n\n")

    return obsall

def plot_temp(axin, temp):
    '''
    add temp to the hydrograph plot on new y axis
    :param axin:
    :param temp:
    :return: new y axis
    '''


    xticks = axin.get_xticks()
    xlim = axin.get_xlim()
    lab = axin.get_xticklabels()

    ax = axin.twinx()
    ax.scatter(temp.index, temp.Value, c='r', marker='.', ls='None', label='Temp')
    ax.set_ylim([30, 75])
    ax.legend(loc='lower right', bbox_to_anchor=(1, 1))
    ax.tick_params(axis='y', colors='red')  # setting up X-axis tick color to red
    ax.spines['right'].set_color('red')  # setting up Y-axis tick color to red

    # ax.set(xticklabels=[])  # remove the tick labels
    # ax.tick_params(bottom=False)  # remove the ticks
    # ax.set_xlim(xlim[0], xlim[1])
    #
    # axin.set_xlim(xlim[0], xlim[1])
    # axin.set_xticks(xticks)
    # axin.set_xticklabels(lab)

    return ax

def scatter(obs2plot, ax):
    import matplotlib.colors as colors
    import matplotlib.cm as cmx
    # Get unique names of species
    uniq = list(set(obs2plot['label']))

    # Set the color map to match the number of species
    z = range(1, len(uniq))
    hot = plt.get_cmap('tab20')
    cNorm = colors.Normalize(vmin=0, vmax=len(uniq))
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=hot)

    x = obs2plot.loc[:, 'Observed']
    y = obs2plot.loc[:, 'Simulated']

    # Plot each type
    for i in range(len(uniq)):
        indx = obs2plot['label'] == uniq[i]
        ax.scatter(x[indx], y[indx], s=15, color=scalarMap.to_rgba(i), label=uniq[i], marker='o', alpha=.5)

    ax.legend(loc = 'upper left', bbox_to_anchor = (0, -.1), ncol = 2)

def plot_one_to_one(obsall, out_folder):


    l = [obsall.loc[:, ['Simulated', 'Observed']].describe().loc['min'].min(),
         obsall.loc[:, ['Simulated', 'Observed']].describe().loc['max'].max()]
    z = [obsall.loc[:, ['Simulated', 'Observed']].describe().loc['min'].min(),
         obsall.loc[:, ['Simulated', 'Observed']].describe().loc['max'].max()]

    zones = obsall.zone.unique()
    n = obsall.zone.nunique()

    fig, axes = plt.subplots(1, n, sharey = True, sharex = True, figsize=(6*n, 6))
    axes = axes.flatten()

    for i in range(n):
        ax = axes[i]
        ax.plot(l, z, ls='-', color='k')
        zone = zones[i]
        scatter(obsall.query(f"zone=='{zone}'"), ax)
        ax.grid(True)
        ax.set_title(zone)
        ax.set_xlabel('Observed (ft)')
        ax.set_ylabel('Simulated (ft)')

    plt.savefig(os.path.join(out_folder, '1to1.png'), dpi=250, bbox_inches='tight')


def get_ts(idx, hdsobj, datestart, ):
    ts = hdsobj.get_ts(idx)

    df = pd.DataFrame(ts[:, 1], columns=['Simulated'])

    df = df.set_index(pd.to_datetime(datestart) + pd.to_timedelta(ts[:, 0], unit='s'))

    return df


def load_obs(name, datestart=None, numdays=109):
    '''

    :param name:
    :param datestart:
    :param numdays:
    :return:
    '''

    fold = r"Waterlevel_Data\MWs_Caissons - AvailableDailyAverages\DailyData\MonitoringWells"

    # need to check if it's a caisson record. if it is, it needs to be loaded differently
    if isinstance(name, str):
        if 'caisson' in name.lower():
            fold = r"Waterlevel_Data\MWs_Caissons - AvailableDailyAverages\DailyData\Caissons"
            caisson = True
        else:
            caisson = False

    else:
        name = 'no filename given'

    path = pathlib.Path(fold).joinpath(name)
    # path = pathlib.Path(fold).joinpath(name.replace(' ', '').replace('-', '_') + '.csv')

    # if end_time is None:
    end_time = pd.to_datetime(datestart) + pd.to_timedelta(numdays, unit='D')

    if path.exists():
        stg = read_stg(path, caisson = caisson)

        if datestart is not None:
            stg = stg.loc[datestart:end_time, :]


    else:
        print(f"path does not exist:\n\n{path}\n")

        stg = pd.DataFrame()

    return stg


def load_temp(name, datestart=None, numdays=365):
    '''
    load the temp a named well. only works for observation wells
    :param name:
    :param datestart:
    :param numdays:
    :return:
    '''

    fold = r"Waterlevel_Data\MWs_Caissons - AvailableDailyAverages\DailyData\MonitoringWells"

    # need to check if it's a caisson record. if it is, it needs to be loaded differently
    if isinstance(name, str):
        if 'caisson' in name.lower():
            fold = r"Waterlevel_Data\MWs_Caissons - AvailableDailyAverages\DailyData\Caissons"
            caisson = True
        else:
            caisson = False
    else:
        caisson = False
        name = 'no filename given'

    if caisson:
        # path = pathlib.Path(fold).joinpath(name)
        path = pathlib.Path(fold).joinpath(name)
    else:
        path = pathlib.Path(fold).joinpath(name.replace('.csv', 'temp.csv'))

    # if end_time is None:
    end_time = pd.to_datetime(datestart) + pd.to_timedelta(numdays, unit='D')

    if path.exists() and (not caisson):
        stg = pd.read_csv(path, parse_dates=[0])
        stg = stg.set_index(stg.columns[0])
        stg = stg.resample('1D').mean()
        stg = stg.loc[(stg.loc[:, 'Value'] > 50) & (stg.loc[:, 'Value'] < 120)]

        if datestart is not None:
            stg = stg.loc[datestart:end_time, :]
    else:
        print(f"path does not exist:\n\n{path}\n")
        stg = pd.DataFrame()

    return stg

def read_stg(path, caisson = False):
    '''
    load the wl record.
    :param path:
    :param caisson: if it's a caisson or not. these records are formatted differently.
    :return: df with daily values of 'Value'
    '''


    print(f"----------\npath does exist:\n{path.name}\n")
    if caisson:
        stg = pd.read_csv(path)
        stg = stg.rename(columns={'DateTime': 'Datetime'})
        stg.loc[:, 'Datetime'] = pd.to_datetime(stg.loc[:, 'Datetime'])
        stg = stg.set_index('Datetime')
        stg = stg.loc[:, ['Value']]
        stg.loc[:, 'Value'] = stg.loc[:, 'Value'].apply(basic.isnumber)
        stg = stg.resample('1D').mean()

        # get the name of the caisson in order to get the elevation offset
        pattern = '[a-z]+'
        repl = ''
        caisson_name = re.sub(pattern, repl, path.name.replace('.csv', ''), flags=re.IGNORECASE)

        stg = stg + caisson_offsets(caisson_name)
        stg = stg.loc[(stg.loc[:, 'Value'] > -50) & (stg.loc[:, 'Value'] < 100)]

    else:
        stg = pd.read_csv(path, parse_dates=[0])
        stg = stg.set_index(stg.columns[0])
        stg = stg.resample('1D').mean()
        stg = stg.loc[(stg.loc[:, 'Value'] > -50) & (stg.loc[:, 'Value'] < 100)]

    return stg

def caisson_offsets(caisson):
    '''
    load elevation in ft ngvd29 of caisson bottoms
    file: // / T:\tschram\Transmission_System\Infrastructure\CollectorCapacityStudy\CollectorData_2019.xlsx
    :param caisson: int or str
    :return: offset in feet ngvd29
    '''


    offsets = {'1': -19.2,
               '2': -18.6,
               '3': -29.55,
               '4': -33.05,
               '5': -17.79,
               '6': -23.7}

    if isinstance(caisson, str):
        pass
    else:
        caisson = str(caisson)

    offset = offsets[caisson]

    return offset


def plot_model_wells(wells_mod, out_folder):
    '''
    plot folium map of wells

    :param wells_mod:
    :param out_folder:
    :return:
    '''
    ibound = gpd.read_file("GIS/iboundlay1.shp")
    ibound = ibound.query("ibound_1==1")

    swr = gpd.read_file("GIS/SWR_Reaches.shp")

    caissons = gpd.read_file('GIS/wells.shp')

    sfr = gpd.read_file('SFR_files/only_sfr_cells.shp')
    wells_mod.to_html(os.path.join(out_folder, 'observation_wells.html'))


    m = ibound.explore(style_kwds={'weight': .5, 'fill': False}, name='Model Boundary')
    swr.explore(m=m, style_kwds={'weight': 3, 'fill': True, 'color': 'yellow'}, name="SWR Cells")
    sfr.explore(m=m, style_kwds={'weight': 3, 'fill': True, 'color': 'grey'}, name='SFR Cells')
    caissons.explore(m=m, style_kwds={'weight': 3, 'fill': False, 'color': 'cyan'}, name='Caisson Cells')

    wells_mod.filter(regex='geom|station|Name|WISKI|Name|depth|Hydrograph').explore(m=m,
                                                marker_kwds={'radius': 5, 'color': 'red'},
                                                popup = 'Hydrograph',
                                                name='Monitoring Wells')

    # wiski_meta.explore( marker_kwds = {'radius':5, 'color':'cyan'}, m = m, name = 'Wiski Wells')
    utils.folium_maps.add_layers(m)
    m.save(os.path.join(out_folder, 'wells.html'))


def load_wells_mod(ml):
    '''
    load modeled wells as gdf
    :param ml:
    :return:
    '''

    wiski_meta = get_wiski()
    wells = get_well_locations()

    wells_mod = pd.merge(wiski_meta, wells.loc[:, ['Well Name', 'Filename',
                                                   'Model Layer',
                                                   'WISKI', 'Notes_SX', 'Notes_SRM', 'ymin','ymax',
                                                   'USGS Map ID (https://pubs.usgs.gov/ds/610/pdf/ds610.pdf)',
                                                   'USGS NWIS ID',
                                                   'WCR (Y/N)', 'Total completed depth (ft bgs)',
                                                   'Total depth (ft bgs)',
                                                   'Screened interval (ft bgs)', 'casing diameter (inches)', ]],
                         left_on='station_name', right_on='WISKI', how='inner')

    # add column with link to open hydrograph
    def ref(x):
        v = f"""
            <p>Hydrograph  <a target="_blank" href="hydrographs/{x}.png">link</a></p>
            """

        return v

    wells_mod.insert(2, 'Hydrograph', wells_mod.loc[:,'station_no'].apply(ref))

    wells_mod = gpd.GeoDataFrame(wells_mod,
                                 geometry=gpd.points_from_xy(wells_mod.station_longitude, wells_mod.station_latitude),
                                 crs=4326).to_crs(2226)

    grid = flopy.utils.gridintersect.GridIntersect(ml.modelgrid)

    inx = [grid.intersect(x, shapetype='point').cellids for x in wells_mod.geometry.tolist()]
    r = [x[0][0] if len(x) > 0 else np.nan for x in inx]
    c = [x[0][1] if len(x) > 0 else np.nan for x in inx]
    inside = [True if len(x) > 0 else False for x in inx]

    wells_mod.loc[:, 'i_r'] = r
    wells_mod.loc[:, 'j_c'] = c
    wells_mod.loc[:, 'inmodearea'] = inside

    zones = ZoneBudget.read_zone_file(os.path.join(ml.model_ws, 'zonebud', 'zonedbud.zbarr'))
    wells_mod.loc[:, 'zone'] = zones[0, r, c]
    aliases = {1: 'Mirabel', 2: 'Wohler', 3: 'Upstream'}
    wells_mod.loc[:, 'zone'] = wells_mod.loc[:, 'zone'].replace(aliases)

    return wells_mod

def get_well_locations():
    p = r"T:\smaples\USGS-LBNL_WQ\Wohler_MW_inventory\Monitoring Wells Site Visit Notes - Wohler+Mirabel_car.xlsx"
    t = pd.read_excel(p)
    t = t.loc[t.loc[:, 'Longitude (iphone gps)'].notnull(), :]
    df = gpd.GeoDataFrame(t, geometry=gpd.points_from_xy(t.loc[:, 'Longitude (iphone gps)'],
                                                         t.loc[:, 'Latitude (iphone gps)'], crs=4326))

    df = df.to_crs(2226)

    return df


def get_wiski():
    # gw = wiski.wiski.get_gw_stations_in_basin(basins= ['LRR*'], final_only = False)

    meta = wiski.wiski.get_gw_stations_wellmeta_in_basin(basins=['LRR*'])
    meta = meta.dropna(subset='station_name')
    meta = gpd.GeoDataFrame(meta, geometry=gpd.points_from_xy(meta.station_longitude, meta.station_latitude),
                            crs=4326).to_crs(2226)

    return meta
