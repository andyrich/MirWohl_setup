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


def run(run_name, reload = False):
    '''

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

    ml = basic.load_model()


    ibound = gpd.read_file("GIS/iboundlay1.shp")
    ibound = ibound.query("ibound_1==1")

    swr = gpd.read_file("GIS/SWR_Reaches.shp")

    sfr = gpd.read_file('SFR_files/only_sfr_cells.shp')

    if reload:
        print('loading data from wiski, spreadsheets, etc')
        wiski_meta = get_wiski()
        wells = get_well_locations()

        wells_mod = pd.merge(wiski_meta, wells.loc[:,['Well Name', 'WISKI','Notes_SX', 'Notes_SRM',
                          'USGS Map ID (https://pubs.usgs.gov/ds/610/pdf/ds610.pdf)','USGS NWIS ID',
                         'WCR (Y/N)','Total completed depth (ft bgs)', 'Total depth (ft bgs)',
                        'Screened interval (ft bgs)', 'casing diameter (inches)',]] ,
                             left_on = 'station_name', right_on = 'WISKI', how = 'inner')

        wells_mod = gpd.GeoDataFrame(wells_mod, geometry = gpd.points_from_xy(wells_mod.station_longitude, wells_mod.station_latitude), crs  = 4326).to_crs(2226)
        wells_mod.to_html(os.path.join(out_folder,'observation_wells.html'))

        grid = flopy.utils.gridintersect.GridIntersect(ml.modelgrid)

        inx = [grid.intersect(x, shapetype = 'point').cellids for x in wells_mod.geometry.tolist() ]
        r = [x[0][0] if len(x)>0 else np.nan for x in inx]
        c = [x[0][1] if len(x)>0 else np.nan for x in inx]
        inside = [True if len(x)>0 else False for x in inx]

        wells_mod.loc[:,'i_r'] = r
        wells_mod.loc[:,'j_c'] = c
        wells_mod.loc[:,'inmodearea'] = inside
    else:
        print('loading wells_mod from file')
        wells_mod = gpd.read_file('GIS/wells_mod.geojson')


    m = wells_mod.filter(regex = 'geom|Name|WISKI|Name|depth').explore( marker_kwds = {'radius':5, 'color':'black'},
                                                             name = 'Monitoring Wells')

    ibound.explore(m = m, style_kwds = {'weight':1,'fill':False}, name = 'Model Boundary')
    swr.explore(m = m, style_kwds = {'weight':3,'fill':True, 'color':'green'}, name = "SWR Cells")
    sfr.explore(m = m, style_kwds = {'weight':3,'fill':True, 'color':'grey'}, name = 'SFR Cells')


    # wiski_meta.explore( marker_kwds = {'radius':5, 'color':'cyan'}, m = m, name = 'Wiski Wells')
    utils.folium_maps.add_layers(m)
    m.save(os.path.join(out_folder, 'wells.html'))

    hds, hdsobj = basic.get_heads(ml)

    partics = os.path.join(out_folder,'hydrographs')
    for _,  wel in wells_mod.iterrows():
        station_name = wel['station_name']
        print(f"plotting {station_name}")
        idx = (0, wel.loc['i_r'], wel.loc['j_c'])
        head = get_ts(idx, hdsobj, datestart)
        obs = load_obs(wel.loc['Well Name'], datestart,numdays=numdays)

        if obs.shape[0]==0:
            skip_gw_data = False
        else:
            skip_gw_data = True

        f = wel['station_no']

        filename=os.path.join(partics,f'{f}.png')
        # if not os.path.exists(filename):
        nwp = gwp.fancy_plot(station_name,group = None,
                             filename=os.path.join(partics,f'{f}.png'),
                             allinfo=None,do_regress=False)

        nwp.do_plot(False, skip_gw_data=skip_gw_data, map_buffer_size = 2500, seasonal = False,
                    plot_dry = False, plot_wet = False,
                  maptype = 'ctx.USGS.USTopo')



        head.plot(ax = nwp.upleft)
        minya, maxya = head.loc[:,'Simulated'].min(), head.loc[:,'Simulated'].max()
        nwp.upleft.set_ylim([minya-10, maxya+10])
        if obs.shape[0]>1:
            obs.rename(columns = {'Value':'Observed'}).plot(ax = nwp.upleft)
            miny, maxy = obs.loc[:,'Value'].min(), obs.loc[:,'Value'].max()
            nwp.upleft.set_ylim([np.nanmin([miny,minya])-10, np.nanmax([maxy,maxya])+10])

        if not skip_gw_data:

            nwp.upleft.set_xlim(left = head.index.min()-pd.to_timedelta('1 w'),
                                right = head.index.max()+pd.to_timedelta('1 w'))

        nwp.upleft.xaxis.set_major_locator(mdates.AutoDateLocator())
        nwp.upleft.xaxis.set_minor_locator(mdates.DayLocator())
        nwp.upleft.xaxis.set_major_formatter(
                    mdates.ConciseDateFormatter(nwp.upleft.xaxis.get_major_locator()))


        plt.savefig(filename,dpi = 250, bbox_inches ='tight')


def get_ts(idx,hdsobj, datestart, ):
    ts = hdsobj.get_ts(idx)

    df = pd.DataFrame(ts[:, 1], columns=['Simulated'])

    df = df.set_index(pd.to_datetime(datestart) + pd.to_timedelta(ts[:, 0], unit='s'))

    return df

def load_obs(name, datestart=None, numdays=109):

    fold = r"T:\arich\Russian_River\MirabelWohler_2022\Waterlevel_Data\MWs_Caissons - AvailableDailyAverages\DailyData\MonitoringWells"

    path = pathlib.Path(fold).joinpath(name.replace(' ', '').replace('-', '_') + '.csv')

    # if end_time is None:
    end_time = pd.to_datetime(datestart) + pd.to_timedelta(numdays, unit='D')

    if path.exists():
        print(f"----------\path does exist:\n{path.name}")
        stg = pd.read_csv(path, parse_dates=[0])
        stg = stg.set_index(stg.columns[0])
        stg = stg.resample('1D').mean()

        if datestart is not None:
            stg = stg.loc[datestart:end_time, :]


    else:
        print(f"path does not exist:\n{path}")
        stg = pd.DataFrame()

    return stg


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