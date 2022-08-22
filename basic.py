import flopy
import os
import geopandas as gpd
import matplotlib.pyplot as plt
import geopandas as gpd
import basic
import contextily as ctx
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import conda_scripts.make_map as mp
import flopy.utils.binaryfile as bf

import shutil

def load_params(run_name = 'calibration'):
    '''
    info, swr_info,sfr_info, riv_keys = basic.load_params()
    riv_keys
    
    '''

    import json

    with open("run_names.txt") as json_data_file:
        data = json.load(json_data_file)
    
    info = data[run_name]
    swr = data[run_name]['parameters']['SWR']
    sfr = data[run_name]['parameters']['SFR']
    riv_keys = data['riv_keys']
    
    with open("model_info.txt") as json_data_file:
        data = json.load(json_data_file)

    riv_keys = data['riv_keys']
    
    return info, swr, sfr, riv_keys

def setup_folder(run_name):
    print(f'setting up folders with on versions\ {run_name}')

    if not os.path.exists(os.path.join('versions', run_name)):
        
        os.mkdir(os.path.join('versions', run_name))
        
    if not os.path.exists(os.path.join('versions', run_name, 'hydrographs')):
        
        os.mkdir(os.path.join('versions', run_name, 'hydrographs'))

    if not os.path.exists(os.path.join('versions', run_name, 'wl_maps')):
        os.mkdir(os.path.join('versions', run_name, 'wl_maps'))
    
    import shutil

    src = 'website_info/subindex.html'
    dst = os.path.join('versions', run_name, 'subindex.html')
    if not os.path.exists(dst):
        shutil.copyfile(src, dst)

    src = 'website_info/gallery.html'
    dst = os.path.join('versions', run_name, 'gallery.html')
    if not os.path.exists(dst):
        shutil.copyfile(src, dst)
    
    

def out_folder(run_name = 'calibration'):
    info, swr, sfr, riv_keys = load_params(run_name)
    
    return os.path.join('versions', info['name'])

def load_model(verbose = False, path = None, nam = 'RRMF.nam'):
    
    if path is None:
        path = 'RR_2022'

    print(path)
    ml = flopy.modflow.Modflow.load(nam, 
                                    model_ws = path,
                                    verbose = verbose,
                                    version = 'mfnwt', 
                                    exe_name = "RR_2022/MODFLOW-NWT_64")
    
    
    return ml

def map_river(m = None, add_basemap = False, fig = None, ax = None, maptype = 'ctx.USGS.USTopo'):
    import matplotlib.patches as mpatches
    print('creating map_river')
    mod = gpd.read_file('GIS/model_boundary.shp')
    routing = gpd.read_file('GIS/nhd_hr_demo_sfr_routing.shp')
    cells = gpd.read_file('GIS/nhd_hr_demo_sfr_cells.shp')
    outlets = gpd.read_file('GIS/nhd_hr_demo_sfr_outlets.shp')
    model_boundary_5070 = mod.to_crs(epsg=2226)
    # print('creating axes')
    if fig is None and ax is None:
        print('creating axes')
        fig = plt.figure(figsize = (6,10), dpi = 250)
        mm = mp.make_map('Mirabel-Wohler Model Area')
        ax = mm.plotloc(fig, shape = model_boundary_5070,maptype=maptype )


    
    cells.plot('name',ax = ax, zorder = 2, facecolor = 'None')
    routing.plot(ax=ax, zorder=3)
    outlets.plot(ax=ax, c='red', zorder=4, label='outlets')
    model_boundary_5070.plot(ax=ax, facecolor='None', 
                             edgecolor='gray',
                             zorder=1
                            )


    LegendElement = [
        mpatches.mlines.Line2D([], [], color='red', linewidth=0., marker='o', label='sfr outlet'),
        mpatches.mlines.Line2D([], [], color='#1f77b4', label='sfr routing'),
        mpatches.Patch(facecolor='None', edgecolor='gray', label='Model Boundary\n(active area)')
    ]

    ax.legend(handles=LegendElement, loc='upper left')
    
    if m is None:
        m = load_model()
        
    f = flopy.plot.PlotMapView(m, ax =ax)
    # f.plot_array(m.bas6.ibound[0])
    f.plot_ibound(color_noflow = 'black', alpha = .1)
    if add_basemap:
        ctx.add_basemap(ax, crs = 2226)

    return fig, ax

def basic_map(m = None, add_basemap = False, fig = None, ax = None, maptype = 'ctx.USGS.USTopo'):
    import matplotlib.patches as mpatches
    mod = gpd.read_file('GIS/model_boundary.shp')
    model_boundary_5070 = mod.to_crs(epsg=2226)
    
    if fig is None and ax is None:
        fig = plt.figure(figsize = (6,10), dpi = 250)
        mm = mp.make_map('Mirabel-Wohler Model Area')
        ax = mm.plotloc(fig, shape = model_boundary_5070, maptype = maptype )

    if m is None:
        m = load_model()
    f = flopy.plot.PlotMapView(m, ax =  ax)

    f.plot_ibound(color_noflow = 'black', alpha = .1)
    
    if add_basemap:
        ctx.add_basemap(ax, crs = 2226)
    return fig, ax

def swr_map(m):
    fig, ax = map_river(m)
    sfr_filt = gpd.read_file('GIS/SWR_reaches.shp')
    sfr_filt.plot(ax = ax, facecolor = 'green', edgecolor = 'y', lw = .1, zorder = 100)

    sfr_filt.loc[[sfr_filt.rno.idxmax()],:].geometry.centroid.plot(ax = ax, marker = '*', edgecolor = 'c', facecolor = 'c', lw = 1, zorder =101)
    sfr_filt.loc[[sfr_filt.rno.idxmin()],:].geometry.centroid.plot(ax = ax, marker = '*', edgecolor = 'orange', facecolor = 'orange', lw = 1, zorder =101)

    ax.text(sfr_filt.loc[[sfr_filt.rno.idxmax()],:].geometry.centroid.x, sfr_filt.loc[[sfr_filt.rno.idxmax()],:].geometry.centroid.y, 'Dam (Last SWR Reach)',zorder = 102)
    ax.text(sfr_filt.loc[[sfr_filt.rno.idxmin()],:].geometry.centroid.x, sfr_filt.loc[[sfr_filt.rno.idxmin()],:].geometry.centroid.y, 'Inflow to SWR',zorder = 102)

    return fig, ax

def set_bounds_to_shape(ax, gdf):
    tb = gdf.to_crs(epsg=4326).total_bounds
    box = [tb[0], tb[2], tb[1], tb[3]]
    
    ax.set_extent(box)

def set_bounds(ax, locname = 'MIRABEL'):
    
    locs =   { "MIRABEL": [-122.909128, 38.4911643, -122.85212, 38.52827]}
    tb = locs[locname.upper()]
    box = [tb[0], tb[2], tb[1], tb[3]]

    ax.set_extent(box)
    
    
def get_swr_reaches(m):
    with open("RR_2022/RRlist.lst", "r") as lst:
        line = lst.readline()

        while line:

            if 'NUMBER OF REACHES' in line:
                print(line)
                numbreaches = int(line.split('=')[1])
                break

            line = lst.readline()

        while line:
            # print('a')
            if 'TIME-INVARIANT SURFACE-WATER ROUTING REACH DATA' in line:
                lst.readline()
                lst.readline()
                x = list()

                for i in range(numbreaches):
                    vi = lst.readline()

                    x.append(vi)
                break
            line = lst.readline()
        df = pd.DataFrame([xi.split() for xi in x], columns = ['REACH',
                                                             'IROUTETYPE',
                                                             'SOLUTION METHOD',
                                                             'IRGNUM',
                                                             'IRG',
                                                             'LAY',
                                                             'ROW',
                                                             'COL',
                                                             'DLEN'])
        

        df = df.astype({'REACH':int,
                         'IROUTETYPE':int,
                         'SOLUTION METHOD': str,
                         'IRGNUM':int,
                         'IRG':int,
                         'LAY':int,
                         'ROW':int,
                         'COL':int,
                         'DLEN':float})
        df.loc[:,'i'] = df.loc[:,'ROW']-1
        df.loc[:,'j'] = df.loc[:,'COL']-1
        
        df.loc[:,'geometry'] = get_mg(df.loc[:,'i'], df.loc[:,'j'], m.modelgrid)
        
        df = gpd.GeoDataFrame(df, geometry = 'geometry', crs = 2226)
        
        return  df
    
def get_mg(xx,yy,mg):
    # xx = xx.reshape((-1,1))
    from shapely.geometry import Polygon
    # xx = xx.reshape((-1,1))
    # f = [Polygon(mg.get_cell_vertices(xi[0],xi[1])) for xi in np.hstack([xx,yy])]
    f = [Polygon(mg.get_cell_vertices(xi[0],xi[1])) for xi in zip(xx,yy)]
    
    return f

def get_heads(m):
    hdsobj = bf.HeadFile(os.path.join(m.model_ws, 'Results','RRbin.hds'))
    hds = hdsobj.get_data(kstpkper=hdsobj.get_kstpkper()[-1])
    
    return hds, hdsobj

