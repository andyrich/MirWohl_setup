import flopy

from pathlib import Path
from shutil import copytree, ignore_patterns
import geopandas as gpd
import contextily as ctx
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import conda_scripts.make_map as mp
import flopy.utils.binaryfile as bf
import matplotlib.gridspec as gridspec
import matplotlib as mpl
import cartopy.crs as ccrs

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

    if not os.path.exists(os.path.join('versions', run_name, 'model_files')):
        os.mkdir(os.path.join('versions', run_name, 'model_files'))

    import shutil
    def replace(src, dst):
        shutil.copyfile(src, dst)

    replace(src = 'website_info/subindex.html',
            dst = os.path.join('versions', run_name, 'subindex.html'))

    replace(src = 'website_info/gallery.html',
            dst = os.path.join('versions', run_name, 'gallery.html'))

    replace(src = 'website_info/reach_numbers.png',
            dst = os.path.join('versions', run_name, 'reach_numbers.png'))

    replace(src = 'website_info/sfr_swr_map.png',
            dst = os.path.join('versions', run_name, 'sfr_swr_map.png'))

    replace(src = 'website_info/SWR Reaches.png',
            dst = os.path.join('versions', run_name, 'SWR Reaches.png'))

    replace(src = 'website_info/xsect_locs.png',
            dst = os.path.join('versions', run_name, 'xsect_locs.png'))

    replace(src = 'website_info/lay 1 top.png',
            dst = os.path.join('versions', run_name, 'lay 1 top.png'))


def copy_mod_files(run_name, path=None):
    '''
    copy model files to version/{run_name}/model_files
    :param run_name:
    :param path:
    :return:
    '''
    if path is None:
        path = 'RR_2022'

    if os.path.exists(os.path.join('versions', run_name, 'model_files')):
        shutil.rmtree(os.path.join('versions', run_name, 'model_files'))

    ignore = ignore_patterns('*.exe', '*RR.rch', '*.lst', '*.evt', '*.chk',
                             '*.git*', '*.ipy*', '*.py', 'pond_inflows*',
                             'dis_versions*', 'Results', '^.bat',
                             'IS*.dat', '*.hds', '*.cbc', 'Conv*', 'day*',
                             'solution*', '*.riv')
    print(f"copying files from:\m{path}\n\nto\n{run_name}\model_files")
    copytree(path, os.path.join('versions', run_name, 'model_files'), ignore=ignore)
    make_model_files_html(run_name)


def make_model_files_html(run_name):
    '''
    make html file showing all input files for model
    :param run_name:
    :return:
    '''
    head = '''<h1>Model Files</h1>\n'''
    b = """<a href="{:}"> {:}<br></a>\n""".format

    gallery = Path(os.path.join('versions', run_name, 'model_files')).glob('**/*')
    print(f'writing html file of model input files in versions\{run_name}\model_files.html')
    with open(os.path.join('versions', run_name, 'model_files.html'), 'w') as outfile:
        outfile.write(head)
        for v in gallery:
            if v.is_file():
                outfile.write(b(v.relative_to(Path('versions', run_name)), v.name))
                # print(outfile.write(b(v,v)))



def out_folder(run_name = 'June2015'):
    info, swr, sfr, riv_keys = load_params(run_name)
    
    return os.path.join('versions', info['name'])

def load_model(verbose = False, path = None, nam = 'RRMF.nam'):
    
    if path is None:
        path = 'RR_2022'

    print(path)
    ml = flopy.modflow.Modflow.load(nam,
                                    # load_only= ['DIS', 'BAS6'],
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

def isnumber(x):
    try:
        float(x)
        return float(x)
    except:
        return np.nan

def plot_maps(ml, run_name):
    map_river(ml)
    plt.savefig(os.path.join('versions', run_name, 'modelmap.png'), dpi=250, bbox_inches='tight')

    # # plot swr model cells
    fig,ax = swr_map(ml)
    plt.savefig(os.path.join('versions', run_name, 'sfr_swr_map.png'), dpi = 250)
    
def plot_all_aquifer_props(ml, run_name):
    fig = plot_aquifer_prop(ml, ml.upw.hk.array)
    fname = os.path.join('versions', run_name, 'hk.png')
    plt.savefig(fname, bbox_inches = 'tight')

    fig = plot_aquifer_prop(ml, ml.upw.vka.array, title='Vertical Hydraulic Conductivity')
    fname = os.path.join('versions', run_name, 'vk.png')
    plt.savefig(fname, bbox_inches = 'tight')

    fig = plot_aquifer_prop(ml, ml.upw.ss.array, vmin=0.001, vmax=0.1, title='Specific Storage')
    fname = os.path.join('versions', run_name, 'ss.png')
    plt.savefig(fname, bbox_inches = 'tight')

    fig = plot_aquifer_prop(ml, ml.upw.ss.array*0+0.001, vmin=0.001, vmax=0.1, title='Specific Yield')
    fname = os.path.join('versions', run_name, 'sy.png')
    plt.savefig(fname, bbox_inches = 'tight')

    plt.close('all')
    
def plot_aquifer_prop(ml, array, vmin=0.0001, vmax=10.,
                      cmap =  'viridis', title = "Horizontal Conductivity"):
    plt.figure()
    # fig, ax = plt.subplots(2,3, figsize =(15,15), constrained_layout=True)
    fig = plt.figure(constrained_layout=True,  figsize =(15,15))
    gs = gridspec.GridSpec(2, 3, figure = fig, height_ratios = [3,1])

    axupper = []
    axlower = []
    for lay in range(3):
        ax = fig.add_subplot(gs[0, lay], projection = ccrs.epsg(2226))

        mapview = flopy.plot.PlotMapView(ml,ax = ax)

        hk = array[lay]
        # hk = ml.upw.hy.array[lay]
        ib = ml.bas6.ibound.array[lay]
        hk[ib==0] = np.nan

        ma = np.ma.array(hk, mask = ml.bas6.ibound.array[lay]==0)
        norm=mpl.colors.LogNorm(vmin = vmin, vmax=vmax)
        quadmesh = mapview.plot_array(ma, norm = norm, cmap = cmap)


        ax.set_title(f"layer {lay+1}")
        ax.tick_params(labelbottom=False, labelleft=False)

        ctx.add_basemap(ax, crs = 2226, url = 
           "https://basemap.nationalmap.gov/arcgis/rest/services/USGSTopo/MapServer/tile/{z}/{y}/{x}",
                        attribution='')

        ax2 = fig.add_subplot(gs[1, lay])
        ax2.hist(hk.reshape(-1))

        axupper.extend([ax])
        axlower.extend([ax2])

    cb1 = fig.colorbar(quadmesh, ax=ax, location='right', shrink=.50)
    fig.suptitle(title)

    return fig, axupper, axlower