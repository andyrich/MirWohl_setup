


import flopy
import os
import geopandas as gpd
import basic
import contextily as ctx
import pandas as pd
import numpy as np
import sys
import matplotlib as mpl
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import conda_scripts.utils as utils


from flopy.utils.postprocessing import (
    get_transmissivities,
    get_water_table,
    get_gradients,
)
import flopy.utils.binaryfile as bf


# In[2]:


import basic




# info, swr_info, sfr_info, riv_keys_info = basic.load_params('June2016')
#
# datestart = info['start_date']
#
# name = info['name']
#
# out_folder = basic.out_folder('June2016')
#
#
# print(datestart)
# print(out_folder)

def run(run_name, head_frequency = 5, add_basemap = False, ml = None, plot_thickness = True):
    '''
    Plot heads in each layer; export the heads and head along with gallery html file to view with
    :param run_name:
    :param head_frequency:
    :return: Nome
    '''
    info, swr_info, sfr_info, riv_keys_info = basic.load_params(run_name)

    datestart = info['start_date']

    name = info['name']

    out_folder = basic.out_folder(run_name)

    print(datestart)
    print(out_folder)

    if ml is None:
        ml = basic.load_model()

    hdsobj = bf.HeadFile(os.path.join(ml.model_ws, 'Results','RRbin.hds'))
    hds = hdsobj.get_data(kstpkper=hdsobj.get_kstpkper()[-1])

    # for more information about GIS export, type help(export_array), for example
    swr_reaches = gpd.read_file("GIS/SWR_Reaches.shp")
    sfr_reaches = gpd.read_file("SFR_files/only_sfr_cells.shp")
    wells = gpd.read_file("GIS/wells.shp")

    head = ''' <html>
    <head>
    <style>
    div.gallery {
      margin: 5px;
      border: 1px solid #ccc;
      float: left;
      width: 1080px;
    }
    
    div.gallery:hover {
      border: 1px solid #777;
    }
    
    div.gallery img {
      width: 100%;
      height: auto;
    }
    
    div.desc {
      padding: 15px;
      text-align: center;
    }
    </style>
    </head>
    <body>'''

    tail = '''</body>
    </html> '''



    times = hdsobj.get_times()[::head_frequency]

    gallery = list()
    for cnt, time in enumerate(times):
        hds = hdsobj.get_data(totim=time)
        date = pd.to_datetime(datestart) + pd.to_timedelta(time, unit = 's')
        date = date.strftime('%b, %d %Y') + f"\n{time/86400:.1f} days since start of {datestart}"
        title = f'Heads at {date}'

        if plot_thickness:
            plot_water_table_and_sat_thickness(hds, swr_reaches, sfr_reaches, ml,wells,  title, add_basemap=add_basemap)
        else:
            plot_all_heads(                    hds, swr_reaches, sfr_reaches, ml, title, add_basemap = add_basemap)

        filename = f'Day{cnt}.png'

        plt.savefig(os.path.join(out_folder, 'wl_maps',filename), bbox_inches = 'tight', dpi = 250)

        gal = """<div class="gallery">
          <a target="_blank" href="wl_maps/{:}">
            <img src="wl_maps/{:}" alt="{:}" width="1000" height="1000">
          </a>
          <div class="desc">{:}</div>
        </div>""".format
        # f = os.listdir(os.path.join(out_folder,'hydrographs'))

        gallery.append(gal(filename,  filename, title.split('\n')[0], title.split('\n')[0]))

    with open(os.path.join(out_folder, 'gallery_wl_new.html'), 'w') as outfile:
        outfile.write(head)
        for v in gallery:
            outfile.write(v)
        outfile.write(tail)


def set_bounds(ax, locname):

    locs =   { "MIRABEL": [-122.909128, 38.4911643, -122.85212, 38.52827]}
    tb = locs[locname.upper()]
    box = [tb[0], tb[2], tb[1], tb[3]]

    ax.set_extent(box)


def plot_water_table_and_sat_thickness(hds, swr_reaches, sfr_reaches, ml, wells, title, add_basemap = True):
    '''
    plot water table elevation and thickness
    :param hds:
    :param swr_reaches:
    :param sfr_reaches:
    :param ml:
    :param title:
    :param add_basemap:
    :return:
    '''
    fig, axes = plt.subplots(2, 1, figsize=(5, 8), gridspec_kw={'hspace': .001},
                             subplot_kw=dict(projection=ccrs.epsg(2226)))

    ax = axes[0]
    mapview = flopy.plot.PlotMapView(ml, ax=ax)
    # arr = np.ma.array(ml.modelgrid.saturated_thick(hds[0]), mask = ml.bas6.ibound.array[0] == 0)
    sat = ml.modelgrid.saturated_thick(hds)
    sat = sat[0] + sat[1]
    arr = np.ma.array(sat, mask=ml.bas6.ibound.array[0] == 0)

    bounds = np.arange(0, 100, 5)
    norm = mpl.colors.BoundaryNorm(bounds, 256)

    # ac = mapview.plot_array(arr, vmin=0, vmax=90, cmap='gist_ncar_r')
    ac = mapview.plot_array(arr, norm=norm, cmap='gist_ncar_r')
    mapview.contour_array(arr, levels=np.arange(0, 100, 20), colors='k')
    basic.set_bounds(ax, 'mirabel')
    ax.set_title('Saturated Thickness')
    plt.colorbar(ac, shrink=.5, ax=ax, label='feet')

    if add_basemap:
        ctx.add_basemap(ax,
                        crs=2226,
                        source=r"C:\GIS\basemap\SRP_hydro.tif")

    wt = flopy.utils.postprocessing.get_water_table(hds, nodata=-999)
    wt = np.ma.array(wt, mask=ml.bas6.ibound.array[0] == 0)

    mapview = flopy.plot.PlotMapView(ml, ax=axes[1])
    bounds = np.arange(0, 80, 5)
    norm = mpl.colors.BoundaryNorm(bounds, 256)
    ac = mapview.plot_array(wt, norm=norm, cmap='rainbow_r')
    mapview.contour_array(wt, levels=np.arange(0, 100, 20), colors='k')
    basic.set_bounds(axes[1], 'mirabel')
    axes[1].set_title('Watertable Elevation')
    plt.colorbar(ac, shrink=.5, ax=axes[1], label='feet')
    if add_basemap:
        ctx.add_basemap(axes[1],
                        crs=2226,
                        source=r"C:\GIS\basemap\SRP_hydro.tif")

    [(swr_reaches.plot(ax=axi), sfr_reaches.plot(ax=axi), wells.plot(ax=axi, facecolor='None')) for axi in axes]

    fig.suptitle(title)

def plot_all_heads(hds, swr_reaches, sfr_reaches, ml, title, add_basemap = True):
    '''
    plot heads for all layers
    :param hds:
    :param swr_reaches:
    :param sfr_reaches:
    :param ml:
    :param title:
    :param add_basemap:
    :return:
    '''

    nlayers = basic.get_num_lays(ml)
    fig, axes = plt.subplots(1, nlayers, figsize=(20, 8.5), subplot_kw=dict(projection=ccrs.epsg(2226)))
    axes = axes.flat

    #when layer 3 is off via ibound, remove layer 3 heads
    hds = hds[0:nlayers]

    for i, hdslayer in enumerate(hds):
        mapview = flopy.plot.PlotMapView(ml, ax=axes[i])
        axes[i].set_facecolor('lightgrey')
        # linecollection = mapview.plot_grid(linewidth = .3)
        # ax.set_title(f"layer {lay+1} elevation (feet)")
        # Hide X and Y axes tick marks
        axes[i].set_xticks([])
        axes[i].set_yticks([])
        # hdslayer[hdslayer==-999] = np.nan
        # im = axes[i].imshow(hdslayer, vmin=0, vmax=75)
        axes[i].set_title("Layer {}".format(i + 1))
        ma = np.ma.array(hdslayer, mask=ml.bas6.ibound.array[i] == 0)
        quadmesh = mapview.plot_array(ma, vmax=50, vmin=10, cmap='gist_ncar_r')
        # ctr = axes[i].contour(ma, colors="k", linewidths=0.5, vmin = 0, vmax = 75)

        set_bounds(axes[i], 'mirabel')
        if add_basemap:
            ctx.add_basemap(axes[1],
                            crs=2226,
                            source=r"C:\GIS\basemap\SRP_hydro.tif")

        swr_reaches.plot(ax=axes[i], facecolor="None", edgecolor='k')
        sfr_reaches.plot(ax=axes[i], facecolor="None", edgecolor='grey')

    # fig.delaxes(axes[-1])
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.03, 0.7])
    fig.colorbar(quadmesh, cax=cbar_ax, label="Head")

    fig.suptitle(title)