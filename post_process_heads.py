


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

def run(run_name, head_frequency = 5, add_basemap = False):
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

    m = basic.load_model()
    hdsobj = bf.HeadFile(os.path.join(m.model_ws, 'Results','RRbin.hds'))
    hds = hdsobj.get_data(kstpkper=hdsobj.get_kstpkper()[-1])

    # for more information about GIS export, type help(export_array), for example
    swr_reaches = gpd.read_file("GIS/SWR_Reaches.shp")
    sfr_reaches = gpd.read_file("SFR_files/only_sfr_cells.shp")

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

    fig, axes = plt.subplots(1, 3, figsize=(20, 8.5), subplot_kw=dict(projection=ccrs.epsg(2226)))
    axes = axes.flat
    for cnt,time in enumerate(times):
        hds = hdsobj.get_data(totim = time)

        date = pd.to_datetime(datestart) + pd.to_timedelta(time, unit = 's')
        date = date.strftime('%b, %d %Y') + f"\n{time/86400:.1f} days since start of {datestart}"
        for i, hdslayer in enumerate(hds):
            mapview = flopy.plot.PlotMapView(m,ax = axes[i])
            # linecollection = mapview.plot_grid(linewidth = .3)
            # ax.set_title(f"layer {lay+1} elevation (feet)")
            # Hide X and Y axes tick marks
            axes[i].set_xticks([])
            axes[i].set_yticks([])
            # hdslayer[hdslayer==-999] = np.nan
            # im = axes[i].imshow(hdslayer, vmin=0, vmax=75)
            axes[i].set_title("Layer {}".format(i + 1))
            ma = np.ma.array(hdslayer,mask = m.bas6.ibound.array[i]==0)
            quadmesh = mapview.plot_array(ma, vmax = 80,vmin  =20,cmap = 'gist_ncar_r')
            # ctr = axes[i].contour(ma, colors="k", linewidths=0.5, vmin = 0, vmax = 75)

            set_bounds(axes[i], 'mirabel')
            if add_basemap:
                ctx.add_basemap(axes[i], crs = 2226)

            swr_reaches.plot(ax = axes[i], facecolor = "None", edgecolor = 'k')
            sfr_reaches.plot(ax = axes[i], facecolor = "None", edgecolor = 'grey')

        # fig.delaxes(axes[-1])
        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.85, 0.15, 0.03, 0.7])
        fig.colorbar(quadmesh, cax=cbar_ax, label="Head");
        title = f'Heads at {date}'
        fig.suptitle(title)
        filename = f'Day{cnt}.png'
        plt.savefig(os.path.join(out_folder, 'wl_maps',filename), bbox_inches = 'tight')


        gal = """<div class="gallery">
          <a target="_blank" href="wl_maps/{:}">
            <img src="wl_maps/{:}" alt="{:}" width="1000" height="1000">
          </a>
          <div class="desc">{:}</div>
        </div>""".format
        f = os.listdir(os.path.join(out_folder,'hydrographs'))

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

