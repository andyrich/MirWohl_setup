import flopy

import geopandas as gpd
import basic
import contextily as ctx
import pandas as pd
import numpy as np
import cartopy.crs as ccrs
import os
import matplotlib.pyplot as plt



import flopy.utils.binaryfile as bf

import warnings


def set_top_as_start(m):
    '''
    use model top as starting heads
    :param m:
    :return:
    '''
    hdsobj = bf.HeadFile(os.path.join(m.model_ws, 'Results' ,'RRbin.hds'))
    hds = hdsobj.get_data(kstpkper=hdsobj.get_kstpkper()[-1])


    fig, axes = plt.subplots(1, 3, figsize=(20, 8.5), subplot_kw=dict(projection=ccrs.epsg(2226)))
    axes = axes.flat
    grid = m.modelgrid
    for i, hdslayer in enumerate([m.dis.top.array, m.dis.top.array ,m.dis.top.array]):

        filename = os.path.join(m.model_ws, 'inputs', f"start_head_lay{ i +1}.txt")
        np.savetxt(filename, hdslayer, fmt = '%.3f', delimiter = ',')

        mapview = flopy.plot.PlotMapView(m ,ax = axes[i])
        linecollection = mapview.plot_grid(linewidth = .3)

        quadmesh = mapview.plot_array(hdslayer, vmax = 80 ,vmin  =20 ,cmap = 'gist_ncar_r')
        ctx.add_basemap(axes[i], crs = 2226)
        mapview.plot_ibound()
        # ax.set_title(f"layer {lay+1} elevation (feet)")
        # Hide X and Y axes tick marks
        axes[i].set_xticks([])
        axes[i].set_yticks([])
        basic.set_bounds(axes[i], 'mirabel')

        # hdslayer[hdslayer==-999] = np.nan
        # im = axes[i].imshow(hdslayer, vmin=0, vmax=75)
        axes[i].set_title("Layer {}".format(i + 1))
        ctr = axes[i].contour(hdslayer, colors="k", linewidths=0.5, vmin = 0, vmax = 75)

        # export head rasters
        # (GeoTiff export requires the rasterio package; for ascii grids, just change the extension to *.asc)
        flopy.export.utils.export_array(
            grid, "Output_heads/heads{}.tif".format(i + 1), hdslayer
        )

        # export head contours to a shapefile
        flopy.export.utils.export_array_contours(
            grid, "Output_heads/heads{}.shp".format(i + 1), hdslayer
        )

    # fig.delaxes(axes[-1])
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.03, 0.7])
    fig.colorbar(quadmesh, cax=cbar_ax, label="Head")





def set_starting_heads(m, kper = None,):
    vmax = 60
    vmin = 20


    hdsobj = bf.HeadFile(os.path.join(m.model_ws, 'Results','RRbin.hds'))
    if kper is None:
        hds = hdsobj.get_data(kstpkper=hdsobj.get_kstpkper()[-1])
    else:
        kx = hdsobj.get_kstpkper()[kper]
        print(f"using stress period: {kx[1]}, ts {kx[0]} as starting heads")
        hds = hdsobj.get_data(kstpkper=kx)


    fig, axes = plt.subplots(1, 3, figsize=(20, 8.5), subplot_kw=dict(projection=ccrs.epsg(2226)))
    axes = axes.flat
    grid = m.modelgrid
    for i, hdslayer in enumerate(hds):

        # if i==0:
        #     warnings.warn('setting thalweg elevations as starting heads')
        #     hdslayer[min_elev.loc[:,'i'],min_elev.loc[:,'j']] = min_elev.loc[:,'thalweg']

        filename = os.path.join(m.model_ws, 'inputs', f"start_head_lay{i+1}.txt")
        np.savetxt(filename, hdslayer, fmt = '%.3f', delimiter = ',')

        mapview = flopy.plot.PlotMapView(m,ax = axes[i])
        linecollection = mapview.plot_grid(linewidth = .3)

        quadmesh = mapview.plot_array(hdslayer, vmax = vmax,vmin  =vmin, cmap = 'gist_ncar_r')
        ctx.add_basemap(axes[i], crs = 2226)
        # ax.set_title(f"layer {lay+1} elevation (feet)")
        # Hide X and Y axes tick marks
        axes[i].set_xticks([])
        axes[i].set_yticks([])
        basic.set_bounds(axes[i], 'mirabel')



        # hdslayer[hdslayer==-999] = np.nan
        # im = axes[i].imshow(hdslayer, vmin=0, vmax=75)
        axes[i].set_title("Layer {}".format(i + 1))
        # ctr = axes[i].contour(hdslayer, colors="k", linewidths=0.5, vmax = 80,vmin  =20,cmap = 'gist_ncar_r')

        # export head rasters
        # (GeoTiff export requires the rasterio package; for ascii grids, just change the extension to *.asc)
        flopy.export.utils.export_array(
            grid, "Output_heads/heads{}.tif".format(i + 1), hdslayer
        )

        # export head contours to a shapefile
        flopy.export.utils.export_array_contours(
            grid, "Output_heads/heads{}.shp".format(i + 1), hdslayer
        )

    # fig.delaxes(axes[-1])
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.03, 0.7])
    fig.colorbar(quadmesh, cax=cbar_ax, label="Head")


    return fig



def set_start_with_burn(m):

    swr = gpd.read_file("GIS/SWR_Reaches.shp")
    min_elev = pd.read_csv(os.path.join('RR_2022/inputs/stream_thalwegs.txt')).set_index('reach')
    min_elev = pd.merge(min_elev, swr.loc[:,['i','j','rno']], left_index = True, right_on = 'rno').drop(columns = 'rno')
    strt = m.bas6.strt.array

    fig, axes = plt.subplots(1, 3, figsize=(20, 8.5), subplot_kw=dict(projection=ccrs.epsg(2226)))
    axes = axes.flat
    grid = m.modelgrid
    for i, hdslayer in enumerate(strt):

        if i==0:
            warnings.warn('setting thalweg elevations as starting heads')
            hdslayer[min_elev.loc[:,'i'],min_elev.loc[:,'j']] = min_elev.loc[:,'thalweg']

        filename = os.path.join(m.model_ws, 'inputs', f"start_head_lay{i+1}.txt")
        np.savetxt(filename, hdslayer, fmt = '%.3f', delimiter = ',')

        mapview = flopy.plot.PlotMapView(m,ax = axes[i])
        linecollection = mapview.plot_grid(linewidth = .3)

        quadmesh = mapview.plot_array(hdslayer, vmax = 80,vmin  =20, cmap = 'gist_ncar_r')
        ctx.add_basemap(axes[i], crs = 2226)
        # ax.set_title(f"layer {lay+1} elevation (feet)")
        # Hide X and Y axes tick marks
        axes[i].set_xticks([])
        axes[i].set_yticks([])
        basic.set_bounds(axes[i], 'mirabel')

        swr.plot(ax = axes[i], facecolor = "None")

        # hdslayer[hdslayer==-999] = np.nan
        # im = axes[i].imshow(hdslayer, vmin=0, vmax=75)
        axes[i].set_title("Layer {}".format(i + 1))
        # ctr = axes[i].contour(hdslayer, colors="k", linewidths=0.5, vmax = 80,vmin  =20,cmap = 'gist_ncar_r')

        # export head rasters
        # (GeoTiff export requires the rasterio package; for ascii grids, just change the extension to *.asc)
        flopy.export.utils.export_array(
            grid, "Output_heads/heads{}.tif".format(i + 1), hdslayer
        )

        # export head contours to a shapefile
        flopy.export.utils.export_array_contours(
            grid, "Output_heads/heads{}.shp".format(i + 1), hdslayer
        )

    # fig.delaxes(axes[-1])
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.03, 0.7])
    fig.colorbar(quadmesh, cax=cbar_ax, label="Head");

