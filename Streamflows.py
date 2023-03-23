import basic
import pandas as pd
import os
import matplotlib.pyplot as plt
import geopandas as gpd
import conda_scripts.plot_help as ph
import conda_scripts.arich_functions as af


def run(out_folder, ml = None, make_map = True):
    '''
    load gage file outputs and plot hydrographs for each.
    :param make_map:
    :param out_folder:
    :param ml:
    :return:
    '''

    if ml is None:
        ml = basic.load_model(load_fast = ['DIS', 'BAS6', 'UPW', 'NWT', 'WEL', 'GHB', 'OC', 'SFR', 'RCH', 'GAGE'])

    elif ml.gage is None:
        raise ValueError('gage is not loaded in the model. needs to be loaded.')

    dat = get_gage_names()

    if make_map:
        gages = plot_streamflow_map(out_folder, ml = ml, dat = dat)
    else:
        gages = None

    streamflows = plot_streamflows(outfolder = out_folder, dat = dat, model_ws= ml.model_ws, save = True)

    return [gages, streamflows]

def get_gage_names():
    dat = ['rr__outflow_from_swr.gag',
     'rr__inflow_to_swr.gag',
     'rr__below_first_cell.gag',
     'dc__below_first_cell.gag',
     'rr__outflow.gag',
     'mark_west_creek__below_first_cell.gag',
     'mark_west_creek__inflow_to_rr.gag']

    return dat

def plot_streamflow_map(outfolder, ml, dat):
    '''
    plot a map of gages in model
    :param ml:
    :param dat:
    :param outfolder:
    :return: geodataframe of gage locations
    '''

    f = pd.DataFrame(ml.gage.gage_data)
    f.loc[:,'name'] = dat
    f.loc[:,'rename'] = f.loc[:,'name'].str.strip('.gag').str.replace('rr', 'Russian River ').str.replace(
                                                    'dc','dry creek ').str.replace(
                                                    '_',' ').str.title()
    gages = pd.merge(pd.DataFrame(ml.sfr.reach_data), f ,
             left_on = ['iseg', 'ireach'], right_on = ['gageloc','gagerch'] )

    shp = af.get_active_mod_df(ml.modelgrid, ml)

    gages = pd.merge(gages, shp, on = ['i','j'])

    gages =gpd.GeoDataFrame(gages, geometry = 'geometry', crs = 2226)
    gages = gages.set_geometry(gages.geometry.centroid)

    fig, ax = basic.map_river(m = ml, add_basemap = True)

    gages.plot(ax = ax,marker = 'o', facecolor = 'w',edgecolor = 'k',zorder = 100)

    ph.label_points(ax, gages, colname = 'rename', fmt = "s")

    ax.set_title('Gage locations in Model')
    plotname = os.path.join(outfolder, 'streamflow', 'streamflow_gages_map.png')
    print(f'saving {plotname}')

    plt.savefig(plotname, dpi=250, bbox_inches='tight')

    return gages

def plot_streamflows(dat, model_ws, outfolder,  save = False):
    '''
    plot streamflow and return a dictionary of flows
    :param dat: list of gage names in order
    :param model_ws: model workspace
    :param outfolder: version name
    :param save: to same plots
    :return: dictionary of flows
    '''

    streamflows = {}

    for filename in dat:

        file = os.path.join(model_ws, 'Results', filename)
        tab = pd.read_csv(file,
                          skiprows= [0],sep = '\s+')
        f = "DATA: Time           Stage            Flow           Depth           Width      Midpt-Flow         Precip.              ET          Runoff     Conductance        HeadDiff       Hyd.Grad."
        tab.columns = f.replace('DATA: ', '').split()

        tab.index = pd.to_datetime('1/1/2014') + pd.to_timedelta(tab.Time, unit = 's')
        tab = tab.drop(columns = 'Time')
        tab = tab.loc[:,tab.abs().sum()>0]
        axes = tab.plot(subplots = True, figsize = (8,6), fontsize = 12)

        [{ax.text(1,.5, ax.get_legend_handles_labels()[1][0],
                 transform=ax.transAxes, va = 'top'), ax.legend().remove()} for ax in axes]

        rename = filename.strip('.gag').replace('rr', 'Russian River ').replace(
                                                    'dc','dry creek ').replace(
                                                    '_',' ').title()
        plt.suptitle(rename)

        if save:
            plotname = os.path.join(outfolder,'streamflow', rename+'.png')
            print(f'saving {plotname}')
            plt.savefig(plotname,dpi = 250, bbox_inches = 'tight')

        print(f"done loading {rename} from\n\t{filename}\n")
        streamflows[filename] = tab

    return streamflows


def streamflow_to_html_subindex(run):
    '''
    add fields to version's subindex
    :param run:
    :return:
    '''

    file = os.path.join('versions', run, 'subindex.html')

    h1 = 'Results - Streamflow'
    fields = {"Streamflow Gage map": r'streamflow\streamflow_gages_map.png',
              'Russian River   Outflow From Swr': r'streamflow\Russian River   Outflow From Swr.png',
              'Russian River   Inflow To Swr': r'streamflow\Russian River   Inflow To Swr.png',
              'Russian River   Below First Cell': r'streamflow\Russian River   Below First Cell.png',
              'Dry Creek   Below First Cell': r'streamflow\Dry Creek   Below First Cell.png',
              'Russian River   Outflow': r'streamflow\Russian River   Outflow.png',
              'Mark West Creek  Below First Cell': r'streamflow\Mark West Creek  Below First Cell.png',
              'Mark West Creek  Inflow To Russian River ': r'streamflow\Mark West Creek  Inflow To Russian River .png'}

    basic.add_subindex_fields(file, h1, fields)


