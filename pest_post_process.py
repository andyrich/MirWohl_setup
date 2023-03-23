import os
import pyemu
import matplotlib.pyplot as plt
import geopandas as gpd
import basic
import conda_scripts.rich_gis as rg

def get_linear_analysis(ml, pest_file_name):
    LA = pyemu.LinearAnalysis(jco=os.path.join(ml.model_ws, f'{pest_file_name}.jco'),
                              pst=os.path.join(ml.model_ws, f'{pest_file_name}.pst'))
    return LA

def par_sensitivity(LA, out_folder):
    
    '''
    get parameter and observation sensitivity
    '''

    sens = LA.get_par_css_dataframe()
    sens.loc[:,['pest_css']].sort_values('pest_css').plot.barh(title = 'Parameter Sensitivities', figsize = (9,6))
    plt.savefig(os.path.join(out_folder, 'parsen.png'), bbox_inches = 'tight')

    return sens

def obs_sensitivity(LA, out_folder):
    '''
    plot observation sensitivity
    :param LA:
    :param out_folder:
    :return:
    '''
    ccs = LA.get_cso_dataframe()
    
    ccs.loc[:,'type'] = ccs.index.map(lambda x: 'drawdown' if x.startswith('d') else 'obs')

    ccs.loc[:,'well'] = ccs.index.map(lambda x: 'LRR{0:04s}'.format( x.replace('d','').replace('l', '').split('.')[0]))
    
    ccs.groupby(['well', 'type']).mean().unstack(1).droplevel(0,1)\
    .sort_values('obs', ascending = False).rename(columns = {'obs':'head'})\
    .plot.bar(title ="Mean Observation Sensitivity", xlabel = '', figsize = (9,6))
    plt.savefig(os.path.join(out_folder, 'obssen.png'), bbox_inches = 'tight')
    
    return ccs

def map_css_mean(ccs, ml, out_folder):
    '''
    map average observation sensitivities per well
    :param ccs:
    :param ml:
    :param out_folder:
    :return:
    '''
    wells_mod = gpd.read_file('GIS/wells_mod.geojson')
    wells_mod = wells_mod.set_index('station_name').join(ccs.groupby(['well', 'type']).mean(
    ).unstack(1).droplevel(0,1))

    wells_mod.loc[:,'obslabel'] = wells_mod.loc[:,'obs'].apply(lambda x: f"{x:.2f}")
    wells_mod.loc[:,'drawdownlabel'] = wells_mod.loc[:,'drawdown'].apply(lambda x: f"{x:.2f}")

    for col in ['obs', 'drawdown']:
        fig, ax = basic.basic_map(m = ml)

        if col == 'obs':
            fac = 50
        else:
            fac = 200

        cdf = wells_mod.dropna(subset = col)
        cdf.plot(col, cmap = 'bwr',
                       legend = True,
                        legend_kwds={
                                    "location":"right",
                                    "shrink":.2
                                    },
                       ax=  ax, 
                       markersize  = cdf.loc[:,col]*50, 
                       edgecolor = 'k')

        basic.set_bounds(ax)
        v = col.replace('obs','Observed Head')
        ax.set_title(f"Mean {v} Sensitivities")
        ax.text(0,1, 'Average Sensitivities Per Well', va ='top',transform = ax.transAxes)
        plt.draw()
        rg.label_points(ax, cdf, basin_name=None, colname = col+'label', already_str = True)
        plt.savefig(os.path.join(out_folder, f'obssen_map_{col}.png'), bbox_inches = 'tight')
        
def map_css_total(ccs, ml, out_folder):
    '''
    map the total observation sensitivities per well
    :param ccs:
    :param ml:
    :param out_folder:
    :return:
    '''
    wells_mod = gpd.read_file('GIS/wells_mod.geojson')
    wells_mod = wells_mod.set_index('station_name').join(ccs.groupby(['well', 'type']).sum(
    ).unstack(1).droplevel(0,1))

    wells_mod.loc[:,'obslabel'] = wells_mod.loc[:,'obs'].apply(lambda x: f"{x:.0f}")
    wells_mod.loc[:,'drawdownlabel'] = wells_mod.loc[:,'drawdown'].apply(lambda x: f"{x:.0f}")

    for col in ['obs', 'drawdown']:
        fig, ax = basic.basic_map(m = ml)
        if col == 'obs':
            fac = .1
        else:
            fac = .3
        cdf = wells_mod.dropna(subset = col)
        cdf.plot(col, cmap = 'bwr',
                            legend = True,
                            legend_kwds={
                            "location":"right",
                            "shrink":.2
                            },
                       ax=  ax, 
                       markersize  = cdf.loc[:,col]*fac, 
                       edgecolor = 'k')

        basic.set_bounds(ax)
        v = col.replace('obs','Observed Head')
        ax.set_title(f"Total {v} Sensitivities")
        ax.text(0,1, 'Total Sensitivities Per Well', va ='top',transform = ax.transAxes)
        plt.draw()
        rg.label_points(ax, cdf, basin_name=None, colname = col+'label', already_str = True)
        plt.savefig(os.path.join(out_folder, f'obssen_map_{col}_Total.png'), bbox_inches = 'tight')