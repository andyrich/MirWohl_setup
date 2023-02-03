import os
import geopandas as gpd
import numpy as np
import pandas as pd
import basic
import matplotlib.pyplot as plt
from flopy.utils import ZoneBudget

def run(name, ml = None):
    info, swr_info, sfr_info, riv_keys_info = basic.load_params(name)

    datestart = info['start_date']
    # numdays = info['numdays']
    # name = info['name']
    out_folder = os.path.join('versions', name)

    if ml is None:
        ml = basic.load_model()

    cbc_f = os.path.join(ml.model_ws, 'Results', 'RRMF.cbc')

    zones = ZoneBudget.read_zone_file(os.path.join(ml.model_ws,'zonebud','zonedbud.zbarr'))
    # arrays = gpd.read_file('GIS/grid.shp')
    # arrays = arrays.dropna(subset='zone')
    # arrays.head()
    #
    # zones = np.zeros((ml.dis.nlay, ml.dis.nrow, ml.dis.ncol), dtype=int)
    #
    # for lay in range(3):
    #     zones[lay, arrays.loc[:, 'row'] - 1, arrays.loc[:, 'column'] - 1] = arrays.loc[:, 'zone'].astype(int)

    print('getting budget')

    aliases = {1: 'Mirabel', 2: 'Wohler', 3: 'Upstream'}

    z = ZoneBudget(cbc_f, z=zones, aliases=aliases)

    df = pd.DataFrame(z.get_budget())
    df.loc[:, 'Date'] = pd.to_datetime(datestart) + pd.to_timedelta(df.loc[:, 'stress_period'], unit='D') + \
                        pd.to_timedelta(df.loc[:, 'stress_period'], unit='H')

    df = df.set_index(['Date', 'name']).drop(columns=['totim', 'time_step', 'stress_period'])

    dfi = df.loc[:, ['Wohler']].groupby(level=[0, 1]).sum().unstack().droplevel(0, 1)
    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(10, 6))
    dfi.loc[:, dfi.sum().abs() > 0].filter(regex='FROM_').drop(columns='FROM_STORAGE').plot(ax=ax[0])
    ax[0].legend(loc='upper left', bbox_to_anchor=(1, 1))

    dfi.loc[:, dfi.sum().abs() > 0].filter(regex='TO_').drop(columns='TO_STORAGE').plot(ax=ax[1])
    ax[1].legend(loc='upper left', bbox_to_anchor=(1, 1))

    fig.suptitle('Wohler Zone Area')
    plt.savefig(os.path.join(out_folder, 'zbud_Wohler_zone.png'), bbox_inches = 'tight', dpi=250)


    ###
    dfi = df.loc[:, ['Mirabel']].groupby(level=[0, 1]).sum().unstack().droplevel(0, 1)
    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(10, 6))
    dfi.loc[:, dfi.sum().abs() > 0].filter(regex='FROM_').drop(columns='FROM_STORAGE').plot(ax=ax[0])
    ax[0].legend(loc='upper left', bbox_to_anchor=(1, 1))

    dfi.loc[:, dfi.sum().abs() > 0].filter(regex='TO_').drop(columns='TO_STORAGE').plot(ax=ax[1])
    ax[1].legend(loc='upper left', bbox_to_anchor=(1, 1))
    fig.suptitle('Mirabel Zone Area')
    plt.savefig(os.path.join(out_folder, 'zbud_Mirabel_zone.png'), bbox_inches='tight', dpi=250)

    ###
    dfi = df.loc[:, ['Upstream']].groupby(level=[0, 1]).sum().unstack().droplevel(0, 1)
    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(10, 6))
    dfi.loc[:, dfi.sum().abs() > 0].filter(regex='FROM_').drop(columns='FROM_STORAGE').plot(ax=ax[0])
    ax[0].legend(loc='upper left', bbox_to_anchor=(1, 1))

    dfi.loc[:, dfi.sum().abs() > 0].filter(regex='TO_').drop(columns='TO_STORAGE').plot(ax=ax[1])
    ax[1].legend(loc='upper left', bbox_to_anchor=(1, 1))
    fig.suptitle('Upstream Zone Area')

    plt.savefig(os.path.join(out_folder, 'zbud_Upstream_zone.png'), bbox_inches='tight', dpi=250)


if __name__ == '__main__':
    print('running zonebudget')
    run('June2017')