import write_inflows
import calendar, os
import basic
import pandas as pd
import matplotlib.pyplot as plt
import make_wells
import numpy as np
import conda_scripts.arich_functions as af
import matplotlib.dates as mdates

def run_long_context():
    rr, dry, mw, total = get_data()
    plot(rr, dry, mw, total)

    make_mean_q(total)

    wells_df = wells()
    quarterly_plot(wells_df)
    plot_total_by_season(wells_df)
    plot_total_by_season_ts(wells_df)


def run_current_context(run):
    '''
    do plot of current year with flows overlain on eachother
    :param run:
    :return:
    '''
    rr, dry, mw, total = get_data()
    # plot the curretn year
    info, swr_info, sfr_info, riv_keys_info = basic.load_params(run)
    year = pd.to_datetime(info['start_date']).year
    outfolder = basic.out_folder(run)
    plot_by_wy(rr, dry, mw, total, outfolder=outfolder, year = year)


def plot(rr, dry, mw, total):
    fig, ax = plt.subplots(4, 1, sharex=True, figsize=(10, 6), )
    plt.yscale("log")

    rr.plot(ax=ax[0])
    dry.plot(ax=ax[1])
    mw.plot(ax=ax[2])
    total.plot(ax=ax[3])
    [(axi.grid(True), axi.set_yscale("log")) for axi in ax]
    [axi.set_ylabel('cfs') for axi in ax]

    plt.savefig('versions/website_info//allflows.png', dpi=300, bbox_inches='tight')

    fig, ax = plt.subplots(4, 1, sharex=True, figsize=(10, 6))
    plt.yscale("log")

    mark = 's'
    rr.resample('1Q').sum().plot(ax=ax[0], marker=mark)
    dry.resample('1Q').sum().plot(ax=ax[1], marker=mark)
    mw.resample('1Q').sum().plot(ax=ax[2], marker=mark)
    total.resample('1Q').sum().plot(ax=ax[3], marker=mark)

    [(axi.grid(True), axi.set_yscale("log")) for axi in ax]
    [axi.set_ylabel('cfs') for axi in ax]
    plt.savefig('versions/website_info//quarterly_flows_ts.png', dpi=300, bbox_inches='tight')

    fig, ax = plt.subplots(4, 1, sharex=True, figsize=(10, 6))
    plt.yscale("log")

    mark = 's'
    make_quarterly(rr).plot(ax=ax[0], marker=mark, legend=True, title='Russian River')
    make_quarterly(dry).plot(ax=ax[1], marker=mark, legend=False, title='Dry Creek')
    make_quarterly(mw).plot(ax=ax[2], marker=mark, legend=False, title='Mark West Creek')
    make_quarterly(total).plot(ax=ax[3], marker=mark, legend=False, title='RR + Dry Creek Creek')

    ax[0].legend(bbox_to_anchor=(1, 0), loc='lower left')
    [(axi.grid(True), axi.set_yscale("log")) for axi in ax]
    [axi.set_ylabel('cfs') for axi in ax]
    plt.savefig('versions/website_info//quarterly_flows_ts_set.png', dpi=300, bbox_inches='tight')


def plot_by_wy(df1, df2, df3, df4, outfolder, year=None, ):
    fig, ax = plt.subplots(4, 1, sharex=True, figsize=(10, 6), )

    for cnt, df in enumerate([df1, df2, df3, df4]):
        for group, df in get_wy(df).groupby('Water Year'):
            df = df.sort_index()
            if group == year:
                c = 'k'
                lw = 3
            else:
                c = None
                lw = 1
            ax[cnt].plot(df.loc[:, 'Date'], df.iloc[:, 0], label=f"WY {group}", c=c, lw=lw)

        ax[cnt].set_yscale("log")
        ax[cnt].set_title(df.columns[0])
        ax[cnt].set_ylabel('cfs')
        ax[cnt].grid(True)
    ax[0].legend(loc='upper left', bbox_to_anchor=(1, 1))

    ax[0].xaxis.set_minor_locator(mdates.MonthLocator(range(1, 13)))
    ax[0].xaxis.set_major_locator(mdates.MonthLocator(range(1, 13)))
    ax[0].xaxis.set_major_formatter(mdates.DateFormatter('%b'))

    plt.savefig(os.path.join(outfolder, 'hydro_yearly_oneplot.png'), bbox_inches = 'tight')


def get_wy(ser):
    ser.loc[:, 'Water Year'] = af.water_year(ser.index)
    ser.loc[:, 'Date'] = af.julian_water_year(ser.index.to_series())

    return ser


def get_data():
    '''

    :param run_name: to load from run_names.txt
    :param reload: if true will load from spreadsheets, wiski, etc and create file. False will just re-load old file.
    :return: None
    '''
    run_name = 'June2012'
    info, swr_info, sfr_info, riv_keys_info = basic.load_params(run_name)

    # datestart = info['start_date']
    datestart = '1/1/2012'
    # numdays = info['numdays']
    numdays = (pd.to_datetime('10/1/2022')-pd.to_datetime(datestart)).days
    name = info['name']

    out_folder = basic.out_folder(run_name)

    # print(datestart)
    # print(out_folder)

    m = basic.load_model()
    flow_dict = write_inflows.flo_dict()
    start_year = pd.to_datetime(datestart).year

    rr = write_inflows.load_riv(station='11464000', title='Russian River', file='RRinflow.dat', figurename='russian_river.png',
                  datestart = datestart, out_folder = out_folder, m = m, numdays=numdays, save_fig=False, write_output=False)

    dry = write_inflows.load_riv(station='11465350', title='Dry Creek', file='Dry_creek.dat', figurename='dry_creek.png',
                   datestart=datestart, out_folder=out_folder, m=m, numdays=numdays, save_fig=False, write_output=False)

    mw = write_inflows.load_riv(station='11466800', title='Mark West Creek', file='MarkWest.dat', figurename='mark_west.png',
                   datestart=datestart, out_folder=out_folder, m=m, numdays=numdays, save_fig=False, write_output=False)

    total = dry.loc[:, 'Q'] + rr.loc[:, 'Q']
    total = total.to_frame('rrtotal')
    
    rr = rr.drop(columns = 'time').rename(columns = {'Q' :"Russian River at Healdsburg"})
    dry = dry.drop(columns = 'time').rename(columns = {'Q' :"Dry Creek"})
    mw = mw.drop(columns = 'time').rename(columns = {'Q' :"Mark West Creek"})
    total = total.rename(columns = {'rrtotal': "Russian River + Dry Creek"})
    
    return rr, dry, mw, total


def make_quarterly(df):
    t = df.resample('1Q').mean()
    t.loc[:,'year'] = t.index.year
    t.loc[:,'month'] = t.index.month

    t = t.set_index(['year','month']).unstack().droplevel(0,1)

    t = t.rename(columns = lambda x: f'{calendar.month_abbr[x-2]} to {calendar.month_abbr[x]}')

    t = t.sort_index()
    
    return t



    
def make_mean_q(total):
    t = make_quarterly(total)

    axs = t.plot.bar(
        subplots = True, legend = False, figsize = (10,6), facecolor = 'k', edgecolor = 'k')


    [axi.annotate(axi.get_title(),
                xy=(1, 1), xycoords='axes fraction',
                xytext=(1., 1.), textcoords='axes fraction',
                bbox=dict(facecolor='wheat'),
                horizontalalignment='right', verticalalignment='top') for axi in axs]

    [(axi.set_title(''), axi.grid(True), axi.set_ylabel('cfs')) for axi in axs]

    [(axi.spines['right'].set_visible(False),
    axi.spines['top'].set_visible(False)) for axi in axs]
    plt.suptitle('Average Discharge for Russian River at Healdsburg + Dry Creek')

    plt.savefig('versions/website_info//quarterly_flows.png',dpi = 300, bbox_inches = 'tight')
    

def wells():
    print('running wel creation package')
    datestart = '6/19/2012'

    numdays = (pd.to_datetime('10/1/2022') - pd.to_datetime(datestart)).days

    df = make_wells.load_caissons()
    df = make_wells.get_period(df, datestart, numdays, False)
    df = df.droplevel(1,0)
    df = df/43560.

    return df


def quarterly_plot(wells_df):    
    # Tick every year on Jan 1st
    from matplotlib.dates import YearLocator, MonthLocator, ConciseDateFormatter
    import adjustText

    # Tick every 5 years on July 4th
    locator = YearLocator(1, month=1, day=1)

    df = wells_df.resample('1Q').sum()
    df.insert(0,0, 0)
    df = df.cumsum(axis=1)

    fig = plt.figure(figsize = (10,6))
    ax = plt.subplot(111, )
    for col in range(1,df.shape[1]):
        ax.bar(df.index, height = df.iloc[:,col]-df.iloc[:,col-1], 
               width=365/5, bottom = df.iloc[:,col-1], edgecolor = 'w',
              align = 'edge', label = df.columns[col])

    ax.xaxis_date()
    formatter = ConciseDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    ax.xaxis.set_minor_locator(MonthLocator([3,6,9],1,4))

    xxx = [[df.index[-1] + pd.to_timedelta(75, unit='D'),
            (df.tail(1).loc[:, df.columns[col - 1]].values + df.tail(1).loc[:, df.columns[col]].values) / 2,
            df.columns[col]] for col in range(1, 7)]
    xxx = np.array(xxx)
    texts = [ax.text(ti, xi, yi, ha='left') for ti, xi, yi in xxx]

    # texts = [ax.text(df.index[-1]+pd.to_timedelta(75, unit = 'D'),
    #                  (df.tail(1).loc[:,df.columns[col-1]].values+df.tail(1).loc[:,df.columns[col]].values)/2,
    #                   df.columns[col])  for col in range(1,7)]

    # print(xxx)
    # adjustText.adjust_text(texts, x = xxx[:,0],  y = xxx[:,1], only_move={'point':'','text':'y'})

    ax.set_ylabel('Acre Feet')
    ax.set_title('Total Pumping per Caisson')

    plt.savefig('versions/website_info//total_yearly_pumping.png',dpi = 300, bbox_inches = 'tight')

def plot_total_by_season(wells_df):
    fig, ax = plt.subplots(6,1, sharex = True, sharey = True, figsize = (10,6))
    # plt.yscale("log")

    for cnt in range(6):
        col = wells_df.columns[cnt]
        make_quarterly(wells_df.loc[:,[col]]).plot.bar(cmap = 'jet', ax = ax[cnt], 
                                                                         legend = True if cnt==0 else False, title = False
                                                                          )

        if cnt==0:
            ax[cnt].legend(loc = 'upper left', bbox_to_anchor = (1,1))

        ax[cnt].set_ylabel('AF/d')
        ax[cnt].annotate(col,
                    xy=(1, 1), xycoords='axes fraction',
                    xytext=(1., 1.), textcoords='axes fraction',
                    bbox=dict(facecolor='wheat'),
                    horizontalalignment='right', verticalalignment='top') 

    plt.suptitle('Average Pumping per Caisson, by Season')
    plt.savefig('versions/website_info//seasonal_pumping.png',dpi = 300, bbox_inches = 'tight')


def plot_total_by_season_ts(wells_df):
    fig, ax = plt.subplots(6,1, sharex = True, sharey = False, figsize = (10,6))
    # plt.yscale("log")

    for cnt in range(6):
        col = wells_df.columns[cnt]
        make_quarterly(wells_df.loc[:,[col]]).plot(cmap = 'jet', ax = ax[cnt], 
                                                   marker = 'o',
                         legend = True if cnt==0 else False, title = False
                                                                          )

        if cnt==0:
            ax[cnt].legend(loc = 'upper left', bbox_to_anchor = (1,1))

        ax[cnt].set_ylabel('AF/d')
        ax[cnt].annotate(col,
                    xy=(1, 1), xycoords='axes fraction',
                    xytext=(1., 1.), textcoords='axes fraction',
                    bbox=dict(facecolor='wheat'),
                    horizontalalignment='right', verticalalignment='top') 

    plt.suptitle('Average Pumping per Caisson, by Season')
    plt.savefig('versions/website_info//seasonal_pumping_ts.png',dpi = 300, bbox_inches = 'tight')


if __name__ == "__main__":
    run_long_context()