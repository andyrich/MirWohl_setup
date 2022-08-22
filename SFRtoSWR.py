

import geopandas as gpd
import basic

import pandas as pd
import numpy as np

import os
import matplotlib.pyplot as plt
import pathlib

from scipy.spatial.distance import pdist, squareform




import warnings

def run(run_name):
    info, swr_info, sfr_info, riv_keys_info = basic.load_params(run_name)

    datestart = info['start_date']

    name = info['name']
    m = basic.load_model()
    out_folder = basic.out_folder(run_name)
    print(datestart)
    print(out_folder)
    basic.map_river(m)
    plt.savefig(os.path.join(out_folder, 'modelmap.png'),dpi = 250, bbox_inches = 'tight')

    print('done with map')


    sfr = gpd.read_file('GIS/nhd_hr_demo_sfr_cells.shp')
    sfr = sfr.query("name=='Russian River'")

    print('loading swr_info')
    IGEOTYPE = swr_info["IGEOTYPE"]
    IGCNDOP = swr_info["IGCNDOP" ]
    GMANNING = swr_info["GMANNING" ]
    LEAKANCE = swr_info["LEAKANCE"]
    getextd = swr_info["getextd"]
    bottom = swr_info['dam_swr_rno']
    reach_min = swr_info["reach_main"]
    IROUTETYPE = swr_info["IROUTETYPE"]
    use_average_reach_len = swr_info["use_average_reach_len"]
    set_start_stage = swr_info['set_start_stage']
    use_thalweg = swr_info['use_thalweg']
    kper = None if swr_info["kper"].lower()=='none' else swr_info["kper"]
    SWR_processor = swr_info["SWR_processor"]
    rain = swr_info['rain']
    evap = swr_info['evap']



    # Function to find adjacent
    # elements in List
    def findAdjacentElements(index, test_list):
        res = []
        for idx, ele in enumerate(test_list):
        # for idx, ele in zip(index, test_list):

            # Checking for all cases to append
            if idx == 0:
                res.append((1, test_list[idx + 1]-1))
            elif idx == len(test_list) - 1:
                res.append((1, test_list[idx - 1]-1))
            else:
                res.append((2, test_list[idx - 1]-1, test_list[idx + 1]-1))
        return dict(zip(index, res))

    def find_route(sfr, bottom, n = 10, reach_min = 578, IROUTETYPE  = 4, use_average_reach_len = False ):
        if reach_min is None:
            reach_min = bottom-n
        print(f'\nselecting reaches:\t{bottom} to {reach_min}')

        sfr = sfr.query(f"rno<={bottom} & rno>={reach_min} ")

        print('dropping duplicates')
        sh = sfr.shape
        sfr = sfr.drop_duplicates(['k','i','j'])
        print(f'\nold length: {sh[0]}\nnew length: {sfr.shape[0]}\n')

        sfr = sfr.sort_values('rno')

        min_r = sfr.rno.min() - 1 # offset to get reach 1
        sfr.loc[:,'rno_og'] = sfr.loc[:,'rno'].copy()
        sfr.loc[:,'rno'] = np.arange(1, sfr.shape[0]+1)
        sfr.loc[:,'outreach'] = np.arange(2, sfr.shape[0]+2) #sfr.loc[:,'rno'].shift(-1, fill_value=-999)
        # sfr.loc[:,'rno'] =  sfr.loc[:,'rno'] - min_r
        # sfr.loc[:,'outreach'] =  sfr.loc[:,'outreach'] - min_r
        sfr.loc[:,'ieqn'] = IROUTETYPE
        sfr.loc[:,'irgnum'] = sfr.loc[:,'rno']
        sfr.loc[:,'krch'] = 1 # layer 1
        sfr.loc[:,'IRCH'] = sfr.loc[:,'i']+1
        sfr.loc[:,'JRCH'] = sfr.loc[:,'j']+1

        # reach location data
        #IREACH IEQN IRGNUM KRCH   IRCH JRCH   RLEN
           # 1    4     01    1     161  62  304.8
           # 2    4     02    1     162  62  304.8
           # 3    4     03    1     163  61  304.8
           # 4    4     04    1     164  61  304.8
           # 5    4     05    1     165  61  304.8
        reach_loc = sfr.loc[:,['rno','ieqn', 'irgnum', 'krch', 'IRCH', 'JRCH', 'rchlen' ]]

        if use_average_reach_len:
            warnings.warn('\nusing average reach lengths, not actual reach lengths\n'*3)
            reach_loc.loc[:,'rchlen'] = reach_loc.loc[:,'rchlen'].mean()

        rout = findAdjacentElements(sfr.loc[:,'rno'].values, sfr.loc[:,'outreach'].values, )

        return sfr, reach_loc, rout

    def write_reach_connect(rout):
        with open('RR_2022/inputs/rout.txt','w') as r:
            for key, value in rout.items():
                v = '    '.join([str(v) for v in value])
                r.write(f"{key}   {v}\n")

    def write_reach_loc_dat(sfr_filt):
        sfr_filt.to_csv('RR_2022/inputs/reach_loc.txt', sep = ' ', index = False, header = False)

    def write_stress_period(sfr_filt):
        '''
          # DATASET 5 - STRESS PERIOD 1
        # ITMP IRDBND IRDRAI IRDEVP IRDLIN IRDGEO IRDSTR IRDSTG IPTFLG [IRDAUX]
         1     5    0     0       1     5      1     5      1      # 5

        '''

        n = sfr_filt.shape[0]
        with open("RR_2022/inputs/stress_period_data.txt",'w') as sp:
            sp.write("# DATASET 5 - STRESS PERIOD\n")
            sp.write( "# ITMP IRDBND IRDRAI IRDEVP IRDLIN IRDGEO IRDSTR IRDSTG IPTFLG [IRDAUX]\n")
            sp.write(f"     1     {n}    0     0       1     {n}      1     {n}      1      # 5")

    def write_dataset_6(sfr_filt):
            with open('RR_2022/inputs/dataset6.txt','w') as r:
                for i in sfr_filt.loc[:,'rno']:
                    r.write(f"{i}  1\n")

    def write_dataset_evap(sfr_filt,evap =0.000):
        sfr_filt.loc[:,'evap'] = evap
        # sfr_filt.loc[:,['rno','evap']]
        sfr_filt.loc[:,['rno','evap']].rename(columns = {'rno':'#rno'}).to_csv('RR_2022/inputs/evap.tab', sep = ' ', index = False)

    def write_dataset_rain(sfr_filt,rain =0.000):
        sfr_filt.loc[:,'rain'] = rain
        # sfr_filt.loc[:,['rno','evap']]
        sfr_filt.loc[:,['rno','rain']].rename(columns = {'rno':'#rno'}).to_csv('RR_2022/inputs/rain.tab', sep = ' ', index = False)

    def write_dataset_10(sfr_filt, shift = 0):
        # DATASET 10 - GEOMETRY ASSIGNMENT DATA
        sfr_filt.loc[:,'IGMODRCH'] = sfr_filt.loc[:,'rno']
        sfr_filt.loc[:,'IGEONUMR'] = sfr_filt.loc[:,'rno']
        sfr_filt.loc[:,'GZSHIFT'] = shift
        #IGMODRCH IGEONUMR GZSHIFT
        sfr_filt.loc[:,['IGMODRCH','IGEONUMR','GZSHIFT' ]].rename(columns = {"IGMODRCH":"#IGMODRCH"}).to_csv('RR_2022/inputs/dataset_10.txt', sep = ' ', index = False)



    def write_dataset_14a(sfr_filt, m, shift = 5):
        bot = m.dis.top
        sfr_filt.loc[:,'top'] = bot[sfr_filt.loc[:,'i'], sfr_filt.loc[:,'j']] + shift

        # print(sfr_filt.loc[:,'top'])
        sfr_filt.loc[:,['rno', 'top']].to_csv('RR_2022/inputs/dataset_14a.txt', sep = ' ', index = False)


    def write_start_stage(sfr_filt, stage_shift_from_mod_top = .5, use_thalweg = True, kper = None):
        '''
        uses ending heads of last model run to create new starting heads
        '''
        path = pathlib.Path(m.model_ws, 'Results', 'ISWRPSTG.dat')
        # path = os.path.join
        if path.exists() and (os.path.getsize(path)>1000):
            ISWRPSTG = pd.read_csv(path).rename(columns = lambda x: x.strip())
            ISWRPSTG = ISWRPSTG.set_index(['TOTIME','SWRDT','KPER','KSTP','KSWR'])

            if kper is not None:
                filt = f"KPER=={kper}"
                print(f'selecting starting heads with {filt}')
                ISWRPSTG = ISWRPSTG.query(filt)

            endh= ISWRPSTG.tail().mean().dropna().to_frame('endheads')
        else:
            class obj:
                def __init__(self):
                    self.shape = [0,0]

            endh = obj()


        if endh.shape[0] == sfr_filt.shape[0] and not use_thalweg:
            print('using ending stage values from previous run')
            endh.insert(0,'reach', value  = np.arange(endh.shape[0]) + 1 )
            # display(endh.head())
            endh.rename(columns = {'reach':'#rno'}).to_csv('RR_2022/inputs/start_stage.tab', sep = '\t', index = False)
            return endh
        else:
            print('using stream thalwegs')
            df = pd.read_csv(os.path.join('RR_2022/inputs/stream_thalwegs.txt'))
            df.loc[:,'thalwegs'] = df.loc[:,'thalweg']+stage_shift_from_mod_top
            print(f'using thalweg + {stage_shift_from_mod_top:.2f}ft for each reach starting stage')

            df.loc[:,['reach','thalweg']].rename(columns = {'reach':'#rno'}).to_csv('RR_2022/inputs/start_stage.tab', sep = '\t', index = False)
            return df.loc[:,['reach','thalweg']].rename(columns = {'reach':'#rno'})

    def plot_swr(sfr_filt):
        fig,ax = plt.subplots()

        ax.plot(sfr_filt.rno, sfr_filt.top)

        roll = sfr_filt.set_index('rno').loc[:,'top'].rolling(20,min_periods = 0).min()
        # display(roll)
        ax.plot(roll.index, roll.values)


    # reach_min = 578
        # bottom, n = 10, reach_min = 578, IROUTETYPE  = 4,
    sfr_filt, reach_loc, rout = find_route(sfr,
                                           bottom,
                                           n = None,
                                           reach_min = reach_min,
                                           IROUTETYPE = IROUTETYPE,
                                           use_average_reach_len=use_average_reach_len)

    print('\n\nstarting processing\n\n')

    write_reach_connect(rout)

    write_reach_loc_dat(reach_loc)

    write_stress_period(sfr_filt)

    write_dataset_6(sfr_filt)

    write_dataset_evap(sfr_filt,evap)
    write_dataset_rain(sfr_filt,rain)
    write_dataset_10(sfr_filt)
    write_dataset_14a(sfr_filt, m)
    plot_swr(sfr_filt)

    print('done with main processing, still need to do geometry processing')
    if set_start_stage:
        endh = write_start_stage(sfr_filt, stage_shift_from_mod_top =.51, use_thalweg = use_thalweg, kper = kper)
        print(f'setting new stages from old run. using kper {kper} as extraction period')
    else:
        print('not setting new starting stages from old run.')
        endh = pd.read_csv('RR_2022/inputs/start_stage.tab', sep = '\t').rename(columns = {'#rno':'reach'})
    ax = endh.set_index('reach').rename(columns = {'endheads':'starting stage'}).plot(title = 'starting stage', marker = '.', figsize = (10,4))
    ax.grid(True); ax.set_ylabel('feet, elevation')
    plt.savefig(os.path.join(out_folder, 'star_stage'))



    print('plotting swr map')
    # # plot swr model cells
    fig,ax = basic.swr_map(m)
    plt.savefig(os.path.join(out_folder, 'sfr_swr_map.png'), dpi = 250)



    xsect_shape = gpd.read_file(r"T:\arich\Russian_River\MirabelWohler_2022\GIS\GEO\RAS_xsections.shp")



    def find_(reach, str):
        x= next(line for line in reach if  line.strip().startswith(str))
        return x

    def read_data(reach):
        lines = []
        for line in reach:
            if 'END' in line.strip():
                break
            else:
                lines.append([float(x.strip()) for x in line.split(',') if (len(x.strip())>0)])
        return lines


    print('reading cross section data')
    # # read and import the cross section data exported from HEC-RAS.
    stream_network = {}
    xsect = {}
    cnt = 1
    with open(r"T:\arich\Russian_River\MirabelWohler_2022\GIS\GEO\RAS.RASexport.sdf", 'r') as reach:
        # next(line for line in reach if  line.startswith("BEGINSTREAMNETWORK:"))
        x = find_(reach,'BEGINSTREAMNETWORK')
        print(x)
        # get next three lines into a list (empty string for nonexistent lines)
        # results = [next(reach, "").rstrip() for line in range(3)]
        # print(results)
        find_(reach,'CENTERLINE:')
        stream_network = read_data(reach)

        find_(reach,'BEGIN CROSS-SECTIONS:')
        while reach:
            try:
                station = find_(reach, 'STATION').strip().split(':')[1]
                find_(reach, 'SURFACE LINE')
                section = read_data(reach)

                xsect[f"REACH_{cnt}"] = section
                cnt = cnt+1
            except:
                break

    print(len(xsect.items()))


    # # convert cross sections to gdf, then find the nearest cross section to sfr_filt centroids


    dfall = pd.DataFrame()
    cnt = 0
    for items in xsect.items():
        df = pd.DataFrame(items[1], columns = ['x','y','z'])
        df.loc[:,'xsect'] = items[0]
        df.loc[:,'dist'] = squareform(pdist(df.loc[:,['x','y']]))[0,:]
        df.loc[:,'NGEOPTS'] = df.shape[0]
        dfall = pd.concat([dfall,df],ignore_index = True)
        cnt=+1

    dfall = gpd.GeoDataFrame(dfall, geometry = gpd.points_from_xy(dfall.loc[:,'x'], dfall.loc[:,'y']), crs = 2226)

    def joinish(sfr_filt, dfall):

        s_ = sfr_filt.copy()

        sfr_filt.loc[:,'gcent'] = sfr_filt.geometry.centroid

        sfr_filt =sfr_filt.set_geometry('gcent')

        sfr_filt = sfr_filt.sjoin_nearest(dfall.loc[:,['geometry','xsect', 'NGEOPTS']], how = 'left')

        sfr_filt = sfr_filt.set_geometry('geometry').drop(columns = 'gcent')

        return sfr_filt

    sfr_filt_with_xsec = joinish(sfr_filt, dfall)


    print('plotting river data xsection locations')
    fig, ax = basic.map_river()
    ax.set_title('Cross Section Locations, From HEC-RAS')
    sfr_filt_with_xsec.plot('xsect',ax = ax, zorder = 100)
    dfall.plot('xsect', ax = ax, markersize = 1)


    plt.savefig(os.path.join(out_folder, 'xsects_loc.png'), dpi = 250)




    def smooth_thalweg(min_elev):
        min_elev = min_elev.loc[:,['THALWEG']].rename(columns = {'THALWEG':'THALWEG_Smoothed'})
        ind = min_elev.index

        for i in range(100):
            min_elev[min_elev.diff().shift(-1)>0] = np.nan
            min_elev = min_elev.dropna()
        min_elev = min_elev.reindex(ind)
        min_elev = min_elev.bfill()

        return min_elev



    def write_dataset_11a_with_geo(sfr_filt,
                                   min_elev = None,
                                   IGEOTYPE = 3,
                                   IGCNDOP = 1,
                                   GMANNING = 0.025,
                                   LEAKANCE = .000001,
                                   getextd = 0.25,
                                   extra = 0.25):
        '''
        write geo section with actual cross section data

        '''
        # IGCNDOP = 0, Fixed conductance is specified for the geometry entry.
        # DATASET 11A - GEOMETRY DATA
    # IGEONUM IGEOTYPE IGCNDOP GMANNING NGEOPTS GWIDTH GBELEV GSSLOPE    GCND      GLK GCNDLN GETEXTD
    #      1    5         1      .25                                     9.2E-04                   0.25

        if min_elev is None:
            print('using observed thalwegs, then smoothing')
            min_elev ={}
            for xxx in sfr_filt.loc[:,['rno','NGEOPTS', 'xsect']].values:
                IGEONUM = xxx[0]
                NGEOPTS = xxx[1]
                xs = xxx[2]
                cur = dfall.query(f"xsect=='{xs}'")
                cur = cur.loc[:,['dist','z']]
                min_elev[IGEONUM] = cur.loc[:,'z'].min()

            min_elev = pd.DataFrame.from_dict(min_elev,orient='index',columns = ['THALWEG'])
            min_elev.index = min_elev.index.set_names('reach')

            mv = smooth_thalweg(min_elev)
            min_elev = min_elev.join(mv)
            # min_elev.loc[:,'THALWEG_Smoothed'] = min_elev.ewm(span = 20).mean()

        with open('RR_2022/inputs/dataset11a_with_geo.txt','w') as r:
            r.write("# DATASET 11A - GEOMETRY DATA\n")
            r.write("# IGEONUM IGEOTYPE IGCNDOP GMANNING NGEOPTS GWIDTH GBELEV GSSLOPE    LEAKANCE      GLK GCNDLN GETEXTD\n")

        with open('RR_2022/inputs/dataset11a_with_geo.txt','a', newline = '') as r:
            for xxx in sfr_filt.loc[:,['rno','NGEOPTS', 'xsect']].values:
                IGEONUM = xxx[0]
                NGEOPTS = xxx[1]
                xs = xxx[2]
                r.write(f"    {IGEONUM}\t{IGEOTYPE}\t{IGCNDOP}\t{GMANNING}\t{NGEOPTS}\t{LEAKANCE} \n")


                cur = dfall.query(f"xsect=='{xs}'")
                cur = cur.loc[:,['dist','z']]
                minz = cur.loc[:,'z'].min() - min_elev.at[IGEONUM, 'THALWEG_Smoothed']
                cur.loc[:,'z'] = cur.loc[:,'z'] -minz

                cur.to_csv(r, sep = '\t',header = False, index = False)


        if sfr_filt.filter(regex = 'THALWEG').columns is not None:
            sfr_filt = sfr_filt.drop(columns = sfr_filt.filter(regex = 'THALWEG').columns)


        sfr_filt = pd.merge(sfr_filt, min_elev, left_on = 'rno', right_index = True )
        sfr_filt = gpd.GeoDataFrame(sfr_filt, geometry = 'geometry', crs = 2226)

        min_elev.to_csv('RR_2022/inputs/stream_thalwegs.txt')
        min_elev.plot()
        # with open('RR_2022/inputs/stream_thalwegs.txt','w') as mint:
        #     mint.write('reach,THALWEG\n')
        #     [mint.write("{}, {}\n".format(item[0], item[1])) for item in min_elev.items()]
        #         # r.write('geo geo geo\n')

        return min_elev, sfr_filt

    # min_elev, sfr_filt_with_xsec =

    print('writing dataset 11a, channel geometry data')
    min_elev, sfr_filt_with_xsec =  write_dataset_11a_with_geo(sfr_filt_with_xsec,
                               IGEOTYPE = IGEOTYPE,
                               IGCNDOP = IGCNDOP,
                               GMANNING =GMANNING,
                               LEAKANCE = LEAKANCE,
                               getextd = getextd,
                               )




    # In[35]:


    # import warnings
    # def write_dataset_11a_with_geo_alt(sfr_filt, min_elev = None,
    #                                    IGEOTYPE = 3, IGCNDOP = 1, GMANNING = 0.025,   LEAKANCE = .000001, getextd = 0.25, extra = 0.25):
    #     '''
    #     write geo with constant cross section data, ie reusing cross sections

    #     '''
    #     # IGCNDOP = 0, Fixed conductance is specified for the geometry entry.
    #     # DATASET 11A - GEOMETRY DATA
    # # IGEONUM IGEOTYPE IGCNDOP GMANNING NGEOPTS GWIDTH GBELEV GSSLOPE    GCND      GLK GCNDLN GETEXTD
    # #      1    5         1      .25                                     9.2E-04                   0.25

    #     if min_elev is None:
    #         print('using observed thalwegs, then smoothing')
    #         min_elev ={}
    #         for xxx in sfr_filt.loc[:,['rno','NGEOPTS', 'xsect']].values:
    #             IGEONUM = xxx[0]
    #             NGEOPTS = xxx[1]
    #             xs = xxx[2]
    #             cur = dfall.query(f"xsect=='{xs}'")
    #             cur = cur.loc[:,['dist','z']]
    #             min_elev[IGEONUM] = cur.loc[:,'z'].min()

    #         min_elev = pd.DataFrame.from_dict(min_elev,orient='index',columns = ['THALWEG'])
    #         min_elev.index = min_elev.index.set_names('reach')
    #         min_elev.loc[:,'THALWEG_Smoothed'] = min_elev.ewm(span = 20).mean()

    #     warnings.warn('\nsimple geo\n'*10)

    #     with open('RR_2022/inputs/dataset11a_with_geo.txt','w') as r:
    #         r.write("# DATASET 11A - GEOMETRY DATA\n")
    #         r.write("# IGEONUM IGEOTYPE IGCNDOP GMANNING NGEOPTS GWIDTH GBELEV GSSLOPE    LEAKANCE      GLK GCNDLN GETEXTD\n")

    #     with open('RR_2022/inputs/dataset11a_with_geo.txt','a', newline = '') as r:
    #         for xxx in sfr_filt.loc[:,['rno','NGEOPTS', 'xsect']].values:
    #             IGEONUM = xxx[0]
    #             xs = 'REACH_418'
    #             NGEOPTS = sfr_filt.drop_duplicates('xsect').query(f"xsect=='{xs}'").NGEOPTS.values[0]


    #             # xs = 1
    #             r.write(f"    {IGEONUM}\t{IGEOTYPE}\t{IGCNDOP}\t{GMANNING}\t{NGEOPTS}\t{LEAKANCE} \n")


    #             cur = dfall.query(f"xsect=='{xs}'")
    #             cur = cur.loc[:,['dist','z']]

    #             minz = cur.loc[:,'z'].min()  - min_elev.at[IGEONUM, 'THALWEG_Smoothed']
    #             # minz = cur.loc[:,'z'].min() - (39-IGEONUM*.1)

    #             cur.loc[:,'z'] = cur.loc[:,'z'] -minz


    #             cur.to_csv(r, sep = '\t',header = False, index = False)

    #     assert not((min_elev.diff().thalweg>=0).any()), 'there are positive changes in elevation from reach to reach'

    #     # with open('RR_2022/inputs/stream_thalwegs.txt','w') as mint:
    #     #     mint.write('reach,thalweg\n')
    #     #     [mint.write("{}, {}\n".format(item[0], item[1])) for item in min_elev.items()]
    #     #         # r.write('geo geo geo\n')

    #     if 'THALWEG' in sfr_filt.columns:
    #         sfr_filt = sfr_filt.drop(columns = 'THALWEG')
    #     sfr_filt = pd.merge(sfr_filt, min_elev, left_on = 'rno', right_index = True )
    #     sfr_filt = gpd.GeoDataFrame(sfr_filt, geometry = 'geometry', crs = 2226)

    #     return min_elev, sfr_filt
    # min_elev, sfr_filt_with_xsec = write_dataset_11a_with_geo_alt(sfr_filt_with_xsec,
    #                            IGEOTYPE = 3,
    #                            IGCNDOP = 0,
    #                            GMANNING = 0.35,
    #                            LEAKANCE = 0.0030,
    #                            getextd = 0.25,
    #                            )

    # min_elev.add(3.0).to_csv(os.path.join('RR_2022', 'inputs', 'start_stage.tab'), sep = '\t',header = None)


    # In[36]:

    print(r'exporting SWR reach data to GIS and T:\arich\Russian_River\MirabelWohler_2022\GIS')
    sfr_filt_with_xsec.to_file('GIS/SWR_Reaches.shp')
    sfr_filt_with_xsec.to_file(r"T:\arich\Russian_River\MirabelWohler_2022\GIS\SWR_reaches.shp")


    print("\n\n\nDone!\n\n\n")







