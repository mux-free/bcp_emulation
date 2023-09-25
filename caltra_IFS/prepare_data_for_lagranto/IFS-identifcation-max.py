from datetime import datetime, date, timedelta
import numpy as np
import xarray as xr
import sys
sys.path.append('/home/ascherrmann/scripts/')
import helper
import argparse
import pickle
import os


labels = np.array([])
Gdates = np.array([])
data = dict()



# MONTHS = np.array(['DEC17','JAN18','FEB18','MAR18','APR18','MAY18','JUN18','JUL18','AUG18','SEP18','OCT18','NOV18'])
## These two are missing
# MONTHS = np.array(['MAR18','APR18'])

MONTHS = np.array(['DEC17','JAN18','FEB18','MAR18','APR18','MAY18','JUN18','JUL18','AUG18','SEP18','OCT18','NOV18'])

MONTHN = np.append(12,np.arange(1,12))


# # for at least 48 h backward trajectories require mature stage to be after the following dates
# mx = 3

months=MONTHS

LON = np.round(np.linspace(-180,180,901),1)
LAT = np.round(np.linspace(0,90,226),1)

ft = date.toordinal(date(1950,1,1))


datastruct = dict()
labels = np.array([])
Gdates = np.array([])
data = dict()




###  restrict myself to Atlantic, tropical and Mediterranean ones
for Month in months:
    datastruct = dict()
    datastruct[Month] = dict()
              
    print(f'\nSTART IDENTIFICATION OF MONTH: {Month}\n------------------------------\n')


    ## LOAD DATA
    ################################# MODIFIED BY MAX ############################################################################
    # - Load my Data and my validation IDs 
    path_trackCYC = f'/net/helium/atmosdyn/IFS-1Y/{Month}/features/tracking/TRACKED_CYCLONES'
    d = np.loadtxt(path_trackCYC,skiprows=1)
    IDs = np.unique(d[:,1])
    print(f'IDs from the TRACKED_CYCLONE file:\n{IDs}')

    ## Load validation cyclone IDs
    path_CYCsplit_info = f'/net/helium/atmosdyn/freimax/data_msc/IFS-18/cyclones/data_random_forest/{Month}/cyclone_split_info.txt'
    with open(path_CYCsplit_info, 'r') as f:
        lines = f.readlines()
    num_cyclones = int(lines[0].split(': ')[1])  # The 'Number of cyclones' line
    id_list_str = lines[1].split(': ')[1].strip('[]\n')
    id_list_cyc = [float(x) for x in id_list_str.split(', ')] if id_list_str else []  # The 'ID list' line
    id_val_str = lines[2].split(': ')[1].strip('[]\n')
    IDs_valCYC = [float(x) for x in id_val_str.split(', ')] if id_val_str else []  # The 'ID of cyclones in validation set' line
    print(f'IDs of the validation cyclones:\n{Month}\t{IDs_valCYC}\n')
    ################################################################################################################################



    # for ids in IDs[:]:            # ALEX
    for ids in IDs_valCYC:          # MAX

        print('Compute on CYC_id: ', ids)

        ## cappear contains all indices of rows in d that have ids (one of the validation IDs)        
        cappear, = np.where((d[:,1]==ids))


        clat = []
        clon = []
        bergeron = []
        
        hourszeta = np.array([])
        hourstoSLPmin = np.array([])
        zetal = np.array([])
        dates = np.array([])
        zeta = np.zeros(len(cappear))
        hours = np.zeros(len(cappear))
        
        
        hoursslp = np.array([])         # MAX
        slpl = np.array([])             # MAX
        slp = np.zeros(len(cappear))    # MAX


        ### find id of minimum SLP in track file
        slpminid = cappear[np.where(d[cappear,6] == np.min(d[cappear,6]))]
        slpminid = slpminid[-1]
        hourstoSLPmin = np.append(hourstoSLPmin, d[cappear,0]-d[slpminid,0])


        ## FROM MAX: Here the cappear file column 4 (5th column) is not correct
        # ### find id of minimum SLP in track file
        # slpminid = cappear[np.where(d[cappear,4] == np.min(d[cappear,4]))]      # NOTE: d[cappear,4] IS EQUAL TO d[:,4]   --> values of 5th column = "inpres"
        # slpminid = slpminid[-1]                                                 # Extract number from array
        # hourstoSLPmin = np.append(hourstoSLPmin, d[cappear,0]-d[slpminid,0])    # Substract time from min SLP from other times to get seconds to/from SLP minimum



        ### step through hrs of the cyclon in trackfile
        for wr,u in enumerate(cappear):
            tmp = d[u]
            k = str(helper.datenum_to_datetime(ft+tmp[0]/24))
            Date = k[0:4]+k[5:7]+k[8:10]+'_'+k[11:13]


            ### if that date has not been loaded yet, load it
            # if Gdates.size == 0 or (~(np.any(Gdates==Date))):     # MAX
            if (~(np.any(Gdates==Date))):                           # ALEX
                # Load dataset with PS, SLP and Vorticity
                dataset_1 = xr.open_dataset(f'/net/helium/atmosdyn/IFS-1Y/{Month}/cdf/S{Date}', drop_variables=['PV','P','TH','THE','RH','PVRCONVT','PVRCONVM','PVRTURBT','PVRTURBM','PVRLS','PVRCOND','PVRSW','PVRLWH','PVRLWC','PVRDEP','PVREVC','PVREVR','PVRSUBI','PVRSUBS','PVRMELTI','PVRMELTS','PVRFRZ','PVRRIME','PVRBF'])
                data[Date] = dataset_1
            else:
                 print('\nIMPRTANT SECTION WAS LEFT OUT\n')
            
            dates = np.append(dates,Date)
            Gdates = np.append(Gdates,Date)
                
                ####################################################



            #hours since first identification
            hours[wr] = tmp[0]
            ### keep adding lon to lon and lat to lat and use srtid, as it is correct with normal crosssections
            CLONIDS, CLATIDS = helper.IFS_radial_ids_correct(200,LAT[np.where(LAT==np.round(tmp[3],1))[0][0]])
            addlon = CLONIDS + np.where(LON==np.round(tmp[2],1))[0][0]
            addlon[np.where((addlon-900)>0)] = addlon[np.where((addlon-900)>0)]-900


            bergeron.append(tmp[10])


            clat.append(CLATIDS.astype(int) + np.where(LAT==np.round(tmp[3],1))[0][0])
            clon.append(addlon.astype(int))
#                 clat = np.vstack((clat, CLATIDS + np.where(LAT==np.round(tmp[3],1))[0][0]))
#                 clon = np.vstack((clon, addlon))

            ### use last lon and lat entries [-1] of center
            PS = data[Date].PS.values[0,0,clat[-1],clon[-1]]
            for e in range(len(CLATIDS)):
                P = helper.modellevel_to_pressure(PS[e])
                I = np.where(abs(P-850)==np.min(abs(P-850)))[0]
                I = I[0].astype(int)
                zetal = np.append(zetal,data[Date].VORT.values[0,I,clat[-1][e],clon[-1][e]])

            zeta[wr] = np.mean(zetal)

        hours = hours - hours[np.where(zeta==np.max(zeta))]
        hourszeta = np.append(hourszeta,hours)

        # allow for at least 48h backward trajectories
        if (((int(dates[np.where(zeta==np.max(zeta))[0][-1]][6:8])>=3)) | (int(dates[np.where(zeta==np.max(zeta))[0][-1]][4:6])!=MONTHN[np.where(MONTHS==Month)[0][0]])):

            ### added 30.10.2020 to ensure only MED cyclones
            ### Add 1 as clat, clon start with zeros as base to stack them

            MatureLat = np.round(np.mean(LAT[clat[np.where(zeta==np.max(zeta))[0][-1]]]),1)
            MatureLon = np.round(np.mean(LON[clon[np.where(zeta==np.max(zeta))[0][-1]]]),1)
            # use latest maximum relative vorticity
    #              MatureLat = np.round(np.mean(LAT[clat[np.where(zeta==np.max(zeta))[0][-1]+1]]),1)
    #              MatureLon = np.round(np.mean(LON[clon[np.where(zeta==np.max(zeta))[0][-1]+1]]),1)
        
        #if time to mature stage is too short remove from data
        if((hours[0]<=(-0)) & (hours[-1]>=0)):      # MAX
        # if((hours[0]<=(-2)) & (hours[-1]>=2)):      # ALEX
            datastruct[Month][ids] = dict()
            datastruct[Month][ids]['clat'] = clat
            datastruct[Month][ids]['clon'] = clon
            datastruct[Month][ids]['zeta'] = zeta
            datastruct[Month][ids]['hzeta'] = hourszeta
            datastruct[Month][ids]['SLP'] = d[cappear,6]
            datastruct[Month][ids]['hSLP'] = hourstoSLPmin
            datastruct[Month][ids]['dates'] = dates
            datastruct[Month][ids]['Matlat'] = MatureLat
            datastruct[Month][ids]['Matlon'] = MatureLon
            # datastruct[Month][ids]['bergeron'] = bergeron

            # ### these are preidentified tropical
            # if(d[cappear[0],-3]==0):
            #         label=0
            # ### MED
            # else:
            #     for b,o in enumerate(LONM[:-1]):
            #         if( ((30-MatureLat)<=0) & ((LATM[b]-MatureLat)>=0) & ((o-MatureLon)<=0) & ((LONM[b+1]-MatureLon)>=0)):
            #                 label=2
            # ### ATL
            #         else:
            #             label=1
            datastruct[Month][ids]['label'] = ids



    # PATH = '/net/helium/atmosdyn/freimax/data_msc/IFS-18/IFS-traj/CYC_validation/'
    PATH = '/net/helium/atmosdyn/freimax/data_msc/IFS-17/trajectories/caltra'
    add = 'CYC-casestudy-IFS17-' + Month

    # Create the directory if it does not exist
    os.makedirs(PATH + '/' + Month, exist_ok=True)

    f = open(PATH + '/' + Month + '/' + add + '.txt','wb')
    pickle.dump(datastruct,f)
    f.close()