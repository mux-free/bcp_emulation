from datetime import datetime, date, timedelta
import numpy as np
import xarray as xr
import sys
sys.path.append('/home/ascherrmann/scripts/')
import helper
import argparse
import pickle

parser = argparse.ArgumentParser(description="produce txt files with identification:")
parser.add_argument('months',default='',type=str,help='month combination like DJF')
dbase = '/atmosdyn2/ascherrmann/'
args = parser.parse_args()
MOT = str(args.months)
PATH = dbase + '010-IFS/data/'

MONTHS = np.array(['DEC17','JAN18','FEB18','MAR18','APR18','MAY18','JUN18','JUL18','AUG18','SEP18','OCT18','NOV18'])
MONTHN = np.append(12,np.arange(1,12))
# for at least 48 h backward trajectories require mature stage to be after the following dates
mx = 3

months=[MOT]

LON = np.round(np.linspace(-180,180,901),1)
LAT = np.round(np.linspace(0,90,226),1)

LATM = np.array([30,42,30,48])
LONM = np.array([-5,2,42])

#LONB = np.array([-105,-75,40,90])
LONB = np.array([-95,-75,-5,15])
LATB = np.array([15,83,0,83,50,83])

LATSS = [LATM, LATB]
LONSS = [LONM,LONB]

ft = date.toordinal(date(1950,1,1))

add = 'All-CYC-entire-year-NEW' + MOT + '-correct'
#r200_lonids, r200_latids = helper.radial_ids_around_center(200)

datastruct = dict()
#llat = len(r200_lonids)

labels = np.array([])
Gdates = np.array([])
data = dict()
###  restrict myself to Atlantic, tropical and Mediterranean ones

for Month in months:
    datastruct[Month] = dict()

    if Month == 'DEC17':
        track = dbase + '002-2020-08-05-Identify-DJF1718-medcyclones/' + Month  + '/TRACKED_CYCLONES'
        d = np.loadtxt(track)
    else:
        track = '/home/atroman/phd/' + Month + '/features/tracking/TRACKED_CYCLONES'
        d = np.loadtxt(track,skiprows=1)

    IDs = np.unique(d[:,1])
    for ids in IDs[:]:
       cappear, = np.where((d[:,1]==ids))
       indomainalready = 0
       for lonb, latb in zip(LONSS,LATSS):
        if indomainalready==1:
            continue
        #first & last & mid track point in boundary
        
        if((d[cappear[0],2]>lonb[0]) & (d[cappear[0],2]<lonb[-1])):

         if ((d[cappear[-1],2]>lonb[0]) & (d[cappear[-1],2]<lonb[-1])):

          if ((d[cappear[int(len(cappear)/2)],2]>lonb[0]) & (d[cappear[int(len(cappear)/2)],2]<lonb[-1])):
           CONTINUE=0

           # check lat boundary
           for k,l in enumerate(lonb[:-1]):
                if ((d[cappear[0],2]>l) & (d[cappear[0],2]<lonb[k+1]) & (d[cappear[0],3]>latb[2*k]) &(d[cappear[0],3]<latb[2*k+1])):
                    CONTINUE=1
                    indomainalready=1
           if (CONTINUE==1):
            #all local data for particular cyclones
#            clat = np.zeros((len(r200_lonids)),dtype=int)
#            clon = np.zeros((len(r200_lonids)),dtype=int)
            clat = []
            clon = []
            hourszeta = np.array([])
            hourstoSLPmin = np.array([])
            zetal = np.array([])
            dates = np.array([])
            zeta = np.zeros(len(cappear))
            hours = np.zeros(len(cappear))

            ### find id of minimum SLP in track file
            slpminid = cappear[np.where(d[cappear,6] == np.min(d[cappear,6]))]
            slpminid = slpminid[-1]
            hourstoSLPmin = np.append(hourstoSLPmin, d[cappear,0]-d[slpminid,0])

            ### step through hrs of the cyclon in trackfile
            for wr,u in enumerate(cappear):
                tmp = d[u]
                k = str(helper.datenum_to_datetime(ft+tmp[0]/24))
                Date = k[0:4]+k[5:7]+k[8:10]+'_'+k[11:13]

                ### if that date has not been loaded yet, load it
                if (~(np.any(Gdates==Date))):
                    data[Date] = xr.open_dataset('/net/thermo/atmosdyn/atroman/phd/'+Month+'/cdf/S'+Date, drop_variables=['PV','P','TH','THE','RH','PVRCONVT','PVRCONVM','PVRTURBT','PVRTURBM','PVRLS','PVRCOND','PVRSW','PVRLWH','PVRLWC','PVRDEP','PVREVC','PVREVR','PVRSUBI','PVRSUBS','PVRMELTI','PVRMELTS','PVRFRZ','PVRRIME','PVRBF'])

                dates = np.append(dates,Date)
                Gdates = np.append(Gdates,Date)

                #hours since first identification
                hours[wr] = tmp[0]
                
                
                ### keep adding lon to lon and lat to lat and use srtid, as it is correct with normal crosssections
                CLONIDS, CLATIDS = helper.IFS_radial_ids_correct(200,LAT[np.where(LAT==np.round(tmp[3],1))[0][0]])
                addlon = CLONIDS + np.where(LON==np.round(tmp[2],1))[0][0]
                addlon[np.where((addlon-900)>0)] = addlon[np.where((addlon-900)>0)]-900

                clat.append(CLATIDS.astype(int) + np.where(LAT==np.round(tmp[3],1))[0][0])
                clon.append(addlon.astype(int))
#                clat = np.vstack((clat, CLATIDS + np.where(LAT==np.round(tmp[3],1))[0][0]))
#                clon = np.vstack((clon, addlon))

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
            if (((int(dates[np.where(zeta==np.max(zeta))[0][-1]][6:8])>=3)) | (int(dates[np.where(zeta==np.max(zeta))[0][-1]][4:6])!=MONTHN[np.where(MONTHS==MOT)[0][0]])):

            ### added 30.10.2020 to ensure only MED cyclones
            ### add 1 as clat, clon start with zeros as base to stack them

             MatureLat = np.round(np.mean(LAT[clat[np.where(zeta==np.max(zeta))[0][-1]]]),1)
             MatureLon = np.round(np.mean(LON[clon[np.where(zeta==np.max(zeta))[0][-1]]]),1)
            # use latest maximum relative vorticity
#             MatureLat = np.round(np.mean(LAT[clat[np.where(zeta==np.max(zeta))[0][-1]+1]]),1)
#             MatureLon = np.round(np.mean(LON[clon[np.where(zeta==np.max(zeta))[0][-1]+1]]),1)
 
             #if time to mature stage is too short remove from data
             if((hours[0]<=(-2)) & (hours[-1]>=2)):
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

                ### these are preidentified tropical
                if(d[cappear[0],-3]==0):
                     label=0
                ### MED
                else:
                    for b,o in enumerate(LONM[:-1]):
                        if( ((30-MatureLat)<=0) & ((LATM[b]-MatureLat)>=0) &
                   ((o-MatureLon)<=0) & ((LONM[b+1]-MatureLon)>=0)):
                             label=2
                ### ATL
                        else:
                         label=1
                datastruct[Month][ids]['label'] = label

f = open(PATH + MOT + '/' + add + '.txt','wb')
pickle.dump(datastruct,f)
f.close()

