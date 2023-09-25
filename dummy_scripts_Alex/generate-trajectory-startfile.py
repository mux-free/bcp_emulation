import numpy as np
import xarray as xr
import sys
sys.path.append('/home/ascherrmann/scripts/')
import helper
import pickle
import os

MONTHS = ['JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC']
MONTHSN = np.arange(1,13,1)
sp = '/home/ascherrmann/010-IFS/'

f = open(sp + 'data/All-CYC-entire-year-NEW-correct.txt','rb')
gd = pickle.load(f)
f.close()

Lat = np.round(np.linspace(0,90,226),2)
Lon = np.round(np.linspace(-180,180,901),2)

for k in gd.keys():
  for j in gd[k].keys():
    mid = np.where(gd[k][j]['hzeta']==0)[0][0]
    clat = gd[k][j]['clat'][mid].astype(int)
    clon = gd[k][j]['clon'][mid].astype(int)
    t = gd[k][j]['dates'][mid]
    lab = gd[k][j]['label']

    yyyy = int(t[0:4])
    MM = int(t[4:6])
    DD = int(t[6:8])
    hh = int(t[9:])

    ana_path='/net/thermo/atmosdyn/atroman/phd/'+ k +'/cdf/'

    sfile = ana_path + 'S' + t
    s = xr.open_dataset(sfile, drop_variables=['P','TH','THE','RH','VORT','PVRCONVT','PVRCONVM','PVRTURBT','PVRTURBM','PVRLS','PVRCOND','PVRSW','PVRLWH','PVRLWC','PVRDEP','PVREVC','PVREVR','PVRSUBI','PVRSUBS','PVRMELTI','PVRMELTS','PVRFRZ','PVRRIME','PVRBF'])

    pt = np.array([])
    plat = np.array([])
    plon = np.array([])
    PS = s.PS.values[0,0,clat,clon]
    pv = s.PV.values[0]
    for l in range(len(clat)):
        P = helper.modellevel_to_pressure(PS[l])
        pid = np.where((P>=700) & (P<=975) & (pv[:,clat[l],clon[l]]>=0.75))[0]
        for i in pid:
               pt = np.append(pt,P[i])
               plat = np.append(plat,Lat[clat[l]])
               plon = np.append(plon,Lon[clon[l]])

    save = np.zeros((len(pt),4))
    save[:,1] = plon
    save[:,2] = plat
    save[:,3] = pt
    if lab==0:
        folder='TRO/'
    elif lab==1:
        folder='ETA/'
    else:
        folder='MED/'

#    if folder=='MED/': 
#    if (os.path.isfile(sp + 'traj/' + folder + 'trastart-mature-' + t + '-ID-' + '%06d'%j + '.txt')):
#        continue
#    else:
    if True:
        np.savetxt(sp + 'ctraj/' + folder + 'trastart-mature-' + t + '-ID-' + '%06d'%j + '.txt',save,fmt='%f', delimiter=' ', newline='\n')
#        for ul in ['ETA/']:
#            if (os.path.isfile(sp + 'traj/' + ul + 'trastart-mature-' + t + '-ID-' + '%06d'%j + '.txt')):
#                os.remove(sp + 'traj/' + ul + 'trastart-mature-' + t + '-ID-' + '%06d'%j + '.txt')
#                os.remove(sp + 'traj/' + ul + 'raw/trajectories-mature-'+ t + '-ID-' + '%06d'%j + '.txt')
#                os.remove(sp + 'traj/' + ul + 'use/trajectories-mature-'+ t + '-ID-' + '%06d'%j + '.txt')
#                if os.path.isfile(sp + 'traj/' + ul + 'traend-' + t + '-ID-' + '%06d'%j + '.txt'):
#                    os.remove(sp + 'traj/' + ul + 'traend-' + t + '-ID-' + '%06d'%j + '.txt')





