import numpy as np
import os
import sys
sys.path.append('/home/ascherrmann/scripts/')
sys.path.append('/home/raphaelp/phd/scripts/basics/')
from useful_functions import get_field_at_level,resize_colorbar_horz,resize_colorbar_vert
import helper
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.collections as mcoll
from matplotlib.colors import ListedColormap, BoundaryNorm
import pickle
import argparse
import xarray as xr
import cartopy
import matplotlib.gridspec as gridspec
import functools
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

def colbar(cmap,minval,maxval,nlevels):
    maplist = [cmap(i) for i in range(cmap.N)]
    newmap = ListedColormap(maplist)
    norm = BoundaryNorm(pvr_levels,cmap.N)
    return newmap, norm

parser = argparse.ArgumentParser(description="plot accumulated average PV gain that is associated with the cyclone and the environment")
parser.add_argument('rdis',default='',type=int,help='distance from center in km that should be considered as cyclonic')
parser.add_argument('type',default='',type=str,help='MED, TRO or ETA')

parser.add_argument('deltaPSP',default='',type=int,help='difference between surface pressure and pressure that should be evaluated as orographical influence')

parser.add_argument('ZBB',default='',type=int,help='evelation in m at which PV changes should be evaluated as orographic')

args = parser.parse_args()
CT = str(args.type)
rdis = int(args.rdis)
deltaPSP = int(args.deltaPSP)
zbb = int(args.ZBB)

pload = '/atmosdyn2/ascherrmann/010-IFS/ctraj/' + CT + '/use/' 
plload = '/atmosdyn2/ascherrmann/010-IFS/ctraj/' + CT + '/'

traj = np.array([])
for d in os.listdir(pload):
    if(d.startswith('trajectories-mature-')):
            traj = np.append(traj,d)           

MON = np.array([])
for d in os.listdir(plload):
    if(d.startswith('traend-')):
        MON = np.append(MON,d)

MON = np.sort(MON)
traj = np.sort(traj)

fsl=10

labs = helper.traced_vars_IFS()

cl=['orange','green','dodgerblue','blue','red','k']
pllegend = ['APV$_{\mathrm{TOT,cyc}}$','APV$_{\mathrm{TOT,env}}$','APV$_{\mathrm{CONVT}}$','APV$_{\mathrm{TURBT}}$', 'APV$_{\mathrm{CONVM}}$', 'APV$_{\mathrm{TURBM}}$','APV$_{\mathrm{RAD}}$','APV$_{\mathrm{LS}}$']


plotvars = ['PVR-T','PVRCONVM','PVRTURBM','APVRAD','PVRLS','APVTOT']
linestyle = ['-',':']

LON=np.linspace(-180,180,901)
LAT=np.linspace(0,90,226)
#deltaLONLAT = helper.convert_radial_distance_to_lon_lat_dis(rdis)

f = open('/atmosdyn2/ascherrmann/010-IFS/data/All-CYC-entire-year-NEW-correct.txt','rb')
td = pickle.load(f)
f.close()

### INFO
### Trajetories start 200km around the center between 975 and 700 hPa
###

wql = 0
meandi = dict()
env = 'env'
cyc = 'cyc'
oro = 'oro'
ORO = dict()
split=[cyc,env]

datadi = dict() ####raw data of traced vars
dipv = dict() ####splited pv is stored here
dit = dict() ### 1 and 0s, 1 if traj is close to center at given time, 0 if not
meandi[env] = dict()
meandi[cyc] = dict()

H = 48
xlim = np.array([-1*H,0])
#total
ylim = np.array([-0.3,1.25])

pressure_stack = np.zeros(H+1)

pvsum = np.where(labs=='PVRCONVT')[0][0]

hoursegments = np.flip(np.arange(-48,1,1))
linewidth=1.5
alpha=1.
cmap = ListedColormap(['saddlebrown','orange'])
norm = BoundaryNorm([0, 0.5, 1], cmap.N)

monsave = np.array([])
idsave = np.array([])
datesave = np.array([])
highORO = np.array([])
fig = plt.figure(figsize=(16,12))
gs = gridspec.GridSpec(ncols=2, nrows=2)

index = 0
for uyt, txt in enumerate(traj):
    montmp = MON[uyt][-9:-4]
    monsave = np.append(monsave,montmp)

    idtmp = int(txt[-10:-4])
    idsave = np.append(idsave,idtmp)


    date=txt[-25:-14]
    if date!='20171214_02' and date!='20180619_03':
        continue
    date=date+'-%03d'%idtmp
    datesave=np.append(datesave,date)

    datadi[date]=dict() #raw data

    dipv[date]=dict()    #accumulated pv is saved here
    dipv[date][env]=dict()
    dipv[date][cyc]=dict()

    ORO[date] = dict()
    ORO[date][env]=dict()
    ORO[date][cyc]=dict()

    htzeta = td[montmp][idtmp]['hzeta']
    zeroid = np.where(htzeta==0)[0][0]
    htzeta = htzeta[:zeroid+1]
    clat = td[montmp][idtmp]['clat'][:zeroid+1]
    clon = td[montmp][idtmp]['clon'][:zeroid+1]
    

    tt = np.loadtxt(pload + txt)
    for k, el in enumerate(labs):
        datadi[date][el] = tt[:,k].reshape(-1,H+1)

    dp = datadi[date]['PS']-datadi[date]['P']
    OL = datadi[date]['OL']
    ZB = datadi[date]['ZB']

    tmpclon= np.array([])
    tmpclat= np.array([])

    ### follow cyclone backwards to find its center
    dit[date] = dict()
    dit[date][env] = np.zeros(datadi[date]['time'].shape)
    dit[date][cyc] = np.zeros(datadi[date]['time'].shape)
    dit[date][oro] = np.zeros(datadi[date]['time'].shape)

    for k in range(0,H+1):
        if(np.where(htzeta==(-k))[0].size):
            tmpq = np.where(htzeta==(-k))[0][0]
            tmpclon = np.append(tmpclon,np.mean(clon[tmpq]))
            tmpclat = np.append(tmpclat,np.mean(clat[tmpq]))
        else:
            ### use boundary position that no traj should be near it
            tmpclon = np.append(tmpclon,860)
            tmpclat = np.append(tmpclat,0)

    deltaPV = np.zeros(datadi[date]['PV'].shape)
    deltaPV[:,1:] = datadi[date]['PV'][:,:-1]-datadi[date]['PV'][:,1:]

    ### check every hours every trajecttory whether it is close to the center ###
    for e, h in enumerate(datadi[date]['time'][0,:]):

        tmplon = tmpclon[e].astype(int)
        tmplat = tmpclat[e].astype(int)

        ### center lon and latitude
        CLON = LON[tmplon]
        CLAT = LAT[tmplat]

        ### 30.10.2020 radial distance instead of square
        ### if traj is in circle of 200km set cyc entry to 0, else env entry 1
        for tr in range(len(datadi[date]['time'])):
            if (helper.convert_dlon_dlat_to_radial_dis_new(CLON-datadi[date]['lon'][tr,e],CLAT-datadi[date]['lat'][tr,e],CLAT)<=rdis):
#            if ( np.sqrt( (CLON-datadi[date]['lon'][tr,e])**2 + (CLAT-datadi[date]['lat'][tr,e])**2) <=  deltaLONLAT):
            ###
                dit[date][cyc][tr,e]=1
            else:
                dit[date][env][tr,e]=1

            ### check for orography
            if ((OL[tr,e]>0.7) & (ZB[tr,e]>zbb) & (dp[tr,e]<deltaPSP)):
                dit[date][oro][tr,e] = 1

    ttmp = dict()
    ttmp[oro] = (dit[date][oro][:,:-1]+dit[date][oro][:,1:])/2.
    for key in split:
        ttmp[key] = (dit[date][key][:,:-1]+dit[date][key][:,1:])/2.
        for k, el in enumerate(labs[pvsum:]):
            dipv[date][key][el] = np.zeros(datadi[date]['time'].shape)
            dipv[date][key][el][:,:-1] = np.flip(np.cumsum(np.flip((datadi[date][el][:,1:] + datadi[date][el][:,:-1])/2.*ttmp[key][:,:],axis=1),axis=1),axis=1)
#            dipv[date][key][el][:,:-1] = np.flip(np.cumsum(np.flip((datadi[date][el][:,1:] + datadi[date][el][:,:-1])/2.*dit[date][key][:,1:],axis=1),axis=1),axis=1)

            ORO[date][key][el] = np.zeros(datadi[date]['time'].shape)
            ORO[date][key][el][:,:-1] = np.flip(np.cumsum(np.flip((datadi[date][el][:,1:] + datadi[date][el][:,:-1])/2.*ttmp[key][:,:]*ttmp[oro][:,:],axis=1),axis=1),axis=1)
#            ORO[date][key][el][:,:-1] = np.flip(np.cumsum(np.flip((datadi[date][el][:,1:] + datadi[date][el][:,:-1])/2.*dit[date][key][:,1:] * dit[date][oro][:,1:],axis=1),axis=1),axis=1)
        
        dipv[date][key]['deltaPV'] = np.zeros(datadi[date]['time'].shape)
        dipv[date][key]['deltaPV'][:,:-1] = np.flip(np.cumsum(np.flip(deltaPV[:,1:] * ttmp[key][:,:],axis=1),axis=1),axis=1)
#        dipv[date][key]['deltaPV'][:,:-1] = np.flip(np.cumsum(np.flip(deltaPV[:,1:] * dit[date][key][:,1:],axis=1),axis=1),axis=1)

        dipv[date][key]['APVTOT'] =np.zeros(datadi[date]['time'].shape)
        ORO[date][key]['APVTOT'] = np.zeros(datadi[date]['time'].shape)
        ORO[date][key]['deltaPV'] = np.zeros(datadi[date]['time'].shape)
        ORO[date][key]['deltaPV'][:,:-1] = np.flip(np.cumsum(np.flip(deltaPV[:,1:] * ttmp[key][:,:] * ttmp[oro][:,:],axis=1),axis=1),axis=1)
        for el in labs[pvsum:]:
            dipv[date][key]['APVTOT'] += dipv[date][key][el]
            ORO[date][key]['APVTOT'] += ORO[date][key][el]

        dipv[date][key]['APVRAD'] = dipv[date][key]['PVRSW'] + dipv[date][key]['PVRLWH'] + dipv[date][key]['PVRLWC']
        dipv[date][key]['PVR-T'] = dipv[date][key]['PVRTURBT'] + dipv[date][key]['PVRCONVT']

        ORO[date][key]['APVRAD'] = ORO[date][key]['PVRSW'] + ORO[date][key]['PVRLWH'] + ORO[date][key]['PVRLWC']
        ORO[date][key]['PVR-T'] = ORO[date][key]['PVRTURBT'] + ORO[date][key]['PVRCONVT']

#        for el in np.append(labs[pvsum:],['APVTOT','APVRAD','PVR-T']):
#            if wql==0:
#                meandi[key][el] = dipv[date][key][el]
#            else:
#                meandi[key][el] = np.concatenate((meandi[key][el], dipv[date][key][el]),axis=0)


    ### PLOTTING
    #select data
    idp = np.where(datadi[date]['PV'][:,0]>=0.75)[0]
    if (len(idp)!=0):

     calpre = np.sum(ORO[date][env]['APVTOT'][idp,:],axis=0)/len(idp)
#     if False:
#        continue
     if date=='20171214_02-073' or date=='20180619_03-111':
         
#     if (calpre[0]!=0):
#      if ((calpre[0]>0.3) & ((calpre[12]/calpre[0]) > 0.75)):
        highORO = np.append(highORO,date)
    
        t = datadi[date]['time'][0]
#        titles= 'PV-contribution'
        ax = fig.add_subplot(gs[1,index])
        for q in range(0,1):
                ax.plot([],[],marker=None,ls='-',color='black')
                ax.plot([],[],marker=None,ls=':',color='black')
                ax.plot([],[],color='orange',ls='-')
                ax.plot([],[],color='saddlebrown',ls='-')
                ax.plot([],[],color='green',ls='-')
                ax.plot([],[],color='dodgerblue',ls='-')
                ax.plot([],[],color='blue',ls='-')
                ax.plot([],[],color='red',ls='-')
                ### this is for all traj, also possible to weight the mean by the traj number in that 
                ### pressure regime

                sq=np.sum(dit[date][cyc][idp,:] + dit[date][env][idp,:],axis=0)
                if(q==0):
                    tmpdata = dipv[date]
                else:
                    tmpdata = ORO[date]

                for pl,key in enumerate(split):
                  for wt, ru in enumerate(plotvars):
                    meantmp = np.array([])
                    stdtmp = np.array([])
                    if ru =='PVR-T':
                        segmentval = np.array([])
                    for xx in range(len(sq)):
                        if sq[xx]>0:
                            meantmp = np.append(meantmp,np.sum(tmpdata[key][ru][idp,xx])/sq[xx])
                            if ru=='PVR-T':
                                if (abs(np.sum(tmpdata[key]['PVRCONVT'][idp,xx]))>=(abs(np.sum(tmpdata[key]['PVRTURBT'][idp,xx])))):
                                    segmentval = np.append(segmentval,1)
                                else:
                                    segmentval = np.append(segmentval,0)
                        else:
                            meantmp = np.append(meantmp,0)
                    if ru =='PVR-T':
                        segments = helper.make_segments(hoursegments,meantmp)
                        lc = mcoll.LineCollection(segments, array=segmentval, cmap=cmap, norm=norm, linestyle=linestyle[pl],linewidth=linewidth, alpha=alpha)
                        ax.add_collection(lc)
                    else:
                        ax.plot(t,meantmp,color=cl[wt],label=pllegend[wt],ls=linestyle[pl])

                ax.axvline(htzeta[0],color='grey',ls='-')
                ax.set_xlim(xlim)
                ax.set_ylim(ylim)
                ax.set_xticks(ticks=np.arange(-48,1,6))
                ax.set_yticks(ticks=np.arange(-0.25,1.26,0.25))
                ax.tick_params(labelright=False,right=True)       

        ax.set_ylabel('acc. PV [PVU]')

        ax.legend(pllegend,fontsize=fsl,loc='upper left')
        ax.set_xlabel('time until mature stage [h]')
        if date=='20171214_02-073':
            te = '(c)'
        else:
            te = '(d)'
        ax.text(-0.11, 0.95, te,fontsize=14, va='top',transform=ax.transAxes)
    index+=1


CT = 'MED'

pload = '/atmosdyn2/ascherrmann/010-IFS/ctraj/' + CT + '/use/'

f = open(pload + 'PV-data-'+CT+'dPSP-100-ZB-800PVedge-0.3-400-correct-distance.txt','rb')
data = pickle.load(f)
f.close()

f = open('/atmosdyn2/ascherrmann/010-IFS/data/All-CYC-entire-year-NEW-correct.txt','rb')
ldata = pickle.load(f)
f.close()

NORO = xr.open_dataset('/atmosdyn2/ascherrmann/010-IFS/data/IFSORO')
ZB = NORO['ZB'].values[0,0]
oro = data['oro']
datadi = data['rawdata']
dipv = data['dipv']

rdis = 400
labs = helper.traced_vars_IFS()
H = 48
a = 1

maxv = 0.61
minv =-0.6
pvr_levels = np.arange(minv,maxv,0.15)

ap = plt.cm.seismic
cmap ,norm = colbar(ap,minv,maxv,len(pvr_levels))
for k in range(75,150):
    cmap.colors[k] = np.array([189/256, 195/256, 199/256, 1.0])

LON = np.arange(-180,180.1,0.4)
LAT = np.arange(0,90.1,0.4)

alpha=1.
linewidth=1
ticklabels=pvr_levels
text = ['(a)','(b)']
pvsum = np.where(labs=='PVRCONVT')[0][0]
index = 0
for q,date in enumerate(oro.keys()):
 if(date=='20171214_02-073') or date=='20180619_03-111':
  mon = data['mons'][q]
  CYID = int(date[-3:])
  idp = np.where(datadi[date]['PV'][:,0]>=0.75)[0]
#  if(np.mean(datadi[date]['OL'][idp,0])<0.9):
  if True:

    tralon = datadi[date]['lon'][idp,:]
    tralat = datadi[date]['lat'][idp,:]

    PVoro = oro[date]['env']['APVTOT'][idp,:]
    deltaPVoro = np.zeros(datadi[date]['time'][idp,:].shape)
    deltaPVoro[:,1:] = PVoro[:,:-1]-PVoro[:,1:]

    pvr = deltaPVoro
    pvr = np.zeros(datadi[date]['time'][idp,:].shape)

    for k in labs[pvsum:]:
        pvr += datadi[date][k][idp,:]
#    pvr[:,1:] = (dipv[date]['cyc']['APVTOT'][idp,:-1] + dipv[date]['env']['APVTOT'][idp,:-1] -
#            dipv[date]['cyc']['APVTOT'][idp,1:] + dipv[date]['env']['APVTOT'][idp,1:])

    tracklo = np.array([])
    trackla = np.array([])
    for u in range(len(ldata[mon][CYID]['clon'])):
        tracklo = np.append(tracklo,np.mean(LON[ldata[mon][CYID]['clon'][u].astype(int)]))
        trackla = np.append(trackla,np.mean(LAT[ldata[mon][CYID]['clat'][u].astype(int)]))
        if ldata[mon][CYID]['dates'][u]==date[:-4]:
            latc = np.mean(LAT[ldata[mon][CYID]['clat'][u].astype(int)])
            lonc = np.mean(LON[ldata[mon][CYID]['clon'][u].astype(int)])
            matureid = u

#    latc = np.mean(np.unique(tralat[:,0]))
#    lonc = np.mean(np.unique(tralon[:,0]))

    if date=='20171214_02-073':
        minpltlatc = np.round(latc-np.floor(helper.convert_radial_distance_to_lon_lat_dis(1400)),0)
        minpltlonc = np.round(lonc-np.floor(helper.convert_radial_distance_to_lon_lat_dis(2500)),0)

        maxpltlatc = np.round(latc+np.round(helper.convert_radial_distance_to_lon_lat_dis(2500),0),0)
        maxpltlonc = np.round(lonc+np.round(helper.convert_radial_distance_to_lon_lat_dis(1500),0),0)
    else:
        minpltlatc = np.round(latc-np.floor(helper.convert_radial_distance_to_lon_lat_dis(1600)),0)
        minpltlonc = np.round(lonc-np.floor(helper.convert_radial_distance_to_lon_lat_dis(2200)),0)

        maxpltlatc = np.round(latc+np.round(helper.convert_radial_distance_to_lon_lat_dis(1600),0),0)
        maxpltlonc = np.round(lonc+np.round(helper.convert_radial_distance_to_lon_lat_dis(1600),0),0)
    print('traj')
    ax = fig.add_subplot(gs[0,index],projection=ccrs.PlateCarree())
    ax.add_feature(cartopy.feature.NaturalEarthFeature('physical',name='land',scale='50m'),zorder=0, edgecolor='black',facecolor='lightgrey',alpha=0.5)
    
    for q in range(len(tralon[:,0])):
        seg = helper.make_segments(tralon[q,:],tralat[q,:])
        z = pvr[q,:]
        lc = mcoll.LineCollection(seg, array=z, cmap=cmap, norm=norm, linewidth=linewidth, alpha=alpha)
        ax=plt.gca()
        ax.add_collection(lc)


    lonticks=np.arange(minpltlonc, maxpltlonc,5)
    latticks=np.arange(minpltlatc, maxpltlatc+1,5)

#    ax.set_aspect('equal')
    ax.set_xticks(lonticks)#, crs=ccrs.PlateCarree())
    ax.set_yticks(latticks)#, crs=ccrs.PlateCarree())
    ax.set_xticklabels(labels=lonticks,fontsize=8)
    ax.set_yticklabels(labels=latticks,fontsize=8)

    ax.contour(LON,LAT,ZB,levels=np.arange(800,3000,400),linewidths=0.5,colors='purple')
    ax.xaxis.set_major_formatter(LongitudeFormatter())
    ax.yaxis.set_major_formatter(LatitudeFormatter())

    ax.set_xlim([minpltlonc,maxpltlonc])
    ax.set_ylim([minpltlatc,maxpltlatc])
#    ax.set_extent([minpltlonc, maxpltlonc, minpltlatc, maxpltlatc], ccrs.PlateCarree())

    ax.plot(tracklo,trackla,color='k')
    ax.scatter(tracklo[0],trackla[0],color='k',marker='o',s=25,zorder=50)

    cbax = fig.add_axes([0, 0, 0.1, 0.1])
    cbar=plt.colorbar(lc, ticks=pvr_levels,cax=cbax)

    func=resize_colorbar_vert(cbax, ax, pad=0.0, size=0.02)
    fig.canvas.mpl_connect('draw_event', func)
    fig.set_figwidth(8)
    fig.set_figheight(6)
    cbar.ax.tick_params(labelsize=8)
    cbar.ax.set_xlabel('PVU h$^{-1}$',fontsize=10)
    cbar.ax.set_xticklabels(ticklabels)
    te = text[index]
    ax.text(-0.07, 0.95, te, transform=ax.transAxes,fontsize=14, va='top')

    index+=1

figname='/atmosdyn2/ascherrmann/paper/cyc-env-PV/fig11.png'
fig.savefig(figname,dpi=300,bbox_inches='tight')
plt.close('all')







