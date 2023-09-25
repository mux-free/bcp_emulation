#!/usr/bin/env python

#%%############ functions ####################
def readcdf(ncfile,varnam):
    infile = netCDF4.Dataset(ncfile, mode='r')
    var = infile.variables[varnam][:]
    return(var)
    
def inter2level(varr3D, parr3D, plevel):
    """
    Interpolates 3-D (level, lat, lon) over level for variable array varr with
    associated pressure grid parr to the scalar pressure level plevel
    """ 
    v_i = interpolate(varr3D[::1,:, :], parr3D[:, :], plevel)
    return(v_i)

def calc_RH(Q,T,p): #Q,T,p can be 3D fields and T is provided in K, I think p should be in Pa rather than hPa, but find out by trying both
    return 0.263 * p * Q/ (np.exp(17.67 * (T-273.16)/(T-29.65)))


###
### to work with pickle and data saving
### import pickle
### f = open('file','wb') #w is for writing
### pickle.dump(data,f) data can be any python object
### f.close()
###
### f = open('file','rb') # r is for reading, make sure to use it, otherwise you will immediately overwrite the file
### data = pickle.load(f)
### f.close()



############# end functions ##################   


## Import modules
from dypy.intergrid import Intergrid
import matplotlib.colors as col
import matplotlib.cm as cm
from mpl_toolkits.basemap import Basemap
import numpy as np
import netCDF4
from netCDF4 import Dataset as ncFile
from dypy.small_tools import interpolate
from dypy.lagranto import Tra
import matplotlib.pyplot as plt
import matplotlib as mpl
import cartopy
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from matplotlib.gridspec import GridSpec
from matplotlib.colors import from_levels_and_colors
import datetime as dt
import sys
sys.path.append('/home/raphaelp/phd/scripts/basics/')
sys.path.append('/home/ascherrmann/scripts/')
import helper
from colormaps import PV_cmap2
from useful_functions import get_field_at_level,resize_colorbar_horz,resize_colorbar_vert
import argparse
parser = argparse.ArgumentParser(description="Vertical cross section for ERA5")

parser.add_argument('date',default='',type=str,help='date of the cross section')
parser.add_argument('lon',default='',type=float,help='center longitude of the cross section')
parser.add_argument('lat',default='',type=float,help='center latitude of the cross section')


args = parser.parse_args()
# date
print(f'\nArgs: {args}')
dat=str(args.date)

lonc = float(args.lon)
latc = float(args.lat)
dis = 500 #km

dlon = helper.convert_radial_distance_to_lon_lat_dis_new(dis,latc)

# Define cross section line for each date (tini)
lon_start = lonc-dlon
lon_end   = lonc+dlon
lat_start = latc
lat_end   = latc 

# Define variable
var='PV'
unit='PVU'#g kg$^{-1}$'
cmap,pv_levels,norm,ticklabels=PV_cmap2()
levels=pv_levels
# Define lower and upper bound for vertical cross section (y-axis)
ymin = 200.
ymax = 1000.

dynfmt = '%Y%m%d_%H'
datobj=dt.datetime.strptime(dat,dynfmt)  # change to python time
outpath  = '/home/freimax/msc_thesis/figures/case_study_RA19/' #Wo Plot gespeichert wird

y=dat[0:4]
m=dat[4:6]
print(dat)

pfile='/net/thermo/atmosdyn/era5/cdf/'+y+'/'+m+'/P'+dat  #für Lothar ändern clim_era5/lothar ohne +y+ und +m+
sfile='/net/thermo/atmosdyn/era5/cdf/'+y+'/'+m+'/S'+dat
bfile='/net/thermo/atmosdyn/era5/cdf/'+y+'/'+m+'/B'+dat
slp=readcdf(bfile,'MSL')
ps=readcdf(pfile,'PS')
PV=readcdf(sfile,'PV')
pv=PV
#q=readcdf(pfile,'Q')
#rh=readcdf(sfile,'RH')
#th=readcdf(sfile,'TH')
#the=readcdf(sfile,'THE')
lons=readcdf(pfile,'lon')
lats=readcdf(pfile,'lat')
hyam=readcdf(pfile,'hyam')  # 137 levels  #für G-file ohne levels bis
hybm=readcdf(pfile,'hybm')  #   ''
ak=hyam[hyam.shape[0]-pv.shape[1]:] # only 98 levs are used:
bk=hybm[hybm.shape[0]-pv.shape[1]:] # reduce to 98 levels 

# Define distance delta for great circle line
ds = 5.

# Extract coordinates of great circle line between start and end point
mvcross    = Basemap()
line,      = mvcross.drawgreatcircle(lon_start, lat_start, lon_end, lat_end, del_s=ds)
path       = line.get_path()
lonp, latp = mvcross(path.vertices[:,0], path.vertices[:,1], inverse=True)
dimpath    = len(lonp)

# calculate pressure on model levels
p3d=np.full((pv.shape[1],pv.shape[2],pv.shape[3]),-999.99)
ps3d=np.tile(ps[0,:,:],(pv.shape[1],1,1)) # write/repete ps to each level of dim 0
p3d=(ak/100.+bk*ps3d.T).T
unit_p3d = 'hPa'

# Extract data along the great circle line between the start and end point
vcross = np.zeros(shape=(pv.shape[1],dimpath)) #PV
vcross_PV = np.zeros(shape=(PV.shape[1],dimpath))
vcross_p  = np.zeros(shape=(p3d.shape[0],dimpath)) #pressure
bottomleft = np.array([lats[0], lons[0]])
topright   = np.array([lats[-1], lons[-1]])

    
for k in range(pv.shape[1]):
    f_vcross     = Intergrid(pv[0,k,:,:], lo=bottomleft, hi=topright, verbose=0)
    f_vcross_PV   = Intergrid(PV[0,k,:,:], lo=bottomleft, hi=topright, verbose=0)
    f_p3d_vcross   = Intergrid(p3d[k,:,:], lo=bottomleft, hi=topright, verbose=0)
    for i in range(dimpath):
        vcross[k,i]     = f_vcross.at([latp[i],lonp[i]])
        vcross_PV[k,i]   = f_vcross_PV.at([latp[i],lonp[i]])
        vcross_p[k,i]   = f_p3d_vcross.at([latp[i],lonp[i]])
    
# Create coorinate array for x-axis
xcoord = np.zeros(shape=(pv.shape[1],dimpath))
for x in range(pv.shape[1]):
    xcoord[x,:] = np.array([ i*ds-dis for i in range(dimpath) ])
            
# Define plot settings (parameter-specific) (for secondary variables)
plt_min_2 = -3
plt_max_2 = 3
plt_d_2   = 0.5
levels_2  = np.arange(plt_min_2, plt_max_2, plt_d_2)

#------------------------------------------------------------------------FIRST PLOT: CROSSSECTION

# Create figure (vertical cross section)
fig = plt.figure()
ax  = fig.add_subplot(1,1,1)

# Plot primary variable data
#ctf = ax.contourf(xcoord, vcross_p,
#                  vcross,
#                  levels = levels,
#                  cmap = cmap,
#                  norm=norm,
#                  extend = 'both')

# Plot secondary variable data
#ct = ax.contour(xcoord, vcross_p,
#                vcross_2,
#                levels = levels_2,
#                colors = 'grey',
#                linewidths = 1.5)

h = ax.contourf(xcoord, vcross_p,
                vcross_PV,
                levels = levels,
                cmap = cmap,
                norm=norm,
                extend = 'both')

# Add contour labels
#ax.clabel(ct,
#          inline = True,
#          inline_spacing = 1,
#          fontsize = 10.,
#          fmt = '%.0f')


# Design axes
ax.text(0.03, 0.95, 'd)', transform=ax.transAxes,fontsize=12, fontweight='bold',va='top')
ax.set_xlabel('Distance from center [km]', fontsize=12)
ax.set_ylabel('Pressure [hPa]', fontsize=12)
ax.set_ylim(bottom=ymin, top=ymax)
ax.set_xlim(-500,500)
ax.set_xticks(ticks=np.arange(-500,500,250))
# Invert y-axis
plt.gca().invert_yaxis()

# Add colorbar
cbax = fig.add_axes([0, 0, 0.1, 0.1])
cbar=plt.colorbar(h, ticks=pv_levels,cax=cbax)
func=resize_colorbar_vert(cbax, ax, pad=0.0, size=0.02)
fig.canvas.mpl_connect('draw_event', func)
cbar.ax.set_xlabel(unit)

## Save figure
figname = str(var)+'_TH_'+ dat + '.png'
#print(figname)
fig.savefig(outpath+str(figname), bbox_inches = 'tight',dpi=300)
#plt.show()
#Close figure
plt.close(fig)

#------------------------------------------------------------------------SECOND PLOT: MAP
# Plot map and crosssection location
#fig2,ax2 = plt.subplots(figsize=(15,6),subplot_kw={'projection': ccrs.PlateCarree()})
#ax2.plot(projection=ccrs.PlateCarree())
#ax2.coastlines(color='grey')
#ax2.add_feature(cartopy.feature.LAND)
#ax2.gridlines(ylocs=np.arange(-90, 91, 30), xlocs=np.arange(-180, 181, 60))
#ax2.set_xticks(np.arange(-180, 181, 60), crs=ccrs.PlateCarree())
#ax2.set_yticks(np.arange(-90, 91, 30), crs=ccrs.PlateCarree())
#ax2.xaxis.set_major_formatter(LongitudeFormatter())
#ax2.yaxis.set_major_formatter(LatitudeFormatter())
#ax2.set_extent([-90,90, 0, 90])
#
#ax2.plot((lon_start,lon_end),(lat_start,lat_end),c='red')
##ax2.plot(lon_start,lat_end,'ro',) 
##Mark "start" of crossection
#
#figname2 = 'MAP_' + dat + '.png'
#fig2.savefig(outpath+figname2, bbox_inches = 'tight')
##plt.show()
#plt.close(fig2)
