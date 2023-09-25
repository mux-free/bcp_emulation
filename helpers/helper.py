import numpy as np
from datetime import datetime, date, timedelta
import joblib




### LOAD RF-MODELS
def load_rf_models():
    print(' =========================================================')
    print('                   *** Load Models ***                    ')
    ## Load models:
    print('  Load model tsubsi')
    model_tsubsi = joblib.load(f'/net/helium/atmosdyn/freimax/data_msc/IFS-18/rf_models/tsubsi/rf_fulldata_gridsearch_f1.joblib')
    
    print('  Load model tmeltsi')
    model_tmeltsi = joblib.load("/net/helium/atmosdyn/freimax/data_msc/IFS-18/rf_models/tmeltsi/full_data_girdsearch_tmeltsi_f1.joblib")
    
    print('  Load model tevr')
    filepath = f"/net/helium/atmosdyn/freimax/data_msc/IFS-18/rf_models/tevr/rf_fulldata_gridsearch_f1.joblib"
    model_tevr = joblib.load(filepath)

    print(' =========================================================')
    return model_tsubsi, model_tmeltsi, model_tevr




def get_memory_usage(data, name_df=''):
    # Print the memory usage of a pandas dataframe
    size_data = data.memory_usage().sum()
    #print("Initial memory usage:", size_dataset1)
    def format_size(size):
        power = 2**10
        n = 0
        size_labels = {0: 'B', 1: 'KB', 2: 'MB', 3: 'GB', 4: 'TB'}
        while size > power:
            size /= power
            n += 1
        return f"{size:.2f} {size_labels[n]}"
    # Usage example
    print('\n- - - - - - - - - - - - - - - -')
    print(f'Memory usage {name_df}: ', format_size(size_data))
    print('- - - - - - - - - - - - - - - -')
    return size_data



def modellevel_to_pressure(PS):
    aklay = np.array([0, 0.01878906, 0.1329688, 0.4280859, 0.924414, 1.62293, 2.524805, 3.634453, 4.962383, 6.515274, 8.3075, 10.34879, 12.65398, 15.23512,  \
                        18.10488, 21.27871, 24.76691, 28.58203, 32.7325, 37.22598, 42.06668, 47.25586, 52.7909, 58.66457, 64.86477, 71.37383, 78.16859, 85.21914,  \
                                92.48985, 99.93845, 107.5174, 115.1732, 122.848, 130.4801, 138.0055, 145.3589, 152.4757, 159.2937, 165.7537, 171.8026, 177.3938, 182.4832,  \
                                        187.0358, 191.0384, 194.494, 197.413, 199.8055, 201.683, 203.0566, 203.9377, 204.339, 204.2719, 203.7509, 202.7876, 201.398, 199.5966,  \
                                                197.3972, 194.8178, 191.874, 188.585, 184.9708, 181.0503, 176.8462, 172.382, 167.6805, 162.7672, 157.6719, 152.4194, 147.0388, 141.5674, \
                                                          136.03, 130.4577, 124.8921, 119.3581, 113.8837, 108.5065, 103.253, 98.1433, 93.19541, 88.42463, 83.83939, 79.43383, 75.1964 ])
    bklay = np.array([0.9988151, 0.9963163, 0.9934933, 0.9902418, 0.9865207, 0.9823067, 0.977575, 0.9722959, 0.9664326, 0.9599506, 0.9528069, 0.944962,  \
                        0.9363701, 0.9269882, 0.9167719, 0.9056743, 0.893654, 0.8806684, 0.8666805, 0.8516564, 0.8355686, 0.8183961, 0.8001264, 0.7807572,  \
                                0.7602971, 0.7387676, 0.7162039, 0.692656, 0.6681895, 0.6428859, 0.6168419, 0.5901701, 0.5629966, 0.5354602, 0.5077097, 0.4799018,  \
                                        0.4521973, 0.424758, 0.3977441, 0.3713087, 0.3455966, 0.3207688, 0.2969762, 0.274298, 0.2527429, 0.2322884, 0.212912, 0.1945903,  \
                                                0.1772999, 0.1610177, 0.145719, 0.1313805, 0.1179764, 0.1054832, 0.0938737, 0.08312202, 0.07320328, 0.06408833, 0.05575071, 0.04816049,  \
                                                        0.04128718, 0.03510125, 0.02956981, 0.02465918, 0.02033665, 0.01656704, 0.01331083, 0.01053374, 0.008197418, 0.006255596, 0.004674384,  \
                                                                0.003414039, 0.002424481, 0.001672322, 0.001121252, 0.0007256266, 0.0004509675, 0.0002694785, 0.0001552459, 8.541815e-05, 4.1635e-05, 1.555435e-05, 3.39945e-06])

    return (aklay + bklay * PS)


def modellevel_ERA5(PS,hya,hyb):
    return 0.01*hya[np.arange(39,137)] + hyb[np.arange(39,137)] * PS


def det_month_dates(y,m):
    if (m<8 and m%2==1) or (m>=8 and m%2==0):
        md=31
    else:
        md=30
    if m==2:
        md=28
        if y%4==0:
            md = 29
    return md

def change_date_by_hours(date,hours):
    y=int(date[:4])
    m=int(date[4:6])
    d=int(date[6:8])
    h=int(date[-2:])
    md = det_month_dates(y,m)
    h+=hours
    if h>=24:
        while h>=24:
            h-=24
            d+=1
            
        if d>md:
            while d>md:
                d-= md
                m+= 1
                md = det_month_dates(y,m)
                if m>12:
                    m=1
                    y+=1
    if h<0:
        while h<0:
            h+=24
            d-=1
        if d<1:
            while d<1:
                if m==1:
                    d+=31
                else:
                    d+= det_month_dates(y,m-1)
                m-= 1
                if m<1:
                    m=12
                    y-=1
    return '%04d%02d%02d_%02d'%(y,m,d,h)

def simulation_time_to_day_string(h):
    dd = '%02d'%(1 + int(h/24))
    hh = '%02d'%(int(h%24))
    return (dd + '_' + hh)

def convert_radial_distance_to_lon_lat_dis(dis):
    return (dis /2./np.pi/6370 * 360)

def convert_lon_lat_dis_to_radial_dis(deltalatlon):
    return (deltalatlon/360 * 2 * np.pi * 6370)

def convert_dlon_dlat_to_radial_dis_new(dlon,dlat,latitude):
    return (np.pi * 6370/180 * np.sqrt(dlat**2 + (dlon**2 * (np.cos(latitude/180 * np.pi))**2)))

def convert_radial_distance_to_lon_lat_dis_new(dis,latitude):
    return (dis /np.pi/6370/np.cos(latitude/180*np.pi) * 180)

def radial_ids_around_center(dis):
    """
    Return longitude, latidue ids that are within radius r=dis around the center
    """
    r = 6370

    DisLon = np.linspace(0,2 * np.pi * r,901)
    DisLat = np.linspace(0,0.5 * np.pi * r,226)

    DisLon = DisLon - DisLon[450]
    DisLat = DisLat- DisLat[113]

    disx, disy = np.meshgrid(DisLon,DisLat)

    r = np.sqrt(disx**2 + disy**2)
    Latids, Lonids = np.where(r<dis)
    Lonids = Lonids-451 #use 451 and 114 as there is a shift in the data set provided by roman
    Latids = Latids-114 #therefore shift it one towards west and south

    return Lonids, Latids

def radial_ids_around_center_calc(dis):
    """
    Return longitude, latidue ids that are within radius r=dis around the center
    """
    r = 6370

    DisLon = np.linspace(0,2 * np.pi * r,901)
    DisLat = np.linspace(0,0.5 * np.pi * r,226)

    DisLon = DisLon - DisLon[450]
    DisLat = DisLat- DisLat[113]

    disx, disy = np.meshgrid(DisLon,DisLat)

    r = np.sqrt(disx**2 + disy**2)
    Latids, Lonids = np.where(r<dis)
    Lonids = Lonids-450 #use 450 and 113 as the shift is alraedy accounted for in the txt files created
    Latids = Latids-113 #therefore this works with the true center and need no shift

    return Lonids, Latids

def radial_ids_around_center_calc_ERA5(dis):
    """
    Return longitude, latidue ids that are within radius r=dis around the center
    """
    r = 6370

    DisLon = np.linspace(0,2 * np.pi * r,721)
    DisLat = np.linspace(0,np.pi * r,361)

    DisLon = DisLon - DisLon[360]
    DisLat = DisLat- DisLat[180]

    disx, disy = np.meshgrid(DisLon,DisLat)

    r = np.sqrt(disx**2 + disy**2)
    Latids, Lonids = np.where(r<dis)
    Lonids = Lonids-360 #use 450 and 113 as the shift is alraedy accounted for in the txt files created
    Latids = Latids-180 #therefore this works with the true center and need no shift

    return Lonids, Latids


def ERA5_radial_ids_correct(rdis,centerlat):
    R=6370
    trash, LATIDS = radial_ids_around_center_calc_ERA5(rdis)
    LATIDS = np.unique(LATIDS).astype(int) #unaffected by latitude strech of longitude 

    lat = np.arange(-90,90.1,0.5)
    latid = np.where(abs(lat-centerlat)==np.min(abs(lat-centerlat)))[0][0]
    latids = latid + LATIDS

    DisLON = np.zeros((len(latids),721))
    DisLAT = np.ones((len(latids),721))
    latdis = np.linspace(0,np.pi * R,361)
    latdis = latdis - latdis[latid]
    latdis = latdis[latids]
    
    for q,k in enumerate(lat[latids]):
        DisLON[q] = np.linspace(0,2 * np.pi * R * np.cos(k/180*np.pi),721)
        DisLAT[q] *=  latdis[q]    
        DisLON[q] = DisLON[q]-DisLON[q][360]

    r = np.sqrt(DisLON**2 + DisLAT**2)
    Latids, Lonids = np.where(r<rdis)

    Lonids=Lonids-360 
    Latids=Latids-(len(latids)-1)/2 #### 200 km corresponds to 7 latitude points in ERA5 

    return Lonids, Latids


def IFS_radial_ids_correct(rdis,centerlat):
    R=6370
    trash, LATIDS = radial_ids_around_center_calc(rdis)
    LATIDS = np.unique(LATIDS).astype(int) #unaffected by latitude strech of longitude

    lat = np.arange(0,90.1,0.4)
    latid = np.where(abs(lat-centerlat)==np.min(abs(lat-centerlat)))[0][0]
    latids = latid + LATIDS

    DisLON = np.zeros((len(latids),901))
    DisLAT = np.ones((len(latids),901))
    latdis = np.linspace(0,np.pi/2 * R,226)
    latdis = latdis - latdis[latid]
    latdis = latdis[latids]

    for q,k in enumerate(lat[latids]):
        DisLON[q] = np.linspace(0,2 * np.pi * R * np.cos(k/180*np.pi),901)
        DisLAT[q] *=  latdis[q]
        DisLON[q] = DisLON[q]-DisLON[q][450]

    r = np.sqrt(DisLON**2 + DisLAT**2)
    Latids, Lonids = np.where(r<rdis)

    Lonids=Lonids-451
    Latids=Latids-(len(latids)-1)/2-1 #### 200 km corresponds to 7 latitude points in ERA5

    return Lonids, Latids

def lonlatids(lon,lat,dis):
    r = 6370
    LAT = np.round(np.linspace(0,90,226),1)
    LON = np.round(np.linspace(-180,180,901),1)
    addlon,addlat = radial_ids_around_center(dis)
    idlon = addlon + np.where(np.round(LON,1)==lon)[0][0].astype(int)
    idlat = addlat + np.where(np.round(LAT,1)==lat)[0][0].astype(int)

    return idlon, idlat

def distance(dis):
    r = 6370
    DisLon = np.linspace(0,2 * np.pi * r,901)
    DisLat = np.linspace(0,0.5 * np.pi * r,226)
    DisLon = DisLon - DisLon[450]
    DisLat = DisLat- DisLat[113]
    lonidst = np.where(abs(DisLon)<dis)[0]
    latidst = np.where(abs(DisLat)<dis)[0]

    disy, disx = np.meshgrid(DisLon[lonidst],DisLat[latidst])

    return disx,disy



def datenum_to_datetime(datenum):
    """
    Convert Matlab datenum into Python datetime.
    :param datenum: Date in datenum format
    :return:        Datetime object corresponding to datenum.
    """
    days = datenum % 1
    hours = days % 1 * 24
    minutes = hours % 1 * 60
    seconds = minutes % 1 * 60
    return datetime.fromordinal(int(datenum)) + timedelta(days=int(days))  + timedelta(hours=int(hours))+ timedelta(minutes=int(minutes)) + timedelta(seconds=round(seconds))


def S_variables():
    return ['PV','P','PS','TH','THE','VORT','RH','PVRCONVT','PVRCONVM','PVRTURBT','PVRTURBM','PVRLS','PVRCOND','PVRSW','PVRLWH','PVRLWC','PVRDEP','PVREVC','PVREVR','PVRSUBI','PVRSUBS','PVRMELTI','PVRMELTS','PVRFRZ','PVRRIME','PVRBF']

def P_variables():
    return ['T','U','V','OMEGA','Q','SWC', 'RWC', 'IWC', 'LWC','SLP','PS','CC'] #heating rates excluded

def make_segments(x, y):
    """
    Linesegments for vertical color lines
    """
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    return segments

def months():
    return np.array(['JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC']), np.arange(1,13,1)

def month_days(yr):
    if yr%4==0:
        return np.array([31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
    else:
        return np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])


# def saturation_pressure(T):
#     return 6.1094e2*np.exp((17.625*T)/(T + 243.04))

# def qs(T,p):
#     return 0.622 * saturation_pressure(T)/p



