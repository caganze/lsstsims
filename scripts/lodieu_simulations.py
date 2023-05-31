#test the model class on JSWT simulations 
import numpy as np
import pandas as pd
#import splat
#import popsims
from astropy.coordinates import SkyCoord, Galactic
#from popsims.galaxy import Pointing, volume_calc, create_pop
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
#sns.set_style("darkgrid", {"axes.facecolor": ".95"})

#plt.style.use('fivethirtyeight')
#plt.style.use('dark_background')

from  matplotlib.colors import Normalize
import astropy.units as u
import popsims
import matplotlib
from tqdm import tqdm
from tqdm import tqdm
import matplotlib as mpl 


from popsims.simulator import Population
from popsims.galaxy import Disk, Halo
from popsims.relations import polynomial_relation
import splat.empirical as spe
import warnings
warnings.filterwarnings("ignore")

POL=(np.load('/users/caganze/research/popsimsdata/abs_mag_relations.npy', allow_pickle=True)).flatten()[0]
print (POL['absmags_spt']['subdwarfs'].keys())
print (POL['absmags_spt']['esd'].keys())

#roman's metal poor models
df_roman=pd.read_csv('/users/caganze/research/roman_new_isochrones.csv').rename(columns={'teff': 'temperature', 
                                 'lum': 'luminosity'})

METAL_POOR_EVOL= popsims.EvolutionaryModel(df_roman)

SDSS={'FOV': 2.5*u.degree*2.5*u.degree,\
      'l':((np.array([0, 360]))*u.degree.to(u.radian)),\
            'b': ((np.array([-90, 90]))*u.degree.to(u.radian))}

SDSS_discrete={'FOV': 2.5*u.degree*2.5*u.degree,\
      'l':((np.array([0, 360]))*u.degree.to(u.radian)),\
            'b': ((np.array([-90, 90]))*u.degree.to(u.radian))}

#get and combine sightlines
dfl_conct=pd.read_csv('/users/caganze/research/lsstsims/lodieu_cands.csv')
ras0=np.nanmedian([dfl_conct['RAJ2000_VHS'].values, dfl_conct['RAJ2000_LAS'].values], axis=0)
decs0=np.nanmedian([dfl_conct['DEJ2000_VHS'].values, dfl_conct['DEJ2000_LAS'].values], axis=0)

#choose 100 random sightlines
sights= np.random.choice(range(len(ras0)), 10)

#footprint=SkyCoord(ra=ras0[sights]*u.degree, dec=decs0[sights]*u.degree)
#look at random pointings
from astropy.io import ascii
t1=ascii.read('/users/caganze/ukidss_sdss_crossmatch.csv').to_pandas()
t2=ascii.read('/users/caganze/vhs_sdss_crossmatch.csv').to_pandas()
t3=ascii.read('/users/caganze/vhs_ps1_crossmatch.csv').to_pandas()

footprint_ukidss=SkyCoord(t1.RA*u.degree, t1.Dec*u.degree)
footprint_vhs= SkyCoord(t2.RA*u.degree, t2.Dec*u.degree)
footprint_ps1=SkyCoord(t3.RA*u.degree, t3.Dec*u.degree)


def compute_mags_from_reference(spt, mag_key, ref):
    vals, unc= polynomial_relation(spt, 'spt', mag_key, ref, nsample=1, xerr=0.0)
    return np.random.normal(vals, unc)

def get_best2018_relation(spt, flt):
    return spe.typeToMag(spt, flt, ref='best2018')

def get_ps1_mags(df):
    gs= get_best2018_relation(df.spt, 'PANSTARRS_G')
    rs=compute_mags_from_reference(df.spt, 'r_ps1', 'freeser2022')
    imags=compute_mags_from_reference(df.spt, 'i_ps1', 'freeser2022')
    zs=compute_mags_from_reference(df.spt, 'z_ps1', 'freeser2022')
    ys=compute_mags_from_reference(df.spt, 'y_ps1', 'freeser2022')
    
    #use beset et al for <16
    best_gs= get_best2018_relation(df.spt, 'PANSTARRS_G')
    best_rs= get_best2018_relation(df.spt, 'PANSTARRS_R')
    best_is= get_best2018_relation(df.spt, 'PANSTARRS_I')
    best_zs= get_best2018_relation(df.spt, 'PANSTARRS_Z')
    best_ys= get_best2018_relation(df.spt, 'PANSTARRS_Y')

    #mask
    mask= df['spt']<=20

    for m, ml in zip([gs, rs, imags, zs, ys], ['G', 'R', 'I', 'Z', 'Y']):
        df['abs_PANSTARRS_{}'.format(ml)]=np.random.normal(m[0], m[1])
        df['PANSTARRS_{}'.format(ml)]=   df['abs_PANSTARRS_{}'.format(ml)]+5*np.log10(df.distance/10.0)
       
    for m, ml in zip([best_gs, best_rs, best_is, best_zs, best_ys], ['G', 'R', 'I', 'Z', 'Y']):
        df['abs_PANSTARRS_{}'.format(ml)]=np.random.normal(m[0], m[1])
        df['PANSTARRS_{}'.format(ml)]=   df['abs_PANSTARRS_{}'.format(ml)]+5*np.log10(df.distance/10.0)
        
    return df

def get_maximum_distances(spt, kind, maglimits):
    #for all the filters, compute the largest volume and draw distances from that
    dmaxs=[]
    for k in maglimits.keys():
            mag_cut= maglimits[k]
            absmag= np.poly1d(POL['absmags_spt'][kind][k]['fit'])(spt)
            dmaxs.append(10.**(-(absmag-mag_cut)/5. + 1.))
    return np.nanmax(dmaxs)

def get_volume(footprint, dmax, gmodel):
    vol=0.
    for s in  footprint:
        l=s.galactic.l.radian
        b=s.galactic.b.radian
        vol += gmodel.volume(l, b, 0.1, dmax)
    return vol


def simulate_survey(keys, maglimit, footprint, ps1=False):
    #use the brighest magnitude cut
    #k= [k for k in maglimit.keys()][0]
    #absmag= np.poly1d(popsims.simulator.POLYNOMIALS['absmags_spt']['dwarfs'][k]['fit'])(15)
    #mag_cut= maglimit[k]
    #dmax=10.**(-(absmag-mag_cut)/5. + 1.)
    #--> to be more precise, can simulate within volume for given spt
    #samples=pd.DataFrame()
    
    #special case for thin disk panstarrs, use the mag relations from freeser
    
    #thin disk
    
    #dfs=[]
    
    #for dmax in [10, 50, 100, 500, 1000, 2000, 5_000, 10_000]:
    #dmax=3_000 #only simulate up to a certain distances
    nsample=1e5
    #dgrid=np.arange(10, 40)
    #dmaxs=np.empty(shape=(2, len (dgrid))).T
    #dmaxs[dgrid<17]=[0.1, 2000]
    #dmaxs[np.logical_and(dgrid>=17, dgrid<=19)]=[0.1, 500]
    #dmaxs[dgrid>19]=[0.1, 200]
    sptgrid=np.arange(14, 40)
    dminss=0.1*np.ones_like(sptgrid)
    dmaxss=1.5*np.array([get_maximum_distances(x, 'dwarfs', maglimit) for x  in sptgrid])
    dmaxss_sd=1.5*np.array([get_maximum_distances(x, 'subdwarfs', maglimit) for x in sptgrid])
    dmaxss_esd=1.5*np.array([get_maximum_distances(x,'esd', maglimit) for x in sptgrid])
    drange=dict(zip(sptgrid,np.vstack([dminss, dmaxss]).T ))
    #drange=dict(zip(dgrid, dmaxs))

    p1=Population(evolmodel= 'burrows1997',
                  imf_power=-0.6,
                  binary_fraction=0.2,
                  age_range=[0, 8],
                  mass_range=[0.01, .1],
                 nsample=nsample)

    p1.simulate()

    #p1.add_distances( Disk(H=300, L=2600), footprint.galactic.l.radian[sights],footprint.galactic.b.radian[sights], 0.1,  dmax, dsteps=1000)
    p1.assign_distance_from_spt_ranges(Disk(H=300, L=2600), footprint.galactic.l.radian, footprint.galactic.b.radian, drange)

    #add magnitudes from pre-defined filters or pre-define polynomial cofficients
    p1.add_magnitudes(keys, get_from='spt',object_type='dwarfs')
    p1.add_kinematics(footprint.ra.degree, footprint.dec.degree, kind='thin_disk', red_prop_motions_keys=keys)

    p2=Population(evolmodel= METAL_POOR_EVOL,
                  imf_power=-0.6,
                  binary_fraction=0.2,
                  age_range=[8, 14],
                  mass_range=[0.01, .1],
                 nsample=nsample)

    p2.simulate()
    print (len(p2.spt))

    #p2.add_distances( Disk(H=900, L=3600), footprint.galactic.l.radian[sights],footprint.galactic.b.radian[sights], 0.1,  dmax)
    p2.assign_distance_from_spt_ranges(Disk(H=900, L=3600), footprint.galactic.l.radian, footprint.galactic.b.radian, drange)

    #add magnitudes from pre-defined filters or pre-define polynomial cofficients
    p2.add_magnitudes(keys, get_from='spt',object_type='subdwarfs', pol=POL['absmags_spt']['subdwarfs'])
    p2.add_kinematics(footprint.ra.degree, footprint.dec.degree, kind='thick_disk', red_prop_motions_keys=keys)



    p3=Population(evolmodel= METAL_POOR_EVOL,
                  imf_power=-0.6,
                  binary_fraction=0.2,
                  age_range=[10, 14],
                  mass_range=[0.01, .1],
                 nsample=nsample)

    p3.simulate()

    #p3.add_distances( Halo(), footprint.galactic.l.radian[sights],footprint.galactic.b.radian[sights], 0.1, dmax)
    p3.assign_distance_from_spt_ranges(Halo(), footprint.galactic.l.radian, footprint.galactic.b.radian, drange)



    #add magnitudes from pre-defined filters or pre-define polynomial cofficients
    p3.add_magnitudes(keys, get_from='spt',object_type='esd', pol=POL['absmags_spt']['esd'])
    p3.add_kinematics(footprint.ra.degree, footprint.dec.degree, kind='halo', red_prop_motions_keys=keys)

    p1.scale_to_local_lf()
    p2.scale_to_local_lf()
    p3.scale_to_local_lf()

    mag_cols= np.concatenate([['abs_'+x for x in keys], keys, ['redH_'+x for x in keys]])

    cols=np.concatenate([mag_cols, ['spt', 'scale', 'mass', 'age', 'temperature',  'r', 'z', 'l', 'b', 'U', 'V', 'W', 'RV',\
                                     'mu_alpha_cosdec', 'mu_delta', 'Vr', 'Vphi', 'Vz',  'scale_unc', 'scale_times_model', 'l', 'b','distance']])

    df1=p1.to_dataframe(cols)
    df1['population']='thin disk'
    #ADD PS1 mags

    df2=p2.to_dataframe(cols)
    df2['population']='thick disk'

    df3=p3.to_dataframe(cols)
    df3['population']='halo'

    df=pd.concat([df1, df2, df3]).reset_index(drop=True)
    #add PS1 mga
    #print (df.columns)
    df=get_ps1_mags(df)

    #apply magnitude cuts

    df=df[np.logical_and.reduce([df[k] <  maglimit[k] for k in maglimit.keys() ])].reset_index(drop=True)
        
    #dfs.append(df)

    #return pd.concat(dfs)

    thind_vols=[get_volume(footprint, x , Disk(H=300, L=2600)) for x in dmaxss]
    thickd_vols=[get_volume(footprint, x , Disk(H=900, L=3600)) for x in dmaxss]
    halo_vols=[get_volume(footprint, x , Halo()) for x in dmaxss]
                 
    res={'data': df,
         'nsample': nsample,
         'volume': {'thin_disk':  thind_vols, 'thick_disk': thickd_vols, 'halo': halo_vols }, #note that solid angle not considered here
         'footprint': footprint,
         'mag_limits':maglimit,
         'sptgrid': sptgrid
    }
    #add an option to add data until we reach desired nsample (after cuts)
    return res
    
    
#run a combined ukidss-sdss 
ukidss_sdss= simulate_survey(['SDSS_G', 'SDSS_R','SDSS_I', 'SDSS_Z', 
                             'UKIDSS_Y', 'UKIDSS_J', 'UKIDSS_H', 'UKIDSS_K'], \
                             {'UKIDSS_Y': 20.3, 'UKIDSS_J': 19.9}, footprint_ukidss)
#Y=20.3,J=19.9,H=18.6,K=18.2
#G=23.15

#run a combined ukidss-sdss 
vhs_sdss= simulate_survey(['SDSS_G', 'SDSS_R','SDSS_I', 'SDSS_Z', 
                             'VISTA_Y', 'VISTA_J', 'VISTA_H', 'VISTA_KS'], \
                             {'SDSS_G': 23.15}, footprint_vhs)
#vhs panstars

vhs_ps=  simulate_survey(['VISTA_Y', 'VISTA_J', 'VISTA_H', 'VISTA_KS'], \
                             {'VISTA_J': 19.5}, footprint_ps1)

#save data
res={'ukidss_sdss': ukidss_sdss, 'vhs_sdss': vhs_sdss, 'vhs_ps': vhs_ps }

print (len(ukidss_sdss), len(vhs_sdss), len(vhs_ps))

np.save('../simulations.npy', res, allow_pickle=True) 