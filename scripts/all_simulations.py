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

#survey filters
df_roman=pd.read_csv('/users/caganze/research/roman_new_isochrones.csv').rename(columns={'teff': 'temperature', 
                                 'lum': 'luminosity'})

METAL_POOR_EVOL= popsims.EvolutionaryModel(df_roman)

def random_points_above_latitute(n, bmin):
    l, b=popsims.random_angles(10*n)
    mask= np.abs(b)>bmin
    l= l[mask][:n]
    b= b[mask][:n]
    return l, b

SURVEY_AREAS= {
       'Rubin Wide': 18_000*u.degree**2, 
       'Roman HLWAS': 2_000*u.degree**2,
       'Roman HLTDS': 20*u.degree**2 ,
       #'Roman GBTDS': 2*u.degree**2,
       'Euclid Wide': 15_000*u.degree**2,
       'Rubin 10 year': 18_000*u.degree**2, 
       'Euclid Deep': 40*u.degree**2,
       #'JWST PASSAGE': 124*4*u.arcmin**2,
       #'JWST JADES': 65*u.arcmin**2,
       #'JWST CEERS': 100*u.arcmin**2,
       #'JWST NGDEEP': 10*u.arcmin**2,
}

SURVEY_DEPTHS= {
       'Rubin Wide': {'LSST_G': 25., 'LSST_R': 24.7, 'LSST_I': 24.0, 'LSST_Z': 23.3, 'LSST_Y': 22.1}, 
       'Roman HLWAS': {'WFI_J129': 26.7},
       'Roman HLTDS': {'WFI_J129': 26.7},
       #'Roman GBTDS': {'WFI_J129': 26.7},
       'Euclid Wide': {'EUCLID_J': 24., 'EUCLID_H': 24. },
       'Rubin 10 year': {'LSST_G': 25.3, 'LSST_R': 25.6, 'LSST_I': 25.4, 'LSST_Z': 24.9, 'LSST_Y': 24.}, 
       'Euclid Deep': {'EUCLID_J': 27., 'EUCLID_H': 27. },
       #'JWST PASSAGE': {'NIRISS_F115W': 27},
       #'JWST JADES': {'NIRISS_F115W': 29},
       #'JWST CEERS':  {'NIRISS_F115W': 30},
       #'JWST NGDEEP': {'NIRISS_F115W': 30},
}

#check the right pointings later
SURVEY_FOOTPRINT= {
       'Rubin Wide': SkyCoord(*random_points_above_latitute(100, 20*u.degree.to(u.radian))*u.radian),
       'Roman HLWAS':SkyCoord(*random_points_above_latitute(100, 20*u.degree.to(u.radian))*u.radian),
       'Roman HLTDS':SkyCoord(*random_points_above_latitute(100, 36*u.degree.to(u.radian))*u.radian), #random
       #'Roman GBTDS': SkyCoord(*popsims.random_angles(100)*u.radian), #random
       'Euclid Wide': SkyCoord(*random_points_above_latitute(100, 20*u.degree.to(u.radian))*u.radian),
       'Rubin 10 year': SkyCoord(*random_points_above_latitute(100, 20*u.degree.to(u.radian))*u.radian),
       'Euclid Deep': SkyCoord(ra=[189.22]*u.degree, dec=[62.2375]*u.degree), #GOODS-NORTH (189.22, 62.2375
       #'JWST PASSAGE': SkyCoord(*popsims.random_angles(10)*u.radian), #random
       #'JWST JADES': SkyCoord(ra=[53]*u.degree, dec=[-27.7]*u.degree), #goods-sotu
       
       #'JWST CEERS':  SkyCoord(*popsims.random_angles(10)*u.radian), #random
       #'JWST NGDEEP':SkyCoord(*popsims.random_angles(10)*u.radian), #random
}


#filters to simulate 
#check the right pointings later
FILTERS= {
       'Rubin Wide':  ['LSST_'+x for x in 'G R I Z Y'.split()],
       'Roman HLWAS':  ['WFI_'+x for x in 'R062 Z087 Y106 J129 H158 F184 Prism Grism'.split()],
       #'Roman HLTDS':  ['WFI_'+x for x in 'R062 Z087 Y106 J129 H158 F184 Prism Grism'.split()],
       #'Roman GBTDS': ['WFI_'+x for x in 'R062 Z087 Y106 J129 H158 F184 Prism Grism'.split()],
       'Euclid Wide': ['EUCLID_'+x for x in 'Y J H'.split()],
       'Rubin 10 year':  ['LSST_'+x for x in 'G R I Z Y'.split()],
       'Euclid Deep': ['EUCLID_'+x for x in 'Y J H'.split()],
       #'JWST PASSAGE':  ['NIRISS_'+x for x in 'F115W F200W F150W'.split()], #CHANGE LATER
       #'JWST JADES': ['NIRISS_'+x for x in 'F115W F200W F150W'.split()],
       #'JWST CEERS':  ['NIRISS_'+x for x in 'F115W F200W F150W'.split()],
       #'JWST NGDEEP':['NIRISS_'+x for x in 'F115W F200W F150W'.split()], #CHANGE LATER
}


#filenames
FILENAMES={
       'Rubin Wide': 'rubin_wide',
       'Roman HLWAS': 'roman_hlwas',
       'Roman HLTDS': 'roman_hltds',
       #'Roman GBTDS': 'roman_gbtds',
       'Euclid Wide': 'euclid_wide',
       'Rubin 10 year': 'rubin_10year',
       'Euclid Deep': 'euclid_deep',
       #JWST PASSAGE': 'jwst_passage',
       #'JWST JADES': 'jwst_jades',
       #'JWST CEERS':  'jwst_ceers',
       #'JWST NGDEEP': 'jwst_ngdeep'
}


def get_maximum_distances(spt, kind, maglimits):
    #for all the filters, compute the largest volume and draw distances from that
    dmaxs=[]
    for k in maglimits.keys():
            mag_cut= maglimits[k]
            absmag= np.poly1d(POL['absmags_spt'][kind][k]['fit'])(spt)
            dmaxs.append(10.**(-(absmag-mag_cut)/5. + 1.))
    return np.nanmax(dmaxs)

def get_volume(footprint, dmax, gmodel):
    vol={}
    #save volumes now as hash because it takes a few more seconds to compute
    for s in  footprint:
        l=s.galactic.l.radian
        b=s.galactic.b.radian
        k= hash((round(s.galactic.l.to(u.degree).value, 1), round(s.galactic.b.to(u.degree).value, 1)))
        vol[k]= gmodel.volume(l, b, 0.1, dmax)
    return vol

def get_pointing(lb):
    #use hash key to get l and b coordinates of pointing
    l, b=lb*u.radian
    return hash((round(l.to(u.degree).value, 1), round(b.to(u.degree).value, 1)))

def simulate_survey(keys, maglimit, footprint, filename, Hthin=300, haloIMF=-0.6):

    nsample=1e6
    sptgrid=np.arange(14, 40)
    dminss=0.1*np.ones_like(sptgrid)
    dmaxss=1.2*np.array([get_maximum_distances(x, 'dwarfs', maglimit) for x  in sptgrid])
    dmaxss_sd=1.2*np.array([get_maximum_distances(x, 'subdwarfs', maglimit) for x in sptgrid])
    dmaxss_esd=1.2*np.array([get_maximum_distances(x,'esd', maglimit) for x in sptgrid])

    drange=dict(zip(sptgrid,np.vstack([dminss, dmaxss]).T ))

    p1=Population(evolmodel= 'burrows1997',
                  imf_power=-0.6,
                  binary_fraction=0.2,
                  age_range=[0, 8],
                  mass_range=[0.01, .1],
                 nsample=nsample)

    p1.simulate()

    p1.assign_distance_from_spt_ranges(Disk(H=Hthin, L=2600), footprint.galactic.l.radian, footprint.galactic.b.radian, drange)

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


    p2.assign_distance_from_spt_ranges(Disk(H=900, L=3600), footprint.galactic.l.radian, footprint.galactic.b.radian, drange)

    
    p2.add_magnitudes(keys, get_from='spt',object_type='subdwarfs', pol=POL['absmags_spt']['subdwarfs'])
    p2.add_kinematics(footprint.ra.degree, footprint.dec.degree, kind='thick_disk', red_prop_motions_keys=keys)



    p3=Population(evolmodel= METAL_POOR_EVOL,
                  imf_power=haloIMF,
                  binary_fraction=0.2,
                  age_range=[10, 14],
                  mass_range=[0.01, .1],
                 nsample=nsample)

    p3.simulate()

   
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
    df1['FeH']=0.

    df2=p2.to_dataframe(cols)
    df2['population']='thick disk'
    df1['FeH']=-0.5

    df3=p3.to_dataframe(cols)
    df3['population']='halo'
    df3['FeH']=-1.5

    df=pd.concat([df1, df2, df3]).reset_index(drop=True)
    df['pointing']= df[['l', 'b']].apply(get_pointing, axis=1)

    #print (df1.columns)
    #print (df2.columns)
    #print (df3.columns)

    thind_vols=[get_volume(footprint, x , Disk(H=300, L=2600)) for x in dmaxss]
    thickd_vols=[get_volume(footprint, x , Disk(H=900, L=3600)) for x in dmaxss]
    halo_vols=[get_volume(footprint, x , Halo()) for x in dmaxss]

    #selection function by pointing!!!
    res={'data': df,
         'nsample': nsample,
         'volume': {'thin disk':  thind_vols, 'thick disk': thickd_vols, 'halo': halo_vols }, #note that solid angle not considered here
         'footprint': footprint,
         'mag_limits':maglimit,
         'sptgrid': sptgrid,
         'Hthin': Hthin, 
         'haloIMF':haloIMF,
    }
    #add an option to add data until we reach desired nsample (after cuts)
    np.save('../simulations{}.npy'.format(filename), res, allow_pickle=True) 


#survey nominal
for survey in FILTERS.keys():
     simulate_survey(FILTERS[survey], SURVEY_DEPTHS[survey], SURVEY_FOOTPRINT[survey], FILENAMES[survey])

#vary scaleheights
survey='Rubin Wide'
for h in [100, 200, 300, 400]:
    simulate_survey(FILTERS[survey], SURVEY_DEPTHS[survey], SURVEY_FOOTPRINT[survey], 'rubin_widethin{}'.format(h),  Hthin=h)

for imf in [2.3, -2.3, 0.0]:
    simulate_survey(FILTERS[survey], SURVEY_DEPTHS[survey], SURVEY_FOOTPRINT[survey], 'rubin_widehalo{}'.format(imf),   haloIMF=imf)
