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


SURVEY_AREAS= {
       'Rubin Wide': 18_000*u.degree**2, 
       'Roman HLWAS': 2_000*u.degree**2,
       'Roman HLTDS': 20*u.degree**2 ,
       'Roman GBTDS': 2*u.degree**2,
       'Euclid Wide': 15_000*u.degree**2,
       'Rubin Deep': 18_000*u.degree**2, 
       'Euclid Deep': 40*u.degree**2,
       'JWST PASSAGE': 124*4*u.arcmin**2,
       'JWST JADES': 65*u.arcmin**2,
       'JWST CEERS': 100*u.arcmin**2,
       'JWST NGDEEP': 10*u.arcmin**2,
}

SURVEY_DEPTHS= {
       'Rubin Wide': {'LSST_G': 25.}, 
       'Roman HLWAS': {'WFI_J129': 26.7},
       'Roman HLTDS': {'WFI_J129': 26.7},
       'Roman GBTDS': {'WFI_J129': 26.7},
       'Euclid Wide': {'EUCLID_J': 24. },
       'Rubin Deep': {'LSST_Z': 25.}, 
       'Euclid Deep': {'EUCLID_J': 27. },
       'JWST PASSAGE': {'NIRISS_F115W': 27},
       'JWST JADES': {'NIRISS_F115W': 29},
       'JWST CEERS':  {'NIRISS_F115W': 30},
       'JWST NGDEEP': {'NIRISS_F115W': 30},
}

#check the right pointings later
SURVEY_FOOTPRINT= {
       'Rubin Wide': SkyCoord(*popsims.random_angles(10)*u.radian), #random
       'Roman HLWAS': SkyCoord(*popsims.random_angles(10)*u.radian), #random
       'Roman HLTDS':SkyCoord(*popsims.random_angles(10)*u.radian), #random
       'Roman GBTDS': SkyCoord(*popsims.random_angles(10)*u.radian), #random
       'Euclid Wide': SkyCoord(*popsims.random_angles(10)*u.radian), #random
       'Rubin Deep': SkyCoord(*popsims.random_angles(1)*u.radian), #random
       'Euclid Deep': SkyCoord(*popsims.random_angles(1)*u.radian), #random
       'JWST PASSAGE': SkyCoord(*popsims.random_angles(1)*u.radian), #random
       'JWST JADES': SkyCoord(ra=53*u.degree, dec=-27.7*u.degree), #goods-sotu
       'JWST CEERS':  SkyCoord(*popsims.random_angles(1)*u.radian), #random
       'JWST NGDEEP':SkyCoord(*popsims.random_angles(1)*u.radian), #random
}


#filters to simulate 
#check the right pointings later
FILTERS= {
       'Rubin Wide':  ['LSST_'+x for x in 'G R I Z Y'.split()],
       'Roman HLWAS':  ['WFI_'+x for x in 'R062 Z087 Y106 J129 H158 F184 Prism Grism'.split()],
       'Roman HLTDS':  ['WFI_'+x for x in 'R062 Z087 Y106 J129 H158 F184 Prism Grism'.split()],
       'Roman GBTDS': ['WFI_'+x for x in 'R062 Z087 Y106 J129 H158 F184 Prism Grism'.split()],
       'Euclid Wide': ['EUCLID_'+x for x in 'Y J H'.split()],
       'Rubin Deep':  ['EUCLID_'+x for x in 'Y J H'.split()],
       'Euclid Deep': ['EUCLID_'+x for x in 'Y J H'.split()],
       'JWST PASSAGE':  ['NIRISS_'+x for x in 'F115W F200W F150W'.split()], #CHANGE LATER
       'JWST JADES': ['NIRISS_'+x for x in 'F115W F200W F150W'.split()],
       'JWST CEERS':  ['NIRISS_'+x for x in 'F115W F200W F150W'.split()],
       'JWST NGDEEP':['NIRISS_'+x for x in 'F115W F200W F150W'.split()], #CHANGE LATER
}


#filenames
FILENAMES={
       'Rubin Wide': 'rubin_wide',
       'Roman HLWAS': 'roman_hlwas',
       'Roman HLTDS': 'roman_hltds',
       'Roman GBTDS': 'roman_gbtds',
       'Euclid Wide': 'euclid_wide',
       'Rubin Deep': 'rubin_deep',
       'Euclid Deep': 'euclid_deep',
       'JWST PASSAGE': 'jwst_passage',
       'JWST JADES': 'jwst_jades',
       'JWST CEERS':  'jwst_ceers',
       'JWST NGDEEP': 'jwst_ngdeep'
}


def get_maximum_volumes(spt, kind, maglimits):
    #for all the filters, compute the largest volume and draw distances from that
    dmaxs=[]
    for k in maglimits.keys():
            mag_cut= maglimits[k]
            absmag= np.poly1d(POL['absmags_spt'][kind][k]['fit'])(spt)
            dmaxs.append(10.**(-(absmag-mag_cut)/5. + 1.))
    return np.nanmax(dmaxs)

def simulate_survey(keys, maglimit, footprint, filename):

    nsample=1e3
    sptgrid=np.arange(15, 40)
    dminss=0.1*np.ones_like(sptgrid)
    dmaxss=1.5*np.array([get_maximum_volumes(x, 'dwarfs', maglimit) for x  in sptgrid])
    dmaxss_sd=1.5*np.array([get_maximum_volumes(x, 'subdwarfs', maglimit) for x in sptgrid])
    dmaxss_esd=1.5*np.array([get_maximum_volumes(x,'esd', maglimit) for x in sptgrid])

    drange=dict(zip(sptgrid,np.vstack([dminss, dmaxss]).T ))

    p1=Population(evolmodel= 'burrows1997',
                  imf_power=-0.6,
                  binary_fraction=0.2,
                  age_range=[0, 8],
                  mass_range=[0.01, .1],
                 nsample=nsample)

    p1.simulate()

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


    p2.assign_distance_from_spt_ranges(Disk(H=900, L=3600), footprint.galactic.l.radian, footprint.galactic.b.radian, drange)

    
    p2.add_magnitudes(keys, get_from='spt',object_type='subdwarfs', pol=POL['absmags_spt']['subdwarfs'])
    p2.add_kinematics(footprint.ra.degree, footprint.dec.degree, kind='thick_disk', red_prop_motions_keys=keys)



    p3=Population(evolmodel= METAL_POOR_EVOL,
                  imf_power=-0.6,
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

    df2=p2.to_dataframe(cols)
    df2['population']='thick disk'

    df3=p3.to_dataframe(cols)
    df3['population']='halo'

    df=pd.concat([df1, df2, df3]).reset_index(drop=True)

    df=df[np.logical_and.reduce([df[k] <  maglimit[k] for k in maglimit.keys() ])].reset_index(drop=True)
        

    np.save('../simulations{}.npy'.format(filename), df, allow_pickle=True) 


survey='Rubin Wide'
simulate_survey(FILTERS[survey], SURVEY_DEPTHS[survey], SURVEY_FOOTPRINT[survey], FILENAMES[survey])