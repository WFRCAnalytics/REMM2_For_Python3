import sys
#from urbansim_defaults import datasources
import urbansim_defaults.datasources as datasources
#from urbansim_defaults import utils
from urbansim_wfrc.utils import misc
import orca_wfrc.orca as sim
import pandas as pd
import numpy as np
import os
import utils as wfrc_utils

@sim.table('valid_parcels', cache=True)
def valid_parcels(store):
    return store['valid_parcels']

@sim.table('parcels', cache=True)
def parcels(store):
    df = store['parcels']
    df = df.query('x>0')
    df = df.query('zone_id>0')
    df = df.query('zone_id<2882')
    df.zone_id = df.zone_id.astype('int')
    #row = pd.read_csv("./data/row_parcels.csv")
    #p = df[~df.index.isin(row.parcel_id)]
    return df
    
@sim.table('parcels_slc', cache=True)
def parcels_slc(parcels):
    p_col = parcels.local_columns + ['ave_sqft_per_unit', 'total_residential_units','total_job_spaces']
    df = parcels.to_frame(p_col)
    return df[df.county_id==2]
    
@sim.table('parcels_utah', cache=True)
def parcels_utah(parcels):
    p_col = parcels.local_columns + ['ave_sqft_per_unit', 'total_residential_units','total_job_spaces']
    df = parcels.to_frame(p_col)
    return df[df.county_id==4]
    
@sim.table('parcels_davis', cache=True)
def parcels_davis(parcels):
    p_col = parcels.local_columns + ['ave_sqft_per_unit', 'total_residential_units','total_job_spaces']
    df = parcels.to_frame(p_col)
    return df[df.county_id==1]

@sim.table('parcels_weber', cache=True)
def parcels_weber(parcels):
    p_col = parcels.local_columns + ['ave_sqft_per_unit', 'total_residential_units','total_job_spaces']
    df = parcels.to_frame(p_col)
    return df[df.county_id==3]
    
@sim.table('households_slc', cache=True)
def households_slc(households):
    df = households.local
    return df[df.cid==2]

@sim.table('households_utah', cache=True)
def households_utah(households):
    df = households.local
    return df[df.cid==4]
    
@sim.table('households_davis', cache=True)
def households_davis(households):
    df = households.local
    return df[df.cid==1]
    
@sim.table('households_weber', cache=True)
def households_weber(households):
    df = households.local
    return df[df.cid==3]

@sim.table('households_for_estimation', cache=True)
def households_for_estimation(store, buildings):
    df = store.households_for_estimation.dropna(subset=['building_id'])
    return df[df.building_id.isin(buildings.index)]
    
@sim.table('jobs_slc', cache=True)
def jobs_slc(jobs):
    df = jobs.to_frame(['cid','building_type_id'])
    return df[df.cid==2]

@sim.table('jobs_slc_ind')
def jobs_slc_ind(jobs):
    df = jobs.to_frame(['cid','building_type_id'])
    return df[(df.cid==2) & (df.building_type_id==3)]
    
@sim.table('jobs_slc_ret')
def jobs_slc_ret(jobs):
    df = jobs.to_frame(['cid','building_type_id'])
    return df[(df.cid==2) & (df.building_type_id==4)]
    
@sim.table('jobs_slc_ofc')
def jobs_slc_ofc(jobs):
    df = jobs.to_frame(['cid','building_type_id'])
    return df[(df.cid==2) & (df.building_type_id==5)]

@sim.table('jobs_davis_ind')
def jobs_davis_ind(jobs):
    df = jobs.to_frame(['cid','building_type_id'])
    return df[(df.cid==1) & (df.building_type_id==3)]
    
@sim.table('jobs_davis_ret')
def jobs_davis_ret(jobs):
    df = jobs.to_frame(['cid','building_type_id'])
    return df[(df.cid==1) & (df.building_type_id==4)]
    
@sim.table('jobs_davis_ofc')
def jobs_davis_ofc(jobs):
    df = jobs.to_frame(['cid','building_type_id'])
    return df[(df.cid==1) & (df.building_type_id==5)]

@sim.table('jobs_weber_ind')
def jobs_weber_ind(jobs):
    df = jobs.to_frame(['cid','building_type_id'])
    return df[(df.cid==3) & (df.building_type_id==3)]
    
@sim.table('jobs_weber_ret')
def jobs_weber_ret(jobs):
    df = jobs.to_frame(['cid','building_type_id'])
    return df[(df.cid==3) & (df.building_type_id==4)]
    
@sim.table('jobs_weber_ofc')
def jobs_weber_ofc(jobs):
    df = jobs.to_frame(['cid','building_type_id'])
    return df[(df.cid==3) & (df.building_type_id==5)]

@sim.table('jobs_utah_ind')
def jobs_utah_ind(jobs):
    df = jobs.to_frame(['cid','building_type_id'])
    return df[(df.cid==4) & (df.building_type_id==3)]
    
@sim.table('jobs_utah_ret')
def jobs_utah_ret(jobs):
    df = jobs.to_frame(['cid','building_type_id'])
    return df[(df.cid==4) & (df.building_type_id==4)]
    
@sim.table('jobs_utah_ofc')
def jobs_utah_ofc(jobs):
    df = jobs.to_frame(['cid','building_type_id'])
    return df[(df.cid==4) & (df.building_type_id==5)]


@sim.table('jobs_utah', cache=True)
def jobs_utah(jobs):
    df = jobs.to_frame(['cid','building_type_id'])
    return df[df.cid==4]

@sim.table('jobs_davis', cache=True)
def jobs_davis(jobs):
    df = jobs.to_frame(['cid','building_type_id'])
    return df[df.cid==1]

@sim.table('jobs_weber', cache=True)
def jobs_weber(jobs):
    df = jobs.to_frame(['cid','building_type_id'])
    return df[df.cid==3]
    
@sim.table('buildings_slc', cache=True)
def buildings_slc(buildings):
    b_col = buildings.local_columns + ['job_spaces','county_id']
    df = buildings.to_frame(b_col)
    df = df[df.county_id==2]
    return df.drop('county_id', axis=1)

@sim.table('buildings_slc_ind', cache=True)
def buildings_slc_ind(buildings):
    b_col = buildings.local_columns + ['job_spaces','county_id']
    df = buildings.to_frame(b_col)
    df = df[(df.county_id==2) & (df.building_type_id == 3)]
    return df.drop('county_id', axis=1)
    
@sim.table('buildings_slc_ret', cache=True)
def buildings_slc_ret(buildings):
    b_col = buildings.local_columns + ['job_spaces','county_id']
    df = buildings.to_frame(b_col)
    df = df[(df.county_id==2) & (df.building_type_id == 4)]
    return df.drop('county_id', axis=1)
    
@sim.table('buildings_slc_ofc', cache=True)
def buildings_slc_ofc(buildings):
    b_col = buildings.local_columns + ['job_spaces','county_id']
    df = buildings.to_frame(b_col)
    df = df[(df.county_id==2) & (df.building_type_id == 5)]
    return df.drop('county_id', axis=1)

@sim.table('buildings_utah', cache=True)
def buildings_utah(buildings):
    b_col = buildings.local_columns + ['job_spaces','county_id']
    df = buildings.to_frame(b_col)
    df = df[df.county_id==4]
    return df.drop('county_id', axis=1)
    
@sim.table('buildings_utah_ind', cache=True)
def buildings_utah_ind(buildings):
    b_col = buildings.local_columns + ['job_spaces','county_id']
    df = buildings.to_frame(b_col)
    df = df[(df.county_id==4) & (df.building_type_id == 3)]
    return df.drop('county_id', axis=1)
    
@sim.table('buildings_utah_ret', cache=True)
def buildings_utah_ret(buildings):
    b_col = buildings.local_columns + ['job_spaces','county_id']
    df = buildings.to_frame(b_col)
    df = df[(df.county_id==4) & (df.building_type_id == 4)]
    return df.drop('county_id', axis=1)
    
@sim.table('buildings_utah_ofc', cache=True)
def buildings_utah_ofc(buildings):
    b_col = buildings.local_columns + ['job_spaces','county_id']
    df = buildings.to_frame(b_col)
    df = df[(df.county_id==4) & (df.building_type_id == 5)]
    return df.drop('county_id', axis=1)
    
@sim.table('buildings_davis', cache=True)
def buildings_davis(buildings):
    b_col = buildings.local_columns + ['job_spaces','county_id']
    df = buildings.to_frame(b_col)
    df = df[df.county_id==1]
    return df.drop('county_id', axis=1)
    
@sim.table('buildings_davis_ind', cache=True)
def buildings_davis_ind(buildings):
    b_col = buildings.local_columns + ['job_spaces','county_id']
    df = buildings.to_frame(b_col)
    df = df[(df.county_id==1) & (df.building_type_id == 3)]
    return df.drop('county_id', axis=1)
    
@sim.table('buildings_davis_ret', cache=True)
def buildings_davis_ret(buildings):
    b_col = buildings.local_columns + ['job_spaces','county_id']
    df = buildings.to_frame(b_col)
    df = df[(df.county_id==1) & (df.building_type_id == 4)]
    return df.drop('county_id', axis=1)
    
@sim.table('buildings_davis_ofc', cache=True)
def buildings_davis_ofc(buildings):
    b_col = buildings.local_columns + ['job_spaces','county_id']
    df = buildings.to_frame(b_col)
    df = df[(df.county_id==1) & (df.building_type_id == 5)]
    return df.drop('county_id', axis=1)
    
@sim.table('buildings_weber', cache=True)
def buildings_weber(buildings):
    b_col = buildings.local_columns + ['job_spaces','county_id']
    df = buildings.to_frame(b_col)
    df = df[df.county_id==3]
    return df.drop('county_id', axis=1)
    
@sim.table('buildings_weber_ind', cache=True)
def buildings_weber_ind(buildings):
    b_col = buildings.local_columns + ['job_spaces','county_id']
    df = buildings.to_frame(b_col)
    df = df[(df.county_id==3) & (df.building_type_id == 3)]
    return df.drop('county_id', axis=1)
    
@sim.table('buildings_weber_ret', cache=True)
def buildings_weber_ret(buildings):
    b_col = buildings.local_columns + ['job_spaces','county_id']
    df = buildings.to_frame(b_col)
    df = df[(df.county_id==3) & (df.building_type_id == 4)]
    return df.drop('county_id', axis=1)
    
@sim.table('buildings_weber_ofc', cache=True)
def buildings_weber_ofc(buildings):
    b_col = buildings.local_columns + ['job_spaces','county_id']
    df = buildings.to_frame(b_col)
    df = df[(df.county_id==3) & (df.building_type_id == 5)]
    return df.drop('county_id', axis=1)

@sim.table('buildings_for_estimation', cache=True)
def buildings_for_estimation(store):
    df = store['buildings_for_estimation']
    return df.dropna(subset=['building_type_id']).query('building_type_id > 0')
    
@sim.table('buildings_for_estimation_grouped', cache=True)
def buildings_for_estimation_grouped(store):
    df = store['buildings_for_estimation_grouped']
    df.unit_price_non_residential[df.trans_year==2000] = (df.unit_price_non_residential[df.trans_year==2000] * 1.27)
    df.unit_price_non_residential[df.trans_year==2001] = (df.unit_price_non_residential[df.trans_year==2001] * 1.23)
    df.unit_price_non_residential[df.trans_year==2002] = (df.unit_price_non_residential[df.trans_year==2002] * 1.21)
    df.unit_price_non_residential[df.trans_year==2003] = (df.unit_price_non_residential[df.trans_year==2003] * 1.19)
    df.unit_price_non_residential[df.trans_year==2004] = (df.unit_price_non_residential[df.trans_year==2004] * 1.15)
    df.unit_price_non_residential[df.trans_year==2005] = (df.unit_price_non_residential[df.trans_year==2005] * 1.12)
    df.unit_price_non_residential[df.trans_year==2006] = (df.unit_price_non_residential[df.trans_year==2006] * 1.08)
    df.unit_price_non_residential[df.trans_year==2007] = (df.unit_price_non_residential[df.trans_year==2007] * 1.05)
    df.unit_price_non_residential[df.trans_year==2008] = (df.unit_price_non_residential[df.trans_year==2008] * 1.01)
    df.unit_price_non_residential[df.trans_year==2009] = (df.unit_price_non_residential[df.trans_year==2009] * 1.02)
    df.unit_price_non_residential[df.trans_year==2011] = (df.unit_price_non_residential[df.trans_year==2011] * 0.97)
    df.unit_price_non_residential[df.trans_year==2012] = (df.unit_price_non_residential[df.trans_year==2012] * 0.95)
    df.unit_price_non_residential[df.trans_year==2013] = (df.unit_price_non_residential[df.trans_year==2013] * 0.94)
    df.unit_price_non_residential[df.trans_year==2014] = (df.unit_price_non_residential[df.trans_year==2014] * 0.92)
    return df.dropna(subset=['building_type_id']).query('building_type_id > 0')
    
@sim.table('buildings', cache=True)
def buildings(store, households, jobs, parcels, building_sqft_per_job, settings):
    df = datasources.buildings(store, households, jobs,
                               building_sqft_per_job, settings)
    df = df[df.parcel_id.isin(parcels.index)]
    return df.query('building_type_id > 0')
    
@sim.table("zones", cache=True)
def zones(parcels):
    s = parcels.zone_id.value_counts()
    df = pd.DataFrame({"parcel_count": s.values}, index=s.index)
    df.index.name = 'zone_id'
    return df
'''
@sim.table("zones_2016", cache=True)
def zones_2016(parcels):
    s = parcels.zone_id_2016.value_counts()
    df = pd.DataFrame({"parcel_count": s.values}, index=s.index)
    df.index.name = 'zone_id_2016'
    return df
'''    
@sim.table("distlrg", cache=True)
def distlrg(parcels):
    p = parcels.to_frame(['distlrg_id','county_id','zone_id'])
    df = p.groupby('distlrg_id').agg({'county_id':'median','zone_id':'count'})
    df = df.rename(columns={'zone_id':'parcel_count'})
    return df
    
@sim.table("distmed", cache=True)
def distmed(parcels):
    p = parcels.to_frame(['distmed_id','county_id','zone_id'])
    df = p.groupby('distmed_id').agg({'county_id':'median','zone_id':'count'})
    df = df.rename(columns={'zone_id':'parcel_count'})
    return df
    
@sim.table("distsml", cache=True)
def distsml(parcels):
    p = parcels.to_frame(['distsml_id','county_id','zone_id'])
    df = p.groupby('distsml_id').agg({'county_id':'median','zone_id':'count'})
    df = df.rename(columns={'zone_id':'parcel_count'})
    return df
    
@sim.table("blocks", cache=True)
def blocks(parcels):
    b = parcels.block_id.value_counts()
    df = pd.DataFrame({"parcel_count": b.values}, index=b.index)
    df.index.name = 'block_id'
    return df

@sim.table('travel_data', cache=True)
def travel_data(store):
    df = store['travel_data']
    return df
    
@sim.table('zoning', cache=True)
def zoning(store):
    df = store['zoning']
    return df
    
# this is the mapping of parcels to zoning attributes
@sim.table('zoning_for_parcels', cache=True)
def zoning_for_parcels(store):
    df = store['zoning_for_parcels']
    df = df.reset_index().drop_duplicates(subset='parcel').set_index('parcel')
    return df

# zoning for use in the "baseline" scenario
# comes in the hdf5
@sim.table('zoning_baseline', cache=True)
def zoning_baseline(store, settings, year):
#    df = pd.merge(zoning_for_parcels.to_frame(),
#                  zoning.to_frame(),
#                  left_on='zoning',
#                  right_index=True)
    df = store['zoning_baseline']
    #if os.path.exists(os.path.join(misc.data_dir(), "zoning_parcels.csv")):
    #    df['parcel_id'] = df.index
    #    alter = pd.read_csv(os.path.join(misc.data_dir(), "zoning_parcels.csv"), index_col='parcel_id')
    #    df = pd.merge(df, alter, how='left', left_index=True, right_index=True, suffixes=('','_x'))
    #    df.max_dua[df.max_dua_x.notnull()] = df.max_dua_x[df.max_dua_x.notnull()]
    #    df.max_far[df.max_far_x.notnull()] = df.max_far_x[df.max_far_x.notnull()]
    #    df = df.drop(['max_dua_x','max_far_x'], axis=1)
    if os.path.exists(os.path.join(misc.data_dir(), "scenario_inputs", settings['scenario'], "zoning_parcels_p.csv")):
        update = pd.read_csv(os.path.join(misc.data_dir(), "scenario_inputs", settings['scenario'], "zoning_parcels_p.csv"))
        update = update[update.year<=year]
        update = update.sort_values(by='year', ascending=1)
        update = update.drop_duplicates("parcel_id","last")
        if update.empty:
            df.max_height = 999
            return df
        df2 = pd.merge(df, update, how='left', left_index=True, right_on='parcel_id', suffixes=('','_x'))
        df2.set_index('parcel_id', inplace=True)
        df2.max_dua[df2.max_dua_x.notnull()] = df2.max_dua_x[df2.max_dua_x.notnull()]
        df2.max_dua[df2.max_dua==0] = np.nan
        df2.max_far[df2.max_far_x.notnull()] = df2.max_far_x[df2.max_far_x.notnull()]
        df2.max_far[df2.max_far==0] = np.nan
        df2.type1[df2.type1_x==1] = 't'
        df2.type1[df2.type1_x==0] = 'f'
        df2.type2[df2.type2_x==1] = 't'
        df2.type2[df2.type2_x==0] = 'f'
        df2.type3[df2.type3_x==1] = 't'
        df2.type3[df2.type3_x==0] = 'f'
        df2.type4[df2.type4_x==1] = 't'
        df2.type4[df2.type4_x==0] = 'f'
        df2.type5[df2.type5_x==1] = 't'
        df2.type5[df2.type5_x==0] = 'f'
        df2.type6[df2.type6_x==1] = 't'
        df2.type6[df2.type6_x==0] = 'f'
        df2.type7[df2.type7_x==1] = 't'
        df2.type7[df2.type7_x==0] = 'f'
        df2.type8[df2.type8_x==1] = 't'
        df2.type8[df2.type8_x==0] = 'f'
        df2 = df2.drop(['year','max_dua_x','max_far_x','type1_x','type2_x','type3_x','type4_x','type5_x','type6_x','type7_x','type8_x'], axis=1)
        df2.max_height = 999
        
        if os.path.exists(os.path.join(misc.data_dir(), "developableparcels.dbf")):
            devbuffer = wfrc_utils.dbf2df(os.path.join(misc.data_dir(), "developableparcels.dbf"))
            undevbuffer = devbuffer.parcel_id.unique()
            df2.type1[df2.index.isin(undevbuffer)] = 'f'
            df2.type2[df2.index.isin(undevbuffer)] = 'f'
            df2.type4[df2.index.isin(undevbuffer)] = 'f'
            df2.type5[df2.index.isin(undevbuffer)] = 'f'
        return df2
    else:
        df.max_height = 999
        return df
    
@sim.table('households', cache=True)
def households(store, settings):
    df = store['households']
    p = store['parcels']
    if settings.get("remove_invalid_building_ids", True):
        # have to do it this way to prevent circular reference
        df.building_id.loc[~df.building_id.isin(store['buildings'].index)] = -1

    fill_nas_cfg = settings.get("table_reprocess", None)
    if fill_nas_cfg is not None:
	    fill_nas_cfg = fill_nas_cfg.get("households", None)
    if fill_nas_cfg is not None:
        df = utils.table_reprocess(fill_nas_cfg, df)

    return df[df.cid.notnull()]

@sim.injectable('homesales')
def homesales(buildings):
    return buildings
    
@sim.injectable("naics_to_empsix")
def naics_to_empsix(settings):
    return settings["naics_to_empsix"]
    
@sim.injectable('sector_id_to_desc', cache=True)
def building_sqft_per_job(settings):
    return settings['sector_id_to_desc']
    
@sim.injectable("summary", cache=True)
def simulation_summary_data(run_number):
    return wfrc_utils.SimulationSummaryData(run_number)
    

# this specifies the relationships between tables
sim.broadcast('zones', 'buildings', cast_index=True, onto_on='zone_id')
sim.broadcast('zones', 'parcels', cast_index=True, onto_on='zone_id')
sim.broadcast('nodes', 'buildings_for_estimation', cast_index=True, onto_on='node_id')
sim.broadcast('parcels', 'buildings_for_estimation', cast_index=True, onto_on='parcel_id')
sim.broadcast('zones', 'buildings_for_estimation', cast_index=True, onto_on='zone_id')
sim.broadcast('nodes', 'buildings_for_estimation_grouped', cast_index=True, onto_on='node_id')
sim.broadcast('parcels', 'buildings_for_estimation_grouped', cast_index=True, onto_on='parcel_id')
sim.broadcast('zones', 'buildings_for_estimation_grouped', cast_index=True, onto_on='zone_id')
sim.broadcast('nodes', 'households_for_estimation', cast_index=True, onto_on='node_id')
sim.broadcast('parcels', 'households_for_estimation', cast_index=True, onto_on='parcel_id')
sim.broadcast('zones', 'households_for_estimation', cast_index=True, onto_on='zone_id')
sim.broadcast('zones', 'jobs', cast_index=True, onto_on='zone_id')
sim.broadcast('nodes', 'jobs', cast_index=True, onto_on='node_id')
sim.broadcast('nodes', 'households', cast_index=True, onto_on='node_id')
