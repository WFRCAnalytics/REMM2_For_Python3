import pandas as pd, numpy as np
from urbansim_defaults import variables, utils
from urbansim.utils import misc
import orca.orca as sim
import utils as wfrc_utils

@sim.column('jobs', 'building_type_id')
def building_type_id(jobs, buildings):
    df = misc.reindex(buildings.building_type_id, jobs.building_id)
    return df.fillna(-1).astype('int')

##################
# PARCELS VARIABLES
##################

@sim.column('parcels', 'ave_sqft_per_unit')
def ave_sqft_per_unit(parcels, nodes, settings, zoning_baseline):
    if len(nodes) == 0:
        # if nodes isn't generated yet
        return pd.Series(index=parcels.index)
    s = misc.reindex(nodes.ave_sqft_per_unit, parcels.node_id)
    clip = settings.get("ave_sqft_per_unit_clip", None)
    if clip is not None:
        s = pd.concat([s, zoning_baseline.type2], axis=1)
        s.right[s.type2=='t'] = s.right[s.type2=='t'].clip(lower=clip['lower'], upper=clip['mf_upper'])
        s.right[s.type2=='f'] = s.right[s.type2=='f'].clip(lower=clip['lower'], upper=clip['sf_upper'])
         
    return s.right.fillna(clip['lower'])

@sim.column("parcels", "parcel_acres")
def parcel_acres(parcels):
    return parcels.local.parcel_acres
    
@sim.column('parcels', 'node_id')
def node_id(parcels, net):
	return net.get_node_ids(parcels.x, parcels.y)
	
@sim.column('parcels', 'lot_size_per_unit')
def log_size_per_unit(parcels):
    ls = parcels.parcel_size / parcels.total_residential_units.replace(0, 1)
    return ls.fillna(0)
    
@sim.column('parcels', 'max_far', cache=True)
def max_far(parcels, scenario, scenario_inputs):
    far = utils.conditional_upzone(scenario, scenario_inputs, "max_far", "far_up").reindex(parcels.index)
    #downtownTAZ = pd.read_csv(r"DOWNTOWNTAZ\DOWNTOWNTAZ.csv")
    #far[(-parcels.zone_id.isin(list(downtownTAZ.TAZID))) & (far > 0.5)] = 0.5
    #far[parcels.county_id == 4] = far[parcels.county_id == 4] * 0.8
    return far
    #far[parcels.parcel_size > 3000000]  = far[parcels.parcel_size > 3000000]* 0.5

@sim.column('parcels', 'max_dua', cache=True)
def max_dua(parcels, year, scenario, scenario_inputs):
    dua = utils.conditional_upzone(scenario, scenario_inputs, "max_dua", "dua_up").reindex(parcels.index)
    dua[(parcels.total_residential_units < 1) & (parcels.total_job_spaces < 1) & (parcels.county_id != 1) & (parcels.county_id != 2)] = dua[(parcels.total_residential_units < 1) & (parcels.total_job_spaces < 1) & (parcels.county_id != 1) & (parcels.county_id != 2)]*0.8
    
    #Salt Lake County 10% increase for base year
    dua[parcels.county_id == 2] = dua[parcels.county_id == 2]*1.1
    
    #Davis County 2030 20% increase of capacity. 2040 40% increase of capacity
    if year >= 2025 and year < 2035:
        dua[parcels.county_id == 1] = dua[parcels.county_id == 1]*1.2
    elif year >=2035:
        dua[parcels.county_id == 1] = dua[parcels.county_id == 1]*1.45
    
    return dua
        
@sim.column('parcels', 'max_height', cache=True)
def max_height(parcels, zoning_baseline):
    return zoning_baseline.max_height.reindex(parcels.index).fillna(0)

@sim.column('parcels', 'residential_purchase_price_sqft')
def residential_purchase_price_sqft(parcels):
    return parcels.building_purchase_price_sqft

@sim.column('parcels', 'residential_sales_price_sqft')
def residential_sales_price_sqft(parcel_sales_price_sqft_func):
    return parcel_sales_price_sqft_func("residential")

# for debugging reasons this is split out into its own function
@sim.column('parcels', 'building_purchase_price_sqft')
def building_purchase_price_sqft():
    return parcel_average_price("residential") * 1

@sim.column('parcels', 'building_purchase_price')
def building_purchase_price(parcels):
    return (parcels.total_sqft * parcels.building_purchase_price_sqft).\
        reindex(parcels.index).fillna(0)
        
"""@sim.column('parcels', 'row')
def row(parcels):
    r = pd.read_csv("./data/row_parcels.csv")
    p = parcels.index.isin(r.parcel_id)
    return p.astype('int')"""

@sim.column('parcels', 'redev_friction')
def redev_friction(parcels,year):
    s = pd.read_csv("./data/redev_friction.csv", index_col="parcel_id")
    sredev = s.redev_friction.reindex(parcels.index).fillna(5)
    sredev[parcels.county_id == 4] = 10
    if year >=2035:
        sredev[parcels.county_id == 1] = sredev[parcels.county_id == 1]/2
    return sredev

@sim.column('parcels', 'parcel_volume')
def parcel_volume(parcels, year, settings):
    if year < 2020:
        file = settings['tdm']['main_dir'] + '2015/6_REMM/volumepoint.dbf'
        s = wfrc_utils.dbf2df(file).set_index('parcel_id')
        svolume = s.RASTERVALU.reindex(parcels.index).fillna(0)
    elif year in range(2020,2028):
        file = settings['tdm']['main_dir'] + '2022/6_REMM/volumepoint.dbf'
        s = wfrc_utils.dbf2df(file).set_index('parcel_id')
        svolume = s.RASTERVALU.reindex(parcels.index).fillna(0)
    elif year in range(2028,2036):
        file = settings['tdm']['main_dir'] + '2027/6_REMM/volumepoint.dbf'
        s = wfrc_utils.dbf2df(file).set_index('parcel_id')
        svolume = s.RASTERVALU.reindex(parcels.index).fillna(0)
    #elif year in range(2031,2035):
    #    file = settings['tdm']['main_dir'] + '2030/' + settings['tdm']['output_distance']
    #    p = wfrc_utils.dbf2df(file).set_index('parcel_id')
    elif year in range(2036,2046):
        file = settings['tdm']['main_dir'] + '2035/6_REMM/volumepoint.dbf'
        s = wfrc_utils.dbf2df(file).set_index('parcel_id')
        svolume = s.RASTERVALU.reindex(parcels.index).fillna(0)
    elif year in range(2046,2099):
        file = settings['tdm']['main_dir'] + '2045/6_REMM/volumepoint.dbf'
        s = wfrc_utils.dbf2df(file).set_index('parcel_id')
        svolume = s.RASTERVALU.reindex(parcels.index).fillna(0)
    else:
        file = settings['tdm']['main_dir'] + '2011/6_REMM/volumepoint.dbf'
        s = wfrc_utils.dbf2df(file).set_index('parcel_id')
        svolume = s.RASTERVALU.reindex(parcels.index).fillna(0)
    return svolume



@sim.column('parcels', 'gridID')
def gridID(parcels):
    s = pd.read_csv("./data/parcelgridID.csv", index_col="parcel_id")
    return s.GRIDID.reindex(parcels.index).fillna(0)

@sim.column('parcels', 'agriculture')
def agriculture(parcels):
    s = pd.read_csv("./data/agriculture.csv", index_col="parcel_id")
    return s.Agriculture.reindex(parcels.index).fillna(0)

@sim.column('parcels', 'land_cost')
def land_cost(parcels):
    result = parcels.building_purchase_price + parcels.land_value
    result[parcels.building_purchase_price > 0] =  result[parcels.building_purchase_price > 0] * parcels.redev_friction
    return result
    
@sim.column('parcels', 'distlrg_res_shift')
def distlrg_res_shift(parcels):
    s = pd.read_csv("./data/calibration/distlrg_shifters.csv", index_col="distlrg_id")
    return misc.reindex(s.res_price_shifter, parcels.distlrg_id)
    
@sim.column('parcels', 'distlrg_nonres_shift')
def distlrg_nonres_shift(parcels):
    s = pd.read_csv("./data/calibration/distlrg_shifters.csv", index_col="distlrg_id")
    return misc.reindex(s.nonres_price_shifter, parcels.distlrg_id)
    
@sim.column('parcels', 'avg_building_age')
def avg_building_age(parcels, buildings):
    return buildings.building_age.groupby(buildings.parcel_id).mean().\
        reindex(parcels.index).fillna(-99)
        
@sim.column('parcels', 'bus_stop_dist', cache=True)
def bus_stop_dist(parcels, year, settings):
    if year < 2020:
        file = settings['tdm']['main_dir'] + '2015/' + settings['tdm']['output_distance']
        p = wfrc_utils.dbf2df(file).set_index('parcel_id')
    elif year in range(2020,2028):
        file = settings['tdm']['main_dir'] + '2022/' + settings['tdm']['output_distance']
        p = wfrc_utils.dbf2df(file).set_index('parcel_id')
    elif year in range(2028,2036):
        file = settings['tdm']['main_dir'] + '2027/' + settings['tdm']['output_distance']
        p = wfrc_utils.dbf2df(file).set_index('parcel_id')
    #elif year in range(2031,2035):
    #    file = settings['tdm']['main_dir'] + '2030/' + settings['tdm']['output_distance']
    #    p = wfrc_utils.dbf2df(file).set_index('parcel_id')
    elif year in range(2036,2046):
        file = settings['tdm']['main_dir'] + '2035/' + settings['tdm']['output_distance']
        p = wfrc_utils.dbf2df(file).set_index('parcel_id')
    elif year in range(2046,2099):
        file = settings['tdm']['main_dir'] + '2045/' + settings['tdm']['output_distance']
        p = wfrc_utils.dbf2df(file).set_index('parcel_id')
    else:
        file = settings['tdm']['main_dir'] + '2011/' + settings['tdm']['output_distance']
        p = wfrc_utils.dbf2df(file).set_index('parcel_id')
    return misc.reindex(p.dist2busst, parcels.parent_parcel)
    #return p.dist2busst

@sim.column('parcels', 'fwy_exit_dist_tdm_output', cache=True)
def fwy_exit_dist_tdm_output(parcels, year, settings):
    if year < 2020:
        file = settings['tdm']['main_dir'] + '2015/' + settings['tdm']['output_fwy']
        ff = wfrc_utils.dbf2df(file)
    elif year in range(2020,2028):
        file = settings['tdm']['main_dir'] + '2022/' + settings['tdm']['output_fwy']
        ff = wfrc_utils.dbf2df(file)
    elif year in range(2028,2036):
        file = settings['tdm']['main_dir'] + '2027/' + settings['tdm']['output_fwy']
        ff = wfrc_utils.dbf2df(file)
    #elif year in range(2031,2035):
    #    file = settings['tdm']['main_dir'] + '2030/' + settings['tdm']['output_fwy']
    #    ff = wfrc_utils.dbf2df(file).set_index('parcel_id')
    elif year in range(2036,2046):
        file = settings['tdm']['main_dir'] + '2035/' + settings['tdm']['output_fwy']
        ff = wfrc_utils.dbf2df(file)
    elif year in range(2046,2099):
        file = settings['tdm']['main_dir'] + '2045/' + settings['tdm']['output_fwy']
        ff = wfrc_utils.dbf2df(file)
    else:
        file = settings['tdm']['main_dir'] + '2011/' + settings['tdm']['output_fwy']
        ff = wfrc_utils.dbf2df(file)
    p = sim.get_table('parcels')
    p = p.to_frame(['x', 'y'])
    fe = ff[(ff.FT == 42) | (ff.FT == 29)]
    nc = pd.read_csv("data/TDMNODE.csv")
    feloc  = pd.merge(fe, nc, how='left', left_on="B", right_on="N", suffixes=('','_x'))
    pdist = np.sqrt((p.x-feloc.loc[0].URBANSIMX)**2 + (p.y-feloc.loc[0].URBANSIMY)**2)
    for i in range(1,len(feloc)):
        pdistx = np.sqrt((p.x-feloc.loc[i].URBANSIMX)**2 + (p.y-feloc.loc[i].URBANSIMY)**2)
        pdist[pdistx < pdist] = pdistx[pdistx < pdist]
    return pdist*0.00018939

@sim.column('parcels', 'rail_stn_dist', cache=True)
def rail_stn_dist(parcels, year, settings):
    if year < 2020:
        file = settings['tdm']['main_dir'] + '2015/' + settings['tdm']['output_distance']
        p = wfrc_utils.dbf2df(file).set_index('parcel_id')
    elif year in range(2020,2028):
        file = settings['tdm']['main_dir'] + '2022/' + settings['tdm']['output_distance']
        p = wfrc_utils.dbf2df(file).set_index('parcel_id')
    elif year in range(2028,2036):
        file = settings['tdm']['main_dir'] + '2027/' + settings['tdm']['output_distance']
        p = wfrc_utils.dbf2df(file).set_index('parcel_id')
    #elif year in range(2031,2035):
    #    file = settings['tdm']['main_dir'] + '2030/' + settings['tdm']['output_distance']
    #    p = wfrc_utils.dbf2df(file).set_index('parcel_id')
    elif year in range(2036,2046):
        file = settings['tdm']['main_dir'] + '2035/' + settings['tdm']['output_distance']
        p = wfrc_utils.dbf2df(file).set_index('parcel_id')
    elif year in range(2046,2099):
        file = settings['tdm']['main_dir'] + '2045/' + settings['tdm']['output_distance']
        p = wfrc_utils.dbf2df(file).set_index('parcel_id')
    else:
        file = settings['tdm']['main_dir'] + '2011/' + settings['tdm']['output_distance']
        p = wfrc_utils.dbf2df(file).set_index('parcel_id')
    return misc.reindex(p.dist2rails, parcels.parent_parcel)
    #return p.dist2rails
          
# these are actually functions that take parameters, but are parcel-related
# so are defined here

@sim.injectable('parcel_median_price', autocall=False)
def parcel_median_price(use):
    btypes = sim.settings['form_to_btype'][use]
    rpm = sim.settings['res_price_multiplier']
    nrpm = sim.settings['nonres_price_multiplier']
    b = sim.get_table('buildings')
    p = sim.get_table('parcels')
    if use == "residential":
        distmed = []
        zones = []
        bframe = b.to_frame(['zone_id','building_type_id','res_price_per_sqft','unit_price_non_residential','distmed_id','residential_units'])
        zonesum = bframe.groupby('zone_id').agg({'residential_units':'sum'})
        tazmax = p.zone_id.max()
        zonesum = zonesum.reindex(index = range(1,tazmax+1), fill_value = 0)
        for zone, units in zonesum.iterrows():
            if int(units) > 20:
                zones.append(zone)
            else:
                distmed.append(zone)
        prz = misc.reindex(b.res_price_per_sqft[(b.zone_id.isin(zones))&(b.building_type_id.isin(btypes))&(b.res_price_per_sqft>0)&(b.res_price_per_sqft < 400)].groupby(b.zone_id).quantile(0.75), p.zone_id)
        prd = misc.reindex(b.res_price_per_sqft[(b.zone_id.isin(distmed))&(b.building_type_id.isin(btypes))&(b.res_price_per_sqft>0)&(b.res_price_per_sqft < 400)].groupby(b.distmed_id).quantile(0.75), p.distmed_id[p.zone_id.isin(distmed)])
        pr = pd.concat([prz[~np.isnan(prz)], prd[~np.isnan(prd)]])
        pr = pr.reindex(index = range(1,p.index.max()+1),fill_value = 50)
        pr = np.clip(pr, 15, 400)*rpm
        return pr * 0.1646 * 2
    elif use == "industrial":
        pr = misc.reindex(b.unit_price_non_residential[(b.building_type_id.isin(btypes))&(b.unit_price_non_residential>0)&(b.unit_price_non_residential < 100)].groupby(b.zone_id).median() * nrpm, p.zone_id).fillna(6)*2
        pr = pr.reindex(index = range(1,p.index.max()+1),fill_value = 12)
        pr = np.clip(pr, 12, 100)
        return pr * 0.6488 * 2
    elif use == "retail":
        pr = misc.reindex(b.unit_price_non_residential[(b.building_type_id.isin(btypes))&(b.unit_price_non_residential>0)&(b.unit_price_non_residential < 500)].groupby(b.zone_id).median() * nrpm, p.zone_id).fillna(120)
        pr = pr.reindex(index = range(1,p.index.max()+1),fill_value = 120)
        pr = np.clip(pr, 120, 500)
        return pr * 0.1111 * 2
    else:
        pr = misc.reindex(b.unit_price_non_residential[(b.building_type_id.isin(btypes))&(b.unit_price_non_residential>0)&(b.unit_price_non_residential < 500)].groupby(b.zone_id).median() * nrpm, p.zone_id).fillna(120)
        pr = pr.reindex(index = range(1,p.index.max()+1),fill_value = 120)
        pr = np.clip(pr, 120, 500)
        return pr * 0.1138 * 2
        
@sim.injectable('parcel_average_price', autocall=False)
def parcel_average_price(use):
    btypes = sim.settings['form_to_btype'][use]
    b = sim.get_table('buildings')
    if use == "residential":
        return misc.reindex(b.res_price_per_sqft[b.building_type_id.isin(btypes)].groupby(b.zone_id).mean(), sim.get_table('parcels').zone_id).fillna(b.res_price_per_sqft[b.building_type_id.isin(btypes)].mean())
    else:
        return misc.reindex(b.unit_price_non_residential[b.building_type_id.isin(btypes)].groupby(b.zone_id).mean(), sim.get_table('parcels').zone_id).fillna(b.unit_price_non_residential[b.building_type_id.isin(btypes)].mean())

@sim.injectable('parcel_sales_price_sqft_func', autocall=False)
def parcel_sales_price_sqft(use):
    return parcel_median_price(use)

@sim.injectable('parcel_is_allowed_func', autocall=False)
def parcel_is_allowed(form):
    settings = sim.get_injectable("settings")
    form_to_btype = settings["form_to_btype"]
    # we have zoning by building type but want
    # to know if specific forms are allowed
    allowed = [sim.get_table('zoning_baseline')
               ['type%d' % typ] == 't' for typ in form_to_btype[form]]
    return pd.concat(allowed, axis=1).max(axis=1).\
        reindex(sim.get_table('parcels').index).fillna(False)
        

####################
#BUILDINGS VARIABLES
####################
@sim.column('buildings', 'job_spaces', cache=True)
def job_spaces(buildings):
    return buildings.local.job_spaces


@sim.column('buildings', 'zone_id', cache=True)
def zone_id(buildings, parcels):
    df =  misc.reindex(parcels.zone_id, buildings.parcel_id)
    return df.fillna(0).astype('int')

@sim.column('buildings', 'node_id', cache=True)
def node_id(buildings, parcels):
    df =  misc.reindex(parcels.node_id, buildings.parcel_id)
    return df.fillna(0)
    
@sim.column('buildings', 'county_id')
def county_id(buildings, parcels):
    df = misc.reindex(parcels.county_id, buildings.parcel_id)
    return df.fillna(0)
    
@sim.column('buildings', 'distlrg_id')
def distlrg_id(buildings, parcels):
    df = misc.reindex(parcels.distlrg_id, buildings.parcel_id)
    return df.fillna(0)
    
@sim.column('buildings', 'distmed_id')
def distmed_id(buildings, parcels):
    df = misc.reindex(parcels.distmed_id, buildings.parcel_id)
    return df.fillna(0)
    
@sim.column('buildings', 'distsml_id')
def distsml_id(buildings, parcels):
    df = misc.reindex(parcels.distsml_id, buildings.parcel_id)
    return df.fillna(0)
    
@sim.column('buildings', 'distmed_1')
def distmed_1(buildings):
    return (buildings.distmed_id==1).astype('int32')

@sim.column('buildings', 'distmed_2')
def distmed_2(buildings):
    return (buildings.distmed_id==2).astype('int32')

@sim.column('buildings', 'distmed_3')
def distmed_3(buildings):
    return (buildings.distmed_id==3).astype('int32')

@sim.column('buildings', 'distmed_4')
def distmed_4(buildings):
    return (buildings.distmed_id==4).astype('int32')

@sim.column('buildings', 'distmed_5')
def distmed_5(buildings):
    return (buildings.distmed_id==5).astype('int32')

@sim.column('buildings', 'distmed_6')
def distmed_6(buildings):
    return (buildings.distmed_id==6).astype('int32')

@sim.column('buildings', 'distmed_7')
def distmed_7(buildings):
    return (buildings.distmed_id==7).astype('int32')

@sim.column('buildings', 'distmed_8')
def distmed_8(buildings):
    return (buildings.distmed_id==8).astype('int32')

@sim.column('buildings', 'distmed_9')
def distmed_9(buildings):
    return (buildings.distmed_id==9).astype('int32')
    
@sim.column('buildings', 'distmed_10')
def distmed_10(buildings):
    return (buildings.distmed_id==10).astype('int32')

@sim.column('buildings', 'distmed_11')
def distmed_11(buildings):
    return (buildings.distmed_id==11).astype('int32')

@sim.column('buildings', 'distmed_12')
def distmed_12(buildings):
    return (buildings.distmed_id==12).astype('int32')

@sim.column('buildings', 'distmed_13')
def distmed_13(buildings):
    return (buildings.distmed_id==13).astype('int32')

@sim.column('buildings', 'distmed_14')
def distmed_14(buildings):
    return (buildings.distmed_id==14).astype('int32')

@sim.column('buildings', 'distmed_15')
def distmed_15(buildings):
    return (buildings.distmed_id==15).astype('int32')

@sim.column('buildings', 'distmed_16')
def distmed_16(buildings):
    return (buildings.distmed_id==16).astype('int32')

@sim.column('buildings', 'distmed_17')
def distmed_17(buildings):
    return (buildings.distmed_id==17).astype('int32')

@sim.column('buildings', 'distmed_18')
def distmed_18(buildings):
    return (buildings.distmed_id==18).astype('int32')

@sim.column('buildings', 'distmed_19')
def distmed_19(buildings):
    return (buildings.distmed_id==19).astype('int32')
    
@sim.column('buildings', 'distmed_20')
def distmed_20(buildings):
    return (buildings.distmed_id==20).astype('int32')
    
@sim.column('buildings', 'distmed_21')
def distmed_21(buildings):
    return (buildings.distmed_id==21).astype('int32')

@sim.column('buildings', 'distmed_22')
def distmed_22(buildings):
    return (buildings.distmed_id==22).astype('int32')

@sim.column('buildings', 'distmed_23')
def distmed_23(buildings):
    return (buildings.distmed_id==23).astype('int32')

@sim.column('buildings', 'distmed_24')
def distmed_24(buildings):
    return (buildings.distmed_id==24).astype('int32')

@sim.column('buildings', 'distmed_25')
def distmed_25(buildings):
    return (buildings.distmed_id==25).astype('int32')

@sim.column('buildings', 'distmed_26')
def distmed_26(buildings):
    return (buildings.distmed_id==26).astype('int32')

@sim.column('buildings', 'distmed_27')
def distmed_27(buildings):
    return (buildings.distmed_id==27).astype('int32')

@sim.column('buildings', 'distmed_28')
def distmed_28(buildings):
    return (buildings.distmed_id==28).astype('int32')

@sim.column('buildings', 'distmed_29')
def distmed_29(buildings):
    return (buildings.distmed_id==29).astype('int32')
    
@sim.column('buildings', 'distmed_30')
def distmed_30(buildings):
    return (buildings.distmed_id==30).astype('int32')
    
@sim.column('buildings', 'distmed_31')
def distmed_31(buildings):
    return (buildings.distmed_id==31).astype('int32')

@sim.column('buildings', 'distmed_32')
def distmed_32(buildings):
    return (buildings.distmed_id==32).astype('int32')

@sim.column('buildings', 'distmed_33')
def distmed_33(buildings):
    return (buildings.distmed_id==33).astype('int32')

@sim.column('buildings', 'distmed_34')
def distmed_34(buildings):
    return (buildings.distmed_id==34).astype('int32')

@sim.column('buildings', 'distmed_35')
def distmed_35(buildings):
    return (buildings.distmed_id==35).astype('int32')

@sim.column('buildings', 'distmed_36')
def distmed_36(buildings):
    return (buildings.distmed_id==36).astype('int32')

@sim.column('buildings', 'distmed_37')
def distmed_37(buildings):
    return (buildings.distmed_id==37).astype('int32')

@sim.column('buildings', 'distmed_38')
def distmed_38(buildings):
    return (buildings.distmed_id==38).astype('int32')

@sim.column('buildings', 'distmed_39')
def distmed_39(buildings):
    return (buildings.distmed_id==39).astype('int32')
    
@sim.column('buildings', 'distmed_40')
def distmed_40(buildings):
    return (buildings.distmed_id==40).astype('int32')
    
@sim.column('buildings', 'distmed_41')
def distmed_41(buildings):
    return (buildings.distmed_id==41).astype('int32')
    
@sim.column('buildings', 'zone_1212')
def zone_1212(buildings):
    return (buildings.zone_id==1212).astype('int32')
    
@sim.column('buildings', 'building_age', cache=True, cache_scope='iteration')
def building_age(buildings, year):
    if year > 2010:
        df = (year - buildings.year_built).clip(lower=0)
        df[df > 200] = 30
        return df
    else:
        df = (2010 - buildings.year_built).clip(lower=0)
        df[df > 200] = 30
        return df
        
"""@sim.column("buildings", "res_price_per_sqft", cache=False)
def res_price_per_sqft(buildings, year):
    return buildings.unit_price_residential / buildings.sqft_per_unit.replace(0, 1)"""

@sim.column("buildings", "building_sqft", cache=True)
def building_sqft(buildings):
    return buildings.local.building_sqft.clip(upper=5000000)

@sim.column("buildings", "real_far", cache=True)
def real_far(buildings):
    df = buildings.building_sqft.fillna(0)/(buildings.parcel_acres.fillna(0.001)*43560)
    df = df.replace(np.inf, 0)
    df = df.clip(upper=50)
    return df.fillna(0)
    
@sim.column("buildings", "land_value", cache=True)
def land_value(buildings, parcels):
    df = misc.reindex(parcels.land_value, buildings.parcel_id)
    return df.fillna(0)

@sim.column("buildings", "utmxi", cache=True)
def utmxi(buildings, parcels):
    df = misc.reindex(parcels.utmxi, buildings.parcel_id)
    return df.fillna(0)

@sim.column("buildings", "utmyi", cache=True)
def utmyi(buildings, parcels):
    df = misc.reindex(parcels.utmyi, buildings.parcel_id)
    return df.fillna(0)

@sim.column('buildings', 'avg_building_age')
def avg_building_age(parcels, buildings):
    df = misc.reindex(parcels.avg_building_age, buildings.parcel_id)
    return df.fillna(0)
    
@sim.column("buildings", "parcel_acres", cache=True)
def parcel_acres(buildings, parcels):
    df = misc.reindex(parcels.parcel_acres, buildings.parcel_id)
    return df.fillna(0)

@sim.column("buildings", "parcel_volume", cache=True)
def parcel_volume(buildings, parcels):
    df = misc.reindex(parcels.parcel_volume, buildings.parcel_id)
    return df.fillna(0)

@sim.column('buildings', 'elevation', cache=True)
def elevation(buildings, parcels):
    return misc.reindex(parcels.elevation, buildings.parcel_id).fillna(parcels.elevation.median())
    
@sim.column('buildings', 'volume_two_way', cache=True)
def volume_two_way(buildings, parcels):
    df = misc.reindex(parcels.volume_two_way, buildings.parcel_id)
    return df.fillna(0)
    
@sim.column('buildings', 'volume_two_way_nofwy', cache=True)
def volume_two_way_nofwy(buildings, parcels):
    df = misc.reindex(parcels.volume_two_way_nofwy, buildings.parcel_id)
    return df.fillna(0)

@sim.column('buildings', 'max_far', cache=True)
def max_far(buildings, parcels):
    df = misc.reindex(parcels.max_far, buildings.parcel_id)
    return df.fillna(0)
    
@sim.column('buildings', 'residential_sqft', cache=True)
def residential_sqft(buildings):
    df = buildings.residential_units * buildings.sqft_per_unit
    return df
    
@sim.column('buildings', 'is_sf', cache=True)
def is_sf(buildings):
    return (buildings.building_type_id==1).astype('int32')
    
@sim.column('buildings', 'is_mf', cache=True)
def is_mf(buildings):
    return (buildings.building_type_id==2).astype('int32')
    
@sim.column('buildings', 'is_industrial', cache=True)
def is_industrial(buildings):
    return (buildings.building_type_id==3).astype('int32')

@sim.column('buildings', 'is_retail', cache=True)
def is_retail(buildings):
    return (buildings.building_type_id==4).astype('int32')
    
@sim.column('buildings', 'is_office', cache=True)
def is_office(buildings):
    return (buildings.building_type_id==5).astype('int32')
    
@sim.column('buildings', 'is_govt', cache=True)
def is_govt(buildings):
    return (buildings.building_type_id==6).astype('int32')

@sim.column('buildings', 'is_mixeduse', cache=True)
def is_mixeduse(buildings):
    return (buildings.building_type_id==7).astype('int32')
    
@sim.column('buildings', 'is_other', cache=True)
def is_other(buildings):
    return (buildings.building_type_id==8).astype('int32')

@sim.column('buildings', 'airport', cache=True)
def airport(buildings, parcels):
    df = misc.reindex(parcels.airport, buildings.parcel_id)
    return df.fillna(0)

@sim.column('buildings', 'airport_distance', cache=True)
def airport_distance(buildings, parcels):
    df = misc.reindex(parcels.airport_distance, buildings.parcel_id)
    return df.fillna(0)
    
@sim.column('buildings', 'fwy_exit_dist', cache=True)
def fwy_exit_dist(buildings, parcels):
    df = misc.reindex(parcels.fwy_exit_dist, buildings.parcel_id)
    return df.fillna(0)

@sim.column('buildings', 'fwy_exit_dist_tdm_output', cache=True)
def fwy_exit_dist_tdm_output(buildings, parcels):
    df = misc.reindex(parcels.fwy_exit_dist_tdm_output, buildings.parcel_id)
    return df.fillna(0)

@sim.column('buildings', 'raildepot_dist', cache=True)
def raildepot_dist(buildings, parcels):
    df = misc.reindex(parcels.raildepot_dist, buildings.parcel_id)
    return df.fillna(0)
    
@sim.column('buildings', 'university_dist', cache=True)
def university_dist(buildings, parcels):
    df = misc.reindex(parcels.university_dist, buildings.parcel_id)
    return df.fillna(0)
    
@sim.column('buildings', 'trail_dist', cache=True)
def trail_dist(buildings, parcels):
    df = misc.reindex(parcels.trail_dist, buildings.parcel_id)
    return df.fillna(0)

@sim.column('buildings', 'stream_dist', cache=True)
def stream_dist(buildings, parcels):
    df = misc.reindex(parcels.stream_dist, buildings.parcel_id)
    return df.fillna(0)
    
@sim.column('buildings', 'rail_stn_dist', cache=True)
def rail_stn_dist(buildings, parcels):
    df = misc.reindex(parcels.rail_stn_dist, buildings.parcel_id)
    return df.fillna(0)
    
@sim.column('buildings', 'bus_rte_dist', cache=True)
def bus_rte_dist(buildings, parcels):
    df = misc.reindex(parcels.bus_rte_dist, buildings.parcel_id)
    return df.fillna(0)

@sim.column('buildings', 'bus_stop_dist', cache=True)
def bus_stop_dist(buildings, parcels):
    df = misc.reindex(parcels.bus_stop_dist, buildings.parcel_id)
    return df.fillna(0)
    
@sim.column('buildings', 'residential_sales_price_sqft', cache=True)
def residential_sales_price_sqft(buildings, parcels):
    df = misc.reindex(parcels.residential_sales_price_sqft, buildings.parcel_id)
    return df.fillna(0)
    
@sim.column('buildings', 'total_job_spaces', cache=True)
def total_job_spaces(buildings, parcels):
    df = misc.reindex(parcels.total_job_spaces, buildings.parcel_id)
    return df.fillna(0)
    
@sim.column('buildings', 'rail_depot', cache=True)
def rail_depot(buildings, parcels):
    df = misc.reindex(parcels.rail_depot, buildings.parcel_id)
    return df.fillna(0)

@sim.column("buildings", "improvement_value", cache=True)
def improvement_value(buildings):
    return (buildings.res_price_per_sqft * buildings.building_sqft).fillna(0)

@sim.column("buildings", "unit_price_residential", cache=True)
def unit_price_residential(buildings):
    df = (buildings.improvement_value / buildings.residential_units)
    df = df.replace([np.inf, -np.inf], np.nan)
    return df.fillna(0)
    
@sim.column('buildings', 'distlrg_med_inc')
def distlrg_med_inc(buildings, distlrg):
    df = misc.reindex(distlrg.distlrg_median_income, buildings.distlrg_id)
    return df.fillna(0)

@sim.column('buildings', 'distmed_med_inc')
def distmed_med_inc(buildings, distmed):
    df = misc.reindex(distmed.distmed_median_income, buildings.distmed_id)
    return df.fillna(0)
    
@sim.column('buildings', 'distsml_med_inc')
def distsml_med_inc(buildings, distsml):
    df = misc.reindex(distsml.distsml_median_income, buildings.distsml_id)
    return df.fillna(0)

####################################
# BUILDINGS FOR ESTIMATION VARIABLES
####################################

@sim.column('buildings_for_estimation', 'node_id', cache=True)
def node_id(buildings_for_estimation, parcels):
    return misc.reindex(parcels.node_id, buildings_for_estimation.parcel_id)
    
@sim.column('buildings_for_estimation', 'zone_id', cache=True)
def zone_id(buildings_for_estimation, parcels):
    return misc.reindex(parcels.zone_id, buildings_for_estimation.parcel_id)
    
@sim.column('buildings_for_estimation', 'county_id', cache=True)
def zone_id(buildings_for_estimation, parcels):
    return misc.reindex(parcels.county_id, buildings_for_estimation.parcel_id)    
    
@sim.column('buildings_for_estimation', 'general_type', cache=True)
def general_type(buildings_for_estimation, building_type_map):
    return buildings_for_estimation.building_type_id.map(building_type_map)
    
@sim.column('buildings_for_estimation', 'sqft_per_unit', cache=True)
def unit_sqft(buildings_for_estimation):
    return buildings_for_estimation.building_sqft / buildings_for_estimation.residential_units.replace(0, 1)


@sim.column('buildings_for_estimation', 'lot_size_per_unit', cache=True)
def lot_size_per_unit(buildings_for_estimation, parcels):
    return misc.reindex(parcels.lot_size_per_unit, buildings_for_estimation.parcel_id)


@sim.column('buildings_for_estimation', 'sqft_per_job', cache=True)
def sqft_per_job(buildings_for_estimation, building_sqft_per_job):
    return buildings_for_estimation.building_type_id.fillna(-1).map(building_sqft_per_job)


@sim.column('buildings_for_estimation', 'job_spaces', cache=True)
def job_spaces(buildings_for_estimation):
    return (buildings_for_estimation.non_residential_sqft /
            buildings_for_estimation.sqft_per_job).fillna(0).astype('int')


@sim.column('buildings_for_estimation', 'vacant_residential_units')
def vacant_residential_units(buildings_for_estimation, households):
    return buildings_for_estimation.residential_units.sub(
        households.building_id.value_counts(), fill_value=0)


@sim.column('buildings_for_estimation', 'vacant_job_spaces')
def vacant_job_spaces(buildings_for_estimation, jobs):
    return buildings_for_estimation.job_spaces.sub(
        jobs.building_id.value_counts(), fill_value=0)
        
@sim.column("buildings_for_estimation", "land_value", cache=True)
def land_value(buildings_for_estimation, parcels):
    df = misc.reindex(parcels.land_value, buildings_for_estimation.parcel_id)
    return df.fillna(0)
    
@sim.column("buildings_for_estimation", "parcel_acres", cache=True)
def parcel_acres(buildings_for_estimation, parcels):
    df = misc.reindex(parcels.parcel_acres, buildings_for_estimation.parcel_id)
    return df.fillna(0)

@sim.column("buildings_for_estimation", "parcel_volume", cache=True)
def parcel_acres(buildings_for_estimation, parcels):
    df = misc.reindex(parcels.parcel_volume, buildings_for_estimation.parcel_id)
    return df.fillna(0)
    
@sim.column('buildings_for_estimation', 'elevation', cache=True)
def elevation(buildings_for_estimation, parcels):
    return misc.reindex(parcels.elevation, buildings_for_estimation.parcel_id)
    
@sim.column('buildings_for_estimation', 'volume_two_way', cache=True)
def volume_two_way(buildings_for_estimation, parcels):
    return misc.reindex(parcels.volume_two_way, buildings_for_estimation.parcel_id)
    
############################################
# GROUPED BUILDINGS FOR ESTIMATION VARIABLES
############################################

@sim.column('buildings_for_estimation_grouped', 'node_id', cache=True)
def node_id(buildings_for_estimation_grouped, parcels):
    return misc.reindex(parcels.node_id, buildings_for_estimation_grouped.parcel_id)
    
@sim.column('buildings_for_estimation_grouped', 'zone_id', cache=True)
def zone_id(buildings_for_estimation_grouped, parcels):
    return misc.reindex(parcels.zone_id, buildings_for_estimation_grouped.parcel_id)
    
@sim.column('buildings_for_estimation_grouped', 'county_id', cache=True)
def zone_id(buildings_for_estimation_grouped, parcels):
    return misc.reindex(parcels.county_id, buildings_for_estimation_grouped.parcel_id)  
    
@sim.column('buildings_for_estimation_grouped', 'distmed_id')
def distmed_id(buildings_for_estimation_grouped, parcels):
    df = misc.reindex(parcels.distmed_id, buildings_for_estimation_grouped.parcel_id)
    return df.fillna(0)
    
@sim.column('buildings_for_estimation_grouped', 'distsml_id')
def distsml_id(buildings_for_estimation_grouped, parcels):
    df = misc.reindex(parcels.distsml_id, buildings_for_estimation_grouped.parcel_id)
    return df.fillna(0)
    
@sim.column('buildings_for_estimation_grouped', 'distmed_1')
def distmed_1(buildings_for_estimation_grouped):
    return (buildings_for_estimation_grouped.distmed_id==1).astype('int32')

@sim.column('buildings_for_estimation_grouped', 'distmed_2')
def distmed_2(buildings_for_estimation_grouped):
    return (buildings_for_estimation_grouped.distmed_id==2).astype('int32')

@sim.column('buildings_for_estimation_grouped', 'distmed_3')
def distmed_3(buildings_for_estimation_grouped):
    return (buildings_for_estimation_grouped.distmed_id==3).astype('int32')

@sim.column('buildings_for_estimation_grouped', 'distmed_4')
def distmed_4(buildings_for_estimation_grouped):
    return (buildings_for_estimation_grouped.distmed_id==4).astype('int32')

@sim.column('buildings_for_estimation_grouped', 'distmed_5')
def distmed_5(buildings_for_estimation_grouped):
    return (buildings_for_estimation_grouped.distmed_id==5).astype('int32')

@sim.column('buildings_for_estimation_grouped', 'distmed_6')
def distmed_6(buildings_for_estimation_grouped):
    return (buildings_for_estimation_grouped.distmed_id==6).astype('int32')

@sim.column('buildings_for_estimation_grouped', 'distmed_7')
def distmed_7(buildings_for_estimation_grouped):
    return (buildings_for_estimation_grouped.distmed_id==7).astype('int32')

@sim.column('buildings_for_estimation_grouped', 'distmed_8')
def distmed_8(buildings_for_estimation_grouped):
    return (buildings_for_estimation_grouped.distmed_id==8).astype('int32')

@sim.column('buildings_for_estimation_grouped', 'distmed_9')
def distmed_9(buildings_for_estimation_grouped):
    return (buildings_for_estimation_grouped.distmed_id==9).astype('int32')
    
@sim.column('buildings_for_estimation_grouped', 'distmed_10')
def distmed_10(buildings_for_estimation_grouped):
    return (buildings_for_estimation_grouped.distmed_id==10).astype('int32')

@sim.column('buildings_for_estimation_grouped', 'distmed_11')
def distmed_11(buildings_for_estimation_grouped):
    return (buildings_for_estimation_grouped.distmed_id==11).astype('int32')

@sim.column('buildings_for_estimation_grouped', 'distmed_12')
def distmed_12(buildings_for_estimation_grouped):
    return (buildings_for_estimation_grouped.distmed_id==12).astype('int32')

@sim.column('buildings_for_estimation_grouped', 'distmed_13')
def distmed_13(buildings_for_estimation_grouped):
    return (buildings_for_estimation_grouped.distmed_id==13).astype('int32')

@sim.column('buildings_for_estimation_grouped', 'distmed_14')
def distmed_14(buildings_for_estimation_grouped):
    return (buildings_for_estimation_grouped.distmed_id==14).astype('int32')

@sim.column('buildings_for_estimation_grouped', 'distmed_15')
def distmed_15(buildings_for_estimation_grouped):
    return (buildings_for_estimation_grouped.distmed_id==15).astype('int32')

@sim.column('buildings_for_estimation_grouped', 'distmed_16')
def distmed_16(buildings_for_estimation_grouped):
    return (buildings_for_estimation_grouped.distmed_id==16).astype('int32')

@sim.column('buildings_for_estimation_grouped', 'distmed_17')
def distmed_17(buildings_for_estimation_grouped):
    return (buildings_for_estimation_grouped.distmed_id==17).astype('int32')

@sim.column('buildings_for_estimation_grouped', 'distmed_18')
def distmed_18(buildings_for_estimation_grouped):
    return (buildings_for_estimation_grouped.distmed_id==18).astype('int32')

@sim.column('buildings_for_estimation_grouped', 'distmed_19')
def distmed_19(buildings_for_estimation_grouped):
    return (buildings_for_estimation_grouped.distmed_id==19).astype('int32')
    
@sim.column('buildings_for_estimation_grouped', 'distmed_20')
def distmed_20(buildings_for_estimation_grouped):
    return (buildings_for_estimation_grouped.distmed_id==20).astype('int32')
    
@sim.column('buildings_for_estimation_grouped', 'distmed_21')
def distmed_21(buildings_for_estimation_grouped):
    return (buildings_for_estimation_grouped.distmed_id==21).astype('int32')

@sim.column('buildings_for_estimation_grouped', 'distmed_22')
def distmed_22(buildings_for_estimation_grouped):
    return (buildings_for_estimation_grouped.distmed_id==22).astype('int32')

@sim.column('buildings_for_estimation_grouped', 'distmed_23')
def distmed_23(buildings_for_estimation_grouped):
    return (buildings_for_estimation_grouped.distmed_id==23).astype('int32')

@sim.column('buildings_for_estimation_grouped', 'distmed_24')
def distmed_24(buildings_for_estimation_grouped):
    return (buildings_for_estimation_grouped.distmed_id==24).astype('int32')

@sim.column('buildings_for_estimation_grouped', 'distmed_25')
def distmed_25(buildings_for_estimation_grouped):
    return (buildings_for_estimation_grouped.distmed_id==25).astype('int32')

@sim.column('buildings_for_estimation_grouped', 'distmed_26')
def distmed_26(buildings_for_estimation_grouped):
    return (buildings_for_estimation_grouped.distmed_id==26).astype('int32')

@sim.column('buildings_for_estimation_grouped', 'distmed_27')
def distmed_27(buildings_for_estimation_grouped):
    return (buildings_for_estimation_grouped.distmed_id==27).astype('int32')

@sim.column('buildings_for_estimation_grouped', 'distmed_28')
def distmed_28(buildings_for_estimation_grouped):
    return (buildings_for_estimation_grouped.distmed_id==28).astype('int32')

@sim.column('buildings_for_estimation_grouped', 'distmed_29')
def distmed_29(buildings_for_estimation_grouped):
    return (buildings_for_estimation_grouped.distmed_id==29).astype('int32')
    
@sim.column('buildings_for_estimation_grouped', 'distmed_30')
def distmed_30(buildings_for_estimation_grouped):
    return (buildings_for_estimation_grouped.distmed_id==30).astype('int32')
    
@sim.column('buildings_for_estimation_grouped', 'distmed_31')
def distmed_31(buildings_for_estimation_grouped):
    return (buildings_for_estimation_grouped.distmed_id==31).astype('int32')

@sim.column('buildings_for_estimation_grouped', 'distmed_32')
def distmed_32(buildings_for_estimation_grouped):
    return (buildings_for_estimation_grouped.distmed_id==32).astype('int32')

@sim.column('buildings_for_estimation_grouped', 'distmed_33')
def distmed_33(buildings_for_estimation_grouped):
    return (buildings_for_estimation_grouped.distmed_id==33).astype('int32')

@sim.column('buildings_for_estimation_grouped', 'distmed_34')
def distmed_34(buildings_for_estimation_grouped):
    return (buildings_for_estimation_grouped.distmed_id==34).astype('int32')

@sim.column('buildings_for_estimation_grouped', 'distmed_35')
def distmed_35(buildings_for_estimation_grouped):
    return (buildings_for_estimation_grouped.distmed_id==35).astype('int32')

@sim.column('buildings_for_estimation_grouped', 'distmed_36')
def distmed_36(buildings_for_estimation_grouped):
    return (buildings_for_estimation_grouped.distmed_id==36).astype('int32')

@sim.column('buildings_for_estimation_grouped', 'distmed_37')
def distmed_37(buildings_for_estimation_grouped):
    return (buildings_for_estimation_grouped.distmed_id==37).astype('int32')

@sim.column('buildings_for_estimation_grouped', 'distmed_38')
def distmed_38(buildings_for_estimation_grouped):
    return (buildings_for_estimation_grouped.distmed_id==38).astype('int32')

@sim.column('buildings_for_estimation_grouped', 'distmed_39')
def distmed_39(buildings_for_estimation_grouped):
    return (buildings_for_estimation_grouped.distmed_id==39).astype('int32')
    
@sim.column('buildings_for_estimation_grouped', 'distmed_40')
def distmed_40(buildings_for_estimation_grouped):
    return (buildings_for_estimation_grouped.distmed_id==40).astype('int32')
    
@sim.column('buildings_for_estimation_grouped', 'distmed_41')
def distmed_41(buildings_for_estimation_grouped):
    return (buildings_for_estimation_grouped.distmed_id==41).astype('int32')  
    
@sim.column('buildings_for_estimation_grouped', 'general_type', cache=True)
def general_type(buildings_for_estimation_grouped, building_type_map):
    return buildings_for_estimation_grouped.building_type_id.map(building_type_map)
    
@sim.column('buildings_for_estimation_grouped', 'building_age', cache=True)
def building_age(buildings_for_estimation_grouped):
    return (buildings_for_estimation_grouped.trans_year - buildings_for_estimation_grouped.year_built).clip(lower=0)

@sim.column('buildings_for_estimation_grouped', 'lot_size_per_unit', cache=True)
def lot_size_per_unit(buildings_for_estimation_grouped, parcels):
    return misc.reindex(parcels.lot_size_per_unit, buildings_for_estimation_grouped.parcel_id)


@sim.column('buildings_for_estimation_grouped', 'sqft_per_job', cache=True)
def sqft_per_job(buildings_for_estimation_grouped, building_sqft_per_job):
    return buildings_for_estimation_grouped.building_type_id.fillna(-1).map(building_sqft_per_job)


@sim.column('buildings_for_estimation_grouped', 'job_spaces', cache=True)
def job_spaces(buildings_for_estimation_grouped):
    return (buildings_for_estimation_grouped.non_residential_sqft /
            buildings_for_estimation_grouped.sqft_per_job).fillna(0).astype('int')

@sim.column("buildings_for_estimation_grouped", "real_far", cache=True)
def real_far(buildings_for_estimation_grouped):
    df = buildings_for_estimation_grouped.building_sqft.fillna(0)/(buildings_for_estimation_grouped.parcel_acres.fillna(0.001)*43560)
    df = df.replace(np.inf, 0)
    df = df.clip(upper=50)
    return df.fillna(0)

@sim.column('buildings_for_estimation_grouped', 'vacant_residential_units')
def vacant_residential_units(buildings_for_estimation_grouped, households):
    return buildings_for_estimation_grouped.residential_units.sub(
        households.building_id.value_counts(), fill_value=0)

@sim.column('buildings_for_estimation_grouped', 'vacant_job_spaces')
def vacant_job_spaces(buildings_for_estimation_grouped, jobs):
    return buildings_for_estimation_grouped.job_spaces.sub(
        jobs.building_id.value_counts(), fill_value=0)
        
@sim.column("buildings_for_estimation_grouped", "land_value", cache=True)
def land_value(buildings_for_estimation_grouped, parcels):
    df = misc.reindex(parcels.land_value, buildings_for_estimation_grouped.parcel_id)
    return df.fillna(0)
    
@sim.column("buildings_for_estimation_grouped", "parcel_acres", cache=True)
def parcel_acres(buildings_for_estimation_grouped, parcels):
    df = misc.reindex(parcels.parcel_acres, buildings_for_estimation_grouped.parcel_id)
    return df.fillna(0)

@sim.column("buildings_for_estimation_grouped", "parcel_volume", cache=True)
def parcel_volume(buildings_for_estimation_grouped, parcels):
    df = misc.reindex(parcels.parcel_volume, buildings_for_estimation_grouped.parcel_id)
    return df.fillna(0)
    
@sim.column('buildings_for_estimation_grouped', 'elevation', cache=True)
def elevation(buildings_for_estimation_grouped, parcels):
    return misc.reindex(parcels.elevation, buildings_for_estimation_grouped.parcel_id)
    
@sim.column('buildings_for_estimation_grouped', 'volume_two_way', cache=True)
def volume_two_way(buildings_for_estimation_grouped, parcels):
    return misc.reindex(parcels.volume_two_way, buildings_for_estimation_grouped.parcel_id)
    
@sim.column('buildings_for_estimation_grouped', 'fwy_exit_dist', cache=True)
def fwy_exit_dist(buildings_for_estimation_grouped, parcels):
    return misc.reindex(parcels.fwy_exit_dist, buildings_for_estimation_grouped.parcel_id)

@sim.column('buildings_for_estimation_grouped', 'fwy_exit_dist_tdm_output', cache=True)
def fwy_exit_dist_tdm_output(buildings_for_estimation_grouped, parcels):
    return misc.reindex(parcels.fwy_exit_dist_tdm_output, buildings_for_estimation_grouped.parcel_id)
    
@sim.column('buildings_for_estimation_grouped', 'total_job_spaces', cache=True)
def total_job_spaces(buildings_for_estimation_grouped, parcels):
    df = misc.reindex(parcels.total_job_spaces, buildings_for_estimation_grouped.parcel_id)
    return df.fillna(0)

@sim.column('buildings_for_estimation_grouped', 'airport_distance', cache=True)
def airport_distance(buildings_for_estimation_grouped, parcels):
    return misc.reindex(parcels.airport_distance, buildings_for_estimation_grouped.parcel_id)

@sim.column('buildings_for_estimation_grouped', 'residential_sales_price_sqft', cache=True)
def residential_sales_price_sqft(buildings_for_estimation_grouped, parcels):
    df = misc.reindex(parcels.residential_sales_price_sqft, buildings_for_estimation_grouped.parcel_id)
    return df.fillna(0)

@sim.column('buildings_for_estimation_grouped', 'rail_depot', cache=True)
def residential_sales_price_sqft(buildings_for_estimation_grouped, parcels):
    df = misc.reindex(parcels.rail_depot, buildings_for_estimation_grouped.parcel_id)
    return df.fillna(0)
    
@sim.column('buildings_for_estimation_grouped', 'rail_stn_dist', cache=True)
def rail_stn_dist(buildings_for_estimation_grouped, parcels):
    df = misc.reindex(parcels.rail_stn_dist, buildings_for_estimation_grouped.parcel_id)
    return df.fillna(0)

################
# JOBS VARIABLES
################

@sim.column('jobs', 'parcel_id')
def parcel_id(jobs, buildings):
    df = misc.reindex(buildings.parcel_id, jobs.building_id)
    return df.fillna(-1).astype("int")
    
@sim.column('jobs', 'county_id')
def county_id(jobs, buildings):
    df = misc.reindex(buildings.county_id, jobs.building_id)
    return df.fillna(-1).astype("int")
    
@sim.column('jobs', 'distlrg_id')
def distlrg_id(jobs, buildings):
    df = misc.reindex(buildings.distlrg_id, jobs.building_id)
    return df.fillna(-1).astype("int")

@sim.column('jobs', 'distmed_id')
def distmed_id(jobs, buildings):
    df = misc.reindex(buildings.distmed_id, jobs.building_id)
    return df.fillna(-1).astype("int")
    
@sim.column('jobs', 'distsml_id')
def distsml_id(jobs, buildings):
    df = misc.reindex(buildings.distsml_id, jobs.building_id)
    return df.fillna(-1).astype("int")

@sim.column('jobs', 'zone_id')
def zone_id(jobs, buildings):
    df = misc.reindex(buildings.zone_id, jobs.building_id)
    return df.fillna(-1).astype("int")
    
@sim.column('jobs', 'building_type_id')
def building_type_id(jobs, buildings):
    df = misc.reindex(buildings.building_type_id, jobs.building_id)
    return df.fillna(-1).astype("int")

@sim.column('jobs', 'b_year_built')
def b_year_built(jobs, buildings):
    df = misc.reindex(buildings.year_built, jobs.building_id)
    return df.fillna(-1).astype("int")
    
################
#ZONES VARIABLES
################
@sim.column('zones', 'building_age_zn')
def building_age_zn(zones, buildings):
    return buildings.building_age.groupby(buildings.zone_id).mean().\
        reindex(zones.index).fillna(-99)

@sim.column("zones", "tdm_county_id")
def tdm_county_id(parcels, zones):
    eq = pd.read_csv("./data/TAZCTYEQ.csv", index_col="Z")
    return eq.COUNTY.reindex(zones.index)
#    return parcels.county_id.groupby(parcels.zone_id).min().\
#        reindex(zones.index).fillna(0)
        
#@sim.column("zones", "res_median_price")
#def res_median_price(buildings, zones, settings):
#    rpm = settings['res_price_multiplier']
#    pr = buildings.res_price_per_sqft[(buildings.building_type_id.\
#        isin([1,2]))&(buildings.res_price_per_sqft>0)].groupby(buildings.zone_id).median().reindex(zones.index).fillna(0)
#    #pr = pr * p.distlrg_res_shift
#    return pr
    
@sim.column("zones", "nonres_median_price")
def nonres_median_price(buildings, zones, settings):
    nrpm = settings['nonres_price_multiplier']
    pr = buildings.unit_price_non_residential[(buildings.building_type_id.\
        isin([3,4,5]))&(buildings.unit_price_non_residential>0)].groupby(buildings.zone_id).median().reindex(zones.index).fillna(0)
    #pr = pr * p.distlrg_nonres_shift
    return pr

@sim.column("zones", "population")
def population(households, zones):
    return households.persons.groupby(households.zone_id).sum().\
        reindex(zones.index).fillna(0)
        
@sim.column("zones", "total_households")
def total_households(households, zones):
    return households.persons.groupby(households.zone_id).count().\
        reindex(zones.index).fillna(0)
'''    
@sim.column('zones', 'hh_target_2015')
def hh_target_2015(zones):
    hh = pd.read_csv("./data/hh_target_2015.csv", index_col="zone_id")
    hhtarget = hh.hh_target_2015.reindex(zones.index).fillna(0)
    return hhtarget

@sim.column('zones', 'hh_choice_control')
def hh_choice_control(zones):
    diff = zones.hh_target_2015 - zones.total_households
    return diff
'''
@sim.column("zones", "total_jobs")
def total_jobs(jobs, zones):
    return jobs.sector_id.groupby(jobs.zone_id).count().\
        reindex(zones.index).fillna(0)

@sim.column("zones", "jobs_1")
def jobs_1(jobs, zones):
    df = jobs.to_frame(['zone_id','sector_id'])
    df = df.query('sector_id==1')
    return df.groupby('zone_id').sector_id.count().\
        reindex(zones.index).fillna(0)
        
@sim.column("zones", "jobs_2")
def jobs_2(jobs, zones):
    df = jobs.to_frame(['zone_id','sector_id'])
    df = df.query('sector_id==2')
    return df.groupby('zone_id').sector_id.count().\
        reindex(zones.index).fillna(0)

@sim.column("zones", "jobs_3")
def jobs_3(jobs, zones):
    df = jobs.to_frame(['zone_id','sector_id'])
    df = df.query('sector_id==3')
    return df.groupby('zone_id').sector_id.count().\
        reindex(zones.index).fillna(0)

@sim.column("zones", "jobs_4")
def jobs_4(jobs, zones):
    df = jobs.to_frame(['zone_id','sector_id'])
    df = df.query('sector_id==4')
    return df.groupby('zone_id').sector_id.count().\
        reindex(zones.index).fillna(0)

@sim.column("zones", "jobs_5")
def jobs_5(jobs, zones):
    df = jobs.to_frame(['zone_id','sector_id'])
    df = df.query('sector_id==5')
    return df.groupby('zone_id').sector_id.count().\
        reindex(zones.index).fillna(0)

@sim.column("zones", "jobs_6")
def jobs_6(jobs, zones):
    df = jobs.to_frame(['zone_id','sector_id'])
    df = df.query('sector_id==6')
    return df.groupby('zone_id').sector_id.count().\
        reindex(zones.index).fillna(0)

@sim.column("zones", "jobs_7")
def jobs_7(jobs, zones):
    df = jobs.to_frame(['zone_id','sector_id'])
    df = df.query('sector_id==7')
    return df.groupby('zone_id').sector_id.count().\
        reindex(zones.index).fillna(0)

@sim.column("zones", "jobs_8")
def jobs_8(jobs, zones):
    df = jobs.to_frame(['zone_id','sector_id'])
    df = df.query('sector_id==8')
    return df.groupby('zone_id').sector_id.count().\
        reindex(zones.index).fillna(0)

@sim.column("zones", "jobs_9")
def jobs_9(jobs, zones):
    df = jobs.to_frame(['zone_id','sector_id'])
    df = df.query('sector_id==9')
    return df.groupby('zone_id').sector_id.count().\
        reindex(zones.index).fillna(0)

@sim.column("zones", "jobs_10")
def jobs_10(jobs, zones):
    df = jobs.to_frame(['zone_id','sector_id'])
    df = df.query('sector_id==10')
    return df.groupby('zone_id').sector_id.count().\
        reindex(zones.index).fillna(0)

@sim.column("zones", "sum_land_value")
def sum_land_value(parcels, zones):
    return parcels.land_value.groupby(parcels.zone_id).sum().\
        reindex(zones.index).fillna(0)
        
@sim.column("zones", "land_value_per_acre")
def land_value_per_acre(parcels, zones):
    return parcels.land_value.groupby(parcels.zone_id).sum().\
        reindex(zones.index).fillna(0)/parcels.parcel_acres.groupby(parcels.zone_id).sum().\
        reindex(zones.index).fillna(.001)
        
@sim.column("zones", "median_income")
def median_income(households, zones):
    return households.income.groupby(households.zone_id).median().\
        reindex(zones.index).fillna(0)
        
@sim.column("zones", "ave_hhsize_zn")
def ave_hhsize_zn(households, zones):
    return households.persons.groupby(households.zone_id).mean().\
        reindex(zones.index).fillna(0)

@sim.column("zones", "ave_age_of_head")
def ave_age_of_head(households, zones):
    return households.age_of_head.groupby(households.zone_id).mean().\
        reindex(zones.index).fillna(0)
        
@sim.column("zones", "ave_hhincome_zn")
def ave_hhincome_zn(households, zones):
    return households.income.groupby(households.zone_id).mean().\
        reindex(zones.index).fillna(0)
        
@sim.column("zones", "ave_hhpropwkrs_zn")
def ave_hhpropwkrs_zn(households, zones):
    return households.proportion_workers.groupby(households.zone_id).mean().\
        reindex(zones.index).fillna(0)
        
@sim.column("zones", "ave_hhcars_zn")
def ave_hhcars_zone(households, zones):
    return households.cars.groupby(households.zone_id).mean().\
        reindex(zones.index).fillna(0)
        
@sim.column("zones", "ave_hhchildren_zn")
def ave_hhchildren_zone(households, zones):
    return households.children.groupby(households.zone_id).mean().\
        reindex(zones.index).fillna(0)
        
@sim.column("zones", "pop_density")
def pop_density(parcels, households, zones):
    return (households.persons.groupby(households.zone_id).sum().\
        reindex(zones.index).fillna(0))/(parcels.parcel_acres.groupby(parcels.zone_id).sum().\
        reindex(zones.index).fillna(.001))
        
@sim.column("zones", "ave_nonres_price_zn")
def ave_nonres_price_zone(buildings, zones):
    return buildings.unit_price_non_residential[buildings.building_type_id.isin([3,4,5])].groupby(buildings.zone_id).mean().\
        reindex(zones.index).fillna(0)

@sim.column("zones", "ave_res_price_zn")
def ave_res_price_zone(buildings, zones):
    return buildings.res_price_per_sqft[buildings.building_type_id.isin([1,2])].groupby(buildings.zone_id).mean().\
        reindex(zones.index).fillna(0)


@sim.column('zones', 'population_within_30_min', cache=True)
def population_within_30_min(households, travel_data):
    p = households.to_frame()
    p = p[['zone_id','persons']].groupby('zone_id').sum()
    p = p[p.index>0]
    p = pd.Series(data=p.persons, index=p.index)
    td = travel_data.to_frame()
    td2 = td.reset_index(level=0)
    maxtaz = td2.index.max()+1  
    p30 = misc.compute_range(td, p, "travel_time", 30, agg=np.sum)
    p30 = p30.reindex(index=range(1,maxtaz),fill_value=0)
    return p30.fillna(0).astype('int')
    
@sim.column('zones', 'population_within_20_min', cache=True)
def population_within_20_min(households, travel_data):
    p = households.to_frame()
    p = p[['zone_id','persons']].groupby('zone_id').sum()
    p = p[p.index>0]
    p = pd.Series(data=p.persons, index=p.index)
    td = travel_data.to_frame()
    td2 = td.reset_index(level=0)
    maxtaz = td2.index.max()+1    
    p20 = misc.compute_range(td, p, "travel_time", 20, agg=np.sum)
    p20 = p20.reindex(index=range(1,maxtaz),fill_value=0)
    return p20.fillna(0).astype('int')

@sim.column('zones', 'population_within_15_min', cache=True)
def population_within_15_min(households, travel_data):
    p = households.to_frame()
    p = p[['zone_id','persons']].groupby('zone_id').sum()
    p = p[p.index>0]
    p = pd.Series(data=p.persons, index=p.index)
    td = travel_data.to_frame()
    td2 = td.reset_index(level=0)
    maxtaz = td2.index.max()+1
    p15 = misc.compute_range(td, p, "travel_time", 15, agg=np.sum)
    p15 = p15.reindex(index=range(1,maxtaz),fill_value=0)
    return p15.fillna(0).astype('int')

@sim.column('zones', 'population_within_10_min', cache=True)
def population_within_10_min(households, travel_data):
    p = households.to_frame()
    p = p[['zone_id','persons']].groupby('zone_id').sum()
    p = p[p.index>0]
    p = pd.Series(data=p.persons, index=p.index)
    td = travel_data.to_frame()
    td2 = td.reset_index(level=0)
    maxtaz = td2.index.max()+1
    p10 = misc.compute_range(td, p, "travel_time", 10, agg=np.sum)
    p10 = p10.reindex(index=range(1,maxtaz),fill_value=0)
    return p10.fillna(0).astype('int')
    
@sim.column('zones', 'population_within_40_min', cache=True)
def population_within_40_min(households, travel_data):
    p = households.to_frame()
    p = p[['zone_id','persons']].groupby('zone_id').sum()
    p = p[p.index>0]
    p = pd.Series(data=p.persons, index=p.index)
    td = travel_data.to_frame()
    td2 = td.reset_index(level=0)
    maxtaz = td2.index.max()+1   
    p40 = misc.compute_range(td, p, "travel_time", 40, agg=np.sum)
    p40 = p40.reindex(index=range(1,maxtaz),fill_value=0)
    return p40.fillna(0).astype('int')

@sim.column('zones', 'population_within_40_min_transit', cache=True)
def population_within_40_min_transit(households, travel_data):
    p = households.to_frame()
    p = p[['zone_id','persons']].groupby('zone_id').sum()
    p = p[p.index>0]
    p = pd.Series(data=p.persons, index=p.index)
    td = travel_data.to_frame()
    td2 = td.reset_index(level=0)
    maxtaz = td2.index.max()+1
    p40 = misc.compute_range(td, p, "travel_time_transit", 40, agg=np.sum)
    p40 = p40.reindex(index=range(1,maxtaz),fill_value=0)
    return p40.fillna(0).astype('int')

@sim.column('zones', 'population_within_30_min_transit', cache=True)
def population_within_30_min_transit(households, travel_data):
    p = households.to_frame()
    p = p[['zone_id','persons']].groupby('zone_id').sum()
    p = p[p.index>0]
    p = pd.Series(data=p.persons, index=p.index)
    td = travel_data.to_frame()
    td2 = td.reset_index(level=0)
    maxtaz = td2.index.max()+1
    p30 = misc.compute_range(td, p, "travel_time_transit", 30, agg=np.sum)
    p30 = p30.reindex(index=range(1,maxtaz),fill_value=0)
    return p30.fillna(0).astype('int')

@sim.column('zones', 'population_within_20_min_transit', cache=True)
def population_within_20_min_transit(households, travel_data):
    p = households.to_frame()
    p = p[['zone_id','persons']].groupby('zone_id').sum()
    p = p[p.index>0]
    p = pd.Series(data=p.persons, index=p.index)
    td = travel_data.to_frame()
    td2 = td.reset_index(level=0)
    maxtaz = td2.index.max()+1
    p20 = misc.compute_range(td, p, "travel_time_transit", 20, agg=np.sum)
    p20 = p20.reindex(index=range(1,maxtaz),fill_value=0)
    return p20.fillna(0).astype('int')

@sim.column('zones', 'jobs_within_30_min', cache=True)
def jobs_within_30_min(jobs, travel_data):
    j = pd.DataFrame({'zone_id':jobs.zone_id})
    j = j[j.zone_id>0]
    td = travel_data.to_frame()
    td2 = td.reset_index(level=0)
    maxtaz = td2.index.max()+1    
    j30 = misc.compute_range(td, j.groupby('zone_id').size(), "travel_time", 30, agg=np.sum)
    j30 = j30.reindex(index=range(1,maxtaz),fill_value=0)
    return j30.fillna(0).astype('int')

@sim.column('zones', 'jobs_within_40_min_transit', cache=True)
def jobs_within_40_min_transit(jobs, travel_data):
    j = pd.DataFrame({'zone_id':jobs.zone_id})
    j = j[j.zone_id>0]
    td = travel_data.to_frame()
    td2 = td.reset_index(level=0)
    maxtaz = td2.index.max()+1
    j40 = misc.compute_range(td, j.groupby('zone_id').size(), "travel_time_transit", 40, agg=np.sum)
    j40 = j40.reindex(index=range(1,maxtaz),fill_value=0)
    return j40.fillna(0).astype('int')

@sim.column('zones', 'jobs_within_30_min_transit', cache=True)
def jobs_within_30_min_transit(jobs, travel_data):
    j = pd.DataFrame({'zone_id':jobs.zone_id})
    j = j[j.zone_id>0]
    td = travel_data.to_frame()
    td2 = td.reset_index(level=0)
    maxtaz = td2.index.max()+1
    j30 = misc.compute_range(td, j.groupby('zone_id').size(), "travel_time_transit", 30, agg=np.sum)
    j30 = j30.reindex(index=range(1,maxtaz),fill_value=0)
    return j30.fillna(0).astype('int')

@sim.column('zones', 'jobs_within_20_min_transit', cache=True)
def jobs_within_20_min_transit(jobs, travel_data):
    j = pd.DataFrame({'zone_id':jobs.zone_id})
    j = j[j.zone_id>0]
    td = travel_data.to_frame()
    td2 = td.reset_index(level=0)
    maxtaz = td2.index.max()+1
    j20 = misc.compute_range(td, j.groupby('zone_id').size(), "travel_time_transit", 20, agg=np.sum)
    j20 = j20.reindex(index=range(1,maxtaz),fill_value=0)
    return j20.fillna(0).astype('int')
    
@sim.column('zones', 'jobs_within_20_min', cache=True)
def jobs_within_20_min(jobs, travel_data):
    j = pd.DataFrame({'zone_id':jobs.zone_id})
    j = j[j.zone_id>0]
    td = travel_data.to_frame()
    td2 = td.reset_index(level=0)
    maxtaz = td2.index.max()+1
    j20 = misc.compute_range(td, j.groupby('zone_id').size(), "travel_time", 20, agg=np.sum)
    j20 = j20.reindex(index=range(1,maxtaz),fill_value=0)
    return j20.fillna(0).astype('int')

@sim.column('zones', 'jobs_within_15_min', cache=True)
def jobs_within_15_min(jobs, travel_data):
    j = pd.DataFrame({'zone_id':jobs.zone_id})
    j = j[j.zone_id>0]
    td = travel_data.to_frame()
    td2 = td.reset_index(level=0)
    maxtaz = td2.index.max()+1
    j15 = misc.compute_range(td, j.groupby('zone_id').size(), "travel_time", 15, agg=np.sum)
    j15 = j15.reindex(index=range(1,maxtaz),fill_value=0)
    return j15.fillna(0).astype('int')

@sim.column('zones', 'jobs_within_10_min', cache=True)
def jobs_within_10_min(jobs, travel_data):
    j = pd.DataFrame({'zone_id':jobs.zone_id})
    j = j[j.zone_id>0]
    td = travel_data.to_frame()
    td2 = td.reset_index(level=0)
    maxtaz = td2.index.max()+1
    j10 = misc.compute_range(td, j.groupby('zone_id').size(), "travel_time", 10, agg=np.sum)
    j10 = j10.reindex(index=range(1,maxtaz),fill_value=0)
    return j10.fillna(0).astype('int')

@sim.column('zones', 'logsumjobs', cache=True)
def logsumjobs(jobs, travel_data):
    j = pd.DataFrame({'zone_id':jobs.zone_id})
    j = j[j.zone_id>0]
    td = travel_data.to_frame()
    jzone = j.groupby('zone_id').size()
    tdprocess = td.reset_index(level = 1)
    tdprocess["attr"] = jzone[tdprocess.to_zone_id].values
    tdprocess["attr1"] = tdprocess["attr"]*np.exp(tdprocess["log2"])
    logsum = tdprocess.groupby(level = 0).attr1.apply(np.sum)
    return logsum.fillna(0).astype('int')

@sim.column('zones', 'logsumpopulation', cache=True)
def logsumpopulation(households, travel_data):
    p = households.to_frame()
    p = p[['zone_id','persons']].groupby('zone_id').sum()
    p = p[p.index>0]
    p = pd.Series(data=p.persons, index=p.index)
    td = travel_data.to_frame()
    tdprocess = td.reset_index(level = 1)
    tdprocess["attr"] = p[tdprocess.to_zone_id].values
    tdprocess["attr1"] = tdprocess["attr"]*np.exp(tdprocess["log2"])
    logsum = tdprocess.groupby(level = 0).attr1.apply(np.sum)
    return logsum.fillna(0).astype('int')

@sim.column('zones', 'logsum_jobs1', cache=True)
def logsum_jobs1(jobs, travel_data):
    j = jobs.to_frame()
    j = j[(j.zone_id>0)&(j.sector_id==1)]
    td = travel_data.to_frame()
    jzone = j.groupby('zone_id').sector_id.count()
    tdprocess = td.reset_index(level = 1)
    tdprocess["attr"] = jzone[tdprocess.to_zone_id].values
    tdprocess["attr1"] = tdprocess["attr"]*np.exp(tdprocess["log2"])
    logsum = tdprocess.groupby(level = 0).attr1.apply(np.sum)
    return logsum.fillna(0).astype('int')
'''
@sim.column('zones', 'logsum_jobs2', cache=True)
def logsum_jobs2(jobs, travel_data):
    j = jobs.to_frame()

    j = j[(j.zone_id>0)&(j.sector_id==2)]

    td = travel_data.to_frame()

    jzone = j.groupby('zone_id').sector_id.count()
    tdprocess = td.reset_index(level = 1)

    tdprocess["attr"] = jzone[tdprocess.to_zone_id].values
    tdprocess["attr1"] = tdprocess["attr"]*np.exp(tdprocess["log2"])
    logsum = tdprocess.groupby(level = 0).attr1.apply(np.sum)
    return logsum.fillna(0).astype('int')
'''

@sim.column('zones', 'logsum_jobs3', cache=True)
def logsum_jobs3(jobs, travel_data):
    j = jobs.to_frame()
    j = j[(j.zone_id>0)&(j.sector_id==3)]
    td = travel_data.to_frame()
    jzone = j.groupby('zone_id').sector_id.count()
    tdprocess = td.reset_index(level = 1)
    tdprocess["attr"] = jzone[tdprocess.to_zone_id].values
    tdprocess["attr1"] = tdprocess["attr"]*np.exp(tdprocess["log2"])
    logsum = tdprocess.groupby(level = 0).attr1.apply(np.sum)
    return logsum.fillna(0).astype('int')

@sim.column('zones', 'logsum_jobs4', cache=True)
def logsum_jobs4(jobs, travel_data):
    j = jobs.to_frame()
    j = j[(j.zone_id>0)&(j.sector_id==4)]
    td = travel_data.to_frame()
    jzone = j.groupby('zone_id').sector_id.count()
    tdprocess = td.reset_index(level = 1)
    tdprocess["attr"] = jzone[tdprocess.to_zone_id].values
    tdprocess["attr1"] = tdprocess["attr"]*np.exp(tdprocess["log2"])
    logsum = tdprocess.groupby(level = 0).attr1.apply(np.sum)
    return logsum.fillna(0).astype('int')

@sim.column('zones', 'logsum_jobs5', cache=True)
def logsum_jobs5(jobs, travel_data):
    j = jobs.to_frame()
    j = j[(j.zone_id>0)&(j.sector_id==5)]
    td = travel_data.to_frame()
    jzone = j.groupby('zone_id').sector_id.count()
    tdprocess = td.reset_index(level = 1)
    tdprocess["attr"] = jzone[tdprocess.to_zone_id].values
    tdprocess["attr1"] = tdprocess["attr"]*np.exp(tdprocess["log2"])
    logsum = tdprocess.groupby(level = 0).attr1.apply(np.sum)
    return logsum.fillna(0).astype('int')

@sim.column('zones', 'logsum_jobs6', cache=True)
def logsum_jobs6(jobs, travel_data):
    j = jobs.to_frame()
    j = j[(j.zone_id>0)&(j.sector_id==6)]
    td = travel_data.to_frame()
    jzone = j.groupby('zone_id').sector_id.count()
    tdprocess = td.reset_index(level = 1)
    tdprocess["attr"] = jzone[tdprocess.to_zone_id].values
    tdprocess["attr1"] = tdprocess["attr"]*np.exp(tdprocess["log2"])
    logsum = tdprocess.groupby(level = 0).attr1.apply(np.sum)
    return logsum.fillna(0).astype('int')

@sim.column('zones', 'logsum_jobs7', cache=True)
def logsum_jobs7(jobs, travel_data):
    j = jobs.to_frame()
    j = j[(j.zone_id>0)&(j.sector_id==7)]
    td = travel_data.to_frame()
    jzone = j.groupby('zone_id').sector_id.count()
    tdprocess = td.reset_index(level = 1)
    tdprocess["attr"] = jzone[tdprocess.to_zone_id].values
    tdprocess["attr1"] = tdprocess["attr"]*np.exp(tdprocess["log2"])
    logsum = tdprocess.groupby(level = 0).attr1.apply(np.sum)
    return logsum.fillna(0).astype('int')
'''
@sim.column('zones', 'logsum_jobs8', cache=True)
def logsum_jobs8(jobs, travel_data):
    j = jobs.to_frame()
    j = j[(j.zone_id>0)&(j.sector_id==8)]
    td = travel_data.to_frame()
    jzone = j.groupby('zone_id').sector_id.count()
    tdprocess = td.reset_index(level = 1)
    tdprocess["attr"] = jzone[tdprocess.to_zone_id].values
    tdprocess["attr1"] = tdprocess["attr"]*np.exp(tdprocess["log2"])
    logsum = tdprocess.groupby(level = 0).attr1.apply(np.sum)
    return logsum.fillna(0).astype('int')
'''
@sim.column('zones', 'logsum_jobs9', cache=True)
def logsum_jobs9(jobs, travel_data):
    j = jobs.to_frame()
    j = j[(j.zone_id>0)&(j.sector_id==9)]
    td = travel_data.to_frame()
    jzone = j.groupby('zone_id').sector_id.count()
    tdprocess = td.reset_index(level = 1)
    tdprocess["attr"] = jzone[tdprocess.to_zone_id].values
    tdprocess["attr1"] = tdprocess["attr"]*np.exp(tdprocess["log2"])
    logsum = tdprocess.groupby(level = 0).attr1.apply(np.sum)
    return logsum.fillna(0).astype('int')

@sim.column('zones', 'logsum_jobs10', cache=True)
def logsum_jobs10(jobs, travel_data):
    j = jobs.to_frame()
    j = j[(j.zone_id>0)&(j.sector_id==10)]
    td = travel_data.to_frame()
    jzone = j.groupby('zone_id').sector_id.count()
    tdprocess = td.reset_index(level = 1)
    tdprocess["attr"] = jzone[tdprocess.to_zone_id].values
    tdprocess["attr1"] = tdprocess["attr"]*np.exp(tdprocess["log2"])
    logsum = tdprocess.groupby(level = 0).attr1.apply(np.sum)
    return logsum.fillna(0).astype('int')

@sim.column('zones', 'logsum_hhinc1', cache=True)
def logsum_hhinc1(households, travel_data):
    hh = households.to_frame()
    hh = hh[hh.income_quartile==1]
    hh = hh[['zone_id','persons']].groupby('zone_id').count()
    hh = hh[hh.index>0]
    hh = pd.Series(data=hh.persons, index=hh.index)
    td = travel_data.to_frame()
    tdprocess = td.reset_index(level = 1)
    tdprocess["attr"] = hh[tdprocess.to_zone_id].values
    tdprocess["attr1"] = tdprocess["attr"]*np.exp(tdprocess["log2"])
    logsum = tdprocess.groupby(level = 0).attr1.apply(np.sum)
    return logsum.fillna(0).astype('int')

@sim.column('zones', 'logsum_hhinc2', cache=True)
def logsum_hhinc2(households, travel_data):
    hh = households.to_frame()
    hh = hh[hh.income_quartile==2]
    hh = hh[['zone_id','persons']].groupby('zone_id').count()
    hh = hh[hh.index>0]
    hh = pd.Series(data=hh.persons, index=hh.index)
    td = travel_data.to_frame()
    tdprocess = td.reset_index(level = 1)
    tdprocess["attr"] = hh[tdprocess.to_zone_id].values
    tdprocess["attr1"] = tdprocess["attr"]*np.exp(tdprocess["log2"])
    logsum = tdprocess.groupby(level = 0).attr1.apply(np.sum)
    return logsum.fillna(0).astype('int')

@sim.column('zones', 'logsum_hhinc3', cache=True)
def logsum_hhinc3(households, travel_data):
    hh = households.to_frame()
    hh = hh[hh.income_quartile==3]
    hh = hh[['zone_id','persons']].groupby('zone_id').count()
    hh = hh[hh.index>0]
    hh = pd.Series(data=hh.persons, index=hh.index)
    td = travel_data.to_frame()
    tdprocess = td.reset_index(level = 1)
    tdprocess["attr"] = hh[tdprocess.to_zone_id].values
    tdprocess["attr1"] = tdprocess["attr"]*np.exp(tdprocess["log2"])
    logsum = tdprocess.groupby(level = 0).attr1.apply(np.sum)
    return logsum.fillna(0).astype('int')

@sim.column('zones', 'logsum_hhinc4', cache=True)
def logsum_hhinc4(households, travel_data):
    hh = households.to_frame()
    hh = hh[hh.income_quartile==4]
    hh = hh[['zone_id','persons']].groupby('zone_id').count()
    hh = hh[hh.index>0]
    hh = pd.Series(data=hh.persons, index=hh.index)
    td = travel_data.to_frame()
    tdprocess = td.reset_index(level = 1)
    tdprocess["attr"] = hh[tdprocess.to_zone_id].values
    tdprocess["attr1"] = tdprocess["attr"]*np.exp(tdprocess["log2"])
    logsum = tdprocess.groupby(level = 0).attr1.apply(np.sum)
    return logsum.fillna(0).astype('int')

@sim.column('zones', 'jobs_1_within_30_min', cache=True)
def jobs_1_within_30_min(jobs, travel_data):
    j = pd.DataFrame({'zone_id':jobs.zone_id,'sector_id':jobs.sector_id})
    j = j[(j.zone_id>0)&(j.sector_id==1)]
    td = travel_data.to_frame()
    td2 = td.reset_index(level=0)
    maxtaz = td2.index.max()+1    
    j30 = misc.compute_range(td, j.groupby('zone_id').size(), "travel_time", 30, agg=np.sum)
    j30 = j30.reindex(index=range(1,maxtaz),fill_value=0)
    return j30.fillna(0).astype('int')

@sim.column('zones', 'jobs_1_within_20_min', cache=True)
def jobs_1_within_20_min(jobs, travel_data):
    j = pd.DataFrame({'zone_id':jobs.zone_id,'sector_id':jobs.sector_id})
    j = j[(j.zone_id>0)&(j.sector_id==1)]
    td = travel_data.to_frame()
    td2 = td.reset_index(level=0)
    maxtaz = td2.index.max()+1
    jx = misc.compute_range(td, j.groupby('zone_id').size(), "travel_time", 20, agg=np.sum)
    jx = jx.reindex(index=range(1,maxtaz),fill_value=0)
    return jx.fillna(0).astype('int')

@sim.column('zones', 'jobs_1_within_10_min', cache=True)
def jobs_1_within_10_min(jobs, travel_data):
    j = pd.DataFrame({'zone_id':jobs.zone_id,'sector_id':jobs.sector_id})
    j = j[(j.zone_id>0)&(j.sector_id==1)]
    td = travel_data.to_frame()
    td2 = td.reset_index(level=0)
    maxtaz = td2.index.max()+1
    jx = misc.compute_range(td, j.groupby('zone_id').size(), "travel_time", 10, agg=np.sum)
    jx = jx.reindex(index=range(1,maxtaz),fill_value=0)
    return jx.fillna(0).astype('int')
'''
@sim.column('zones', 'jobs_2_within_30_min', cache=True)
def jobs_2_within_30_min(jobs, travel_data):
    j = pd.DataFrame({'zone_id':jobs.zone_id,'sector_id':jobs.sector_id})
    j = j[(j.zone_id>0)&(j.sector_id==2)]
    td = travel_data.to_frame()
    td2 = td.reset_index(level=0)
    maxtaz = td2.index.max()+1    
    j30 = misc.compute_range(td, j.groupby('zone_id').size(), "travel_time", 30, agg=np.sum)
    j30 = j30.reindex(index=range(1,maxtaz),fill_value=0)
    return j30.fillna(0).astype('int')

@sim.column('zones', 'jobs_2_within_20_min', cache=True)
def jobs_2_within_20_min(jobs, travel_data):
    j = pd.DataFrame({'zone_id':jobs.zone_id,'sector_id':jobs.sector_id})
    j = j[(j.zone_id>0)&(j.sector_id==2)]
    td = travel_data.to_frame()
    td2 = td.reset_index(level=0)
    maxtaz = td2.index.max()+1
    jx = misc.compute_range(td, j.groupby('zone_id').size(), "travel_time", 20, agg=np.sum)
    jx = jx.reindex(index=range(1,maxtaz),fill_value=0)
    return jx.fillna(0).astype('int')

@sim.column('zones', 'jobs_2_within_10_min', cache=True)
def jobs_2_within_10_min(jobs, travel_data):
    j = pd.DataFrame({'zone_id':jobs.zone_id,'sector_id':jobs.sector_id})
    j = j[(j.zone_id>0)&(j.sector_id==2)]
    td = travel_data.to_frame()
    td2 = td.reset_index(level=0)
    maxtaz = td2.index.max()+1
    jx = misc.compute_range(td, j.groupby('zone_id').size(), "travel_time", 10, agg=np.sum)
    jx = jx.reindex(index=range(1,maxtaz),fill_value=0)
    return jx.fillna(0).astype('int')
'''
@sim.column('zones', 'jobs_3_within_30_min', cache=True)
def jobs_3_within_30_min(jobs, travel_data):
    j = pd.DataFrame({'zone_id':jobs.zone_id,'sector_id':jobs.sector_id})
    j = j[(j.zone_id>0)&(j.sector_id==3)]
    td = travel_data.to_frame()
    td2 = td.reset_index(level=0)
    maxtaz = td2.index.max()+1    
    j30 = misc.compute_range(td, j.groupby('zone_id').size(), "travel_time", 30, agg=np.sum)
    j30 = j30.reindex(index=range(1,maxtaz),fill_value=0)
    return j30.fillna(0).astype('int')

@sim.column('zones', 'jobs_3_within_20_min', cache=True)
def jobs_3_within_20_min(jobs, travel_data):
    j = pd.DataFrame({'zone_id':jobs.zone_id,'sector_id':jobs.sector_id})
    j = j[(j.zone_id>0)&(j.sector_id==3)]
    td = travel_data.to_frame()
    td2 = td.reset_index(level=0)
    maxtaz = td2.index.max()+1
    jx = misc.compute_range(td, j.groupby('zone_id').size(), "travel_time", 20, agg=np.sum)
    jx = jx.reindex(index=range(1,maxtaz),fill_value=0)
    return jx.fillna(0).astype('int')

@sim.column('zones', 'jobs_3_within_10_min', cache=True)
def jobs_3_within_10_min(jobs, travel_data):
    j = pd.DataFrame({'zone_id':jobs.zone_id,'sector_id':jobs.sector_id})
    j = j[(j.zone_id>0)&(j.sector_id==3)]
    td = travel_data.to_frame()
    td2 = td.reset_index(level=0)
    maxtaz = td2.index.max()+1
    jx = misc.compute_range(td, j.groupby('zone_id').size(), "travel_time", 10, agg=np.sum)
    jx = jx.reindex(index=range(1,maxtaz),fill_value=0)
    return jx.fillna(0).astype('int')

@sim.column('zones', 'jobs_4_within_30_min', cache=True)
def jobs_4_within_30_min(jobs, travel_data):
    j = pd.DataFrame({'zone_id':jobs.zone_id,'sector_id':jobs.sector_id})
    j = j[(j.zone_id>0)&(j.sector_id==4)]
    td = travel_data.to_frame()
    td2 = td.reset_index(level=0)
    maxtaz = td2.index.max()+1    
    j30 = misc.compute_range(td, j.groupby('zone_id').size(), "travel_time", 30, agg=np.sum)
    j30 = j30.reindex(index=range(1,maxtaz),fill_value=0)
    return j30.fillna(0).astype('int')

@sim.column('zones', 'jobs_4_within_20_min', cache=True)
def jobs_4_within_20_min(jobs, travel_data):
    j = pd.DataFrame({'zone_id':jobs.zone_id,'sector_id':jobs.sector_id})
    j = j[(j.zone_id>0)&(j.sector_id==4)]
    td = travel_data.to_frame()
    td2 = td.reset_index(level=0)
    maxtaz = td2.index.max()+1
    jx = misc.compute_range(td, j.groupby('zone_id').size(), "travel_time", 20, agg=np.sum)
    jx = jx.reindex(index=range(1,maxtaz),fill_value=0)
    return jx.fillna(0).astype('int')

@sim.column('zones', 'jobs_4_within_10_min', cache=True)
def jobs_4_within_10_min(jobs, travel_data):
    j = pd.DataFrame({'zone_id':jobs.zone_id,'sector_id':jobs.sector_id})
    j = j[(j.zone_id>0)&(j.sector_id==4)]
    td = travel_data.to_frame()
    td2 = td.reset_index(level=0)
    maxtaz = td2.index.max()+1
    jx = misc.compute_range(td, j.groupby('zone_id').size(), "travel_time", 10, agg=np.sum)
    jx = jx.reindex(index=range(1,maxtaz),fill_value=0)
    return jx.fillna(0).astype('int')

@sim.column('zones', 'jobs_5_within_30_min', cache=True)
def jobs_5_within_30_min(jobs, travel_data):
    j = pd.DataFrame({'zone_id':jobs.zone_id,'sector_id':jobs.sector_id})
    j = j[(j.zone_id>0)&(j.sector_id==5)]
    td = travel_data.to_frame()
    td2 = td.reset_index(level=0)
    maxtaz = td2.index.max()+1    
    j30 = misc.compute_range(td, j.groupby('zone_id').size(), "travel_time", 30, agg=np.sum)
    j30 = j30.reindex(index=range(1,maxtaz),fill_value=0)
    return j30.fillna(0).astype('int')

@sim.column('zones', 'jobs_5_within_20_min', cache=True)
def jobs_5_within_20_min(jobs, travel_data):
    j = pd.DataFrame({'zone_id':jobs.zone_id,'sector_id':jobs.sector_id})
    j = j[(j.zone_id>0)&(j.sector_id==5)]
    td = travel_data.to_frame()
    td2 = td.reset_index(level=0)
    maxtaz = td2.index.max()+1
    jx = misc.compute_range(td, j.groupby('zone_id').size(), "travel_time", 20, agg=np.sum)
    jx = jx.reindex(index=range(1,maxtaz),fill_value=0)
    return jx.fillna(0).astype('int')

@sim.column('zones', 'jobs_5_within_10_min', cache=True)
def jobs_5_within_10_min(jobs, travel_data):
    j = pd.DataFrame({'zone_id':jobs.zone_id,'sector_id':jobs.sector_id})
    j = j[(j.zone_id>0)&(j.sector_id==5)]
    td = travel_data.to_frame()
    td2 = td.reset_index(level=0)
    maxtaz = td2.index.max()+1
    jx = misc.compute_range(td, j.groupby('zone_id').size(), "travel_time", 10, agg=np.sum)
    jx = jx.reindex(index=range(1,maxtaz),fill_value=0)
    return jx.fillna(0).astype('int')

@sim.column('zones', 'jobs_6_within_30_min', cache=True)
def jobs_6_within_30_min(jobs, travel_data):
    j = pd.DataFrame({'zone_id':jobs.zone_id,'sector_id':jobs.sector_id})
    j = j[(j.zone_id>0)&(j.sector_id==6)]
    td = travel_data.to_frame()
    td2 = td.reset_index(level=0)
    maxtaz = td2.index.max()+1
    j30 = misc.compute_range(td, j.groupby('zone_id').size(), "travel_time", 30, agg=np.sum)
    j30 = j30.reindex(index=range(1,maxtaz),fill_value=0)
    return j30.fillna(0).astype('int')

@sim.column('zones', 'jobs_6_within_20_min', cache=True)
def jobs_6_within_20_min(jobs, travel_data):
    j = pd.DataFrame({'zone_id':jobs.zone_id,'sector_id':jobs.sector_id})
    j = j[(j.zone_id>0)&(j.sector_id==6)]
    td = travel_data.to_frame()
    td2 = td.reset_index(level=0)
    maxtaz = td2.index.max()+1
    jx = misc.compute_range(td, j.groupby('zone_id').size(), "travel_time", 20, agg=np.sum)
    jx = jx.reindex(index=range(1,maxtaz),fill_value=0)
    return jx.fillna(0).astype('int')

@sim.column('zones', 'jobs_6_within_10_min', cache=True)
def jobs_6_within_10_min(jobs, travel_data):
    j = pd.DataFrame({'zone_id':jobs.zone_id,'sector_id':jobs.sector_id})
    j = j[(j.zone_id>0)&(j.sector_id==6)]
    td = travel_data.to_frame()
    td2 = td.reset_index(level=0)
    maxtaz = td2.index.max()+1
    jx = misc.compute_range(td, j.groupby('zone_id').size(), "travel_time", 10, agg=np.sum)
    jx = jx.reindex(index=range(1,maxtaz),fill_value=0)
    return jx.fillna(0).astype('int')

@sim.column('zones', 'jobs_7_within_30_min', cache=True)
def jobs_7_within_30_min(jobs, travel_data):
    j = pd.DataFrame({'zone_id':jobs.zone_id,'sector_id':jobs.sector_id})
    j = j[(j.zone_id>0)&(j.sector_id==7)]
    td = travel_data.to_frame()
    td2 = td.reset_index(level=0)
    maxtaz = td2.index.max()+1
    j30 = misc.compute_range(td, j.groupby('zone_id').size(), "travel_time", 30, agg=np.sum)
    j30 = j30.reindex(index=range(1,maxtaz),fill_value=0)
    return j30.fillna(0).astype('int')

@sim.column('zones', 'jobs_7_within_20_min', cache=True)
def jobs_7_within_20_min(jobs, travel_data):
    j = pd.DataFrame({'zone_id':jobs.zone_id,'sector_id':jobs.sector_id})
    j = j[(j.zone_id>0)&(j.sector_id==7)]
    td = travel_data.to_frame()
    td2 = td.reset_index(level=0)
    maxtaz = td2.index.max()+1
    jx = misc.compute_range(td, j.groupby('zone_id').size(), "travel_time", 20, agg=np.sum)
    jx = jx.reindex(index=range(1,maxtaz),fill_value=0)
    return jx.fillna(0).astype('int')

@sim.column('zones', 'jobs_7_within_10_min', cache=True)
def jobs_7_within_10_min(jobs, travel_data):
    j = pd.DataFrame({'zone_id':jobs.zone_id,'sector_id':jobs.sector_id})
    j = j[(j.zone_id>0)&(j.sector_id==7)]
    td = travel_data.to_frame()
    td2 = td.reset_index(level=0)
    maxtaz = td2.index.max()+1
    jx = misc.compute_range(td, j.groupby('zone_id').size(), "travel_time", 10, agg=np.sum)
    jx = jx.reindex(index=range(1,maxtaz),fill_value=0)
    return jx.fillna(0).astype('int')
'''    
@sim.column('zones', 'jobs_8_within_30_min', cache=True)
def jobs_8_within_30_min(jobs, travel_data):
    j = pd.DataFrame({'zone_id':jobs.zone_id,'sector_id':jobs.sector_id})
    j = j[(j.zone_id>0)&(j.sector_id==8)]
    td = travel_data.to_frame()
    td2 = td.reset_index(level=0)
    maxtaz = td2.index.max()+1
    j30 = misc.compute_range(td, j.groupby('zone_id').size(), "travel_time", 30, agg=np.sum)
    j30 = j30.reindex(index=range(1,maxtaz),fill_value=0)
    return j30.fillna(0).astype('int')

@sim.column('zones', 'jobs_8_within_20_min', cache=True)
def jobs_8_within_20_min(jobs, travel_data):
    j = pd.DataFrame({'zone_id':jobs.zone_id,'sector_id':jobs.sector_id})
    j = j[(j.zone_id>0)&(j.sector_id==8)]
    td = travel_data.to_frame()
    td2 = td.reset_index(level=0)
    maxtaz = td2.index.max()+1
    jx = misc.compute_range(td, j.groupby('zone_id').size(), "travel_time", 20, agg=np.sum)
    jx = jx.reindex(index=range(1,maxtaz),fill_value=0)
    return jx.fillna(0).astype('int')

@sim.column('zones', 'jobs_8_within_10_min', cache=True)
def jobs_8_within_10_min(jobs, travel_data):
    j = pd.DataFrame({'zone_id':jobs.zone_id,'sector_id':jobs.sector_id})
    j = j[(j.zone_id>0)&(j.sector_id==8)]
    td = travel_data.to_frame()
    td2 = td.reset_index(level=0)
    maxtaz = td2.index.max()+1
    jx = misc.compute_range(td, j.groupby('zone_id').size(), "travel_time", 10, agg=np.sum)
    jx = jx.reindex(index=range(1,maxtaz),fill_value=0)
    return jx.fillna(0).astype('int')
'''    
@sim.column('zones', 'jobs_9_within_30_min', cache=True)
def jobs_9_within_30_min(jobs, travel_data):
    j = pd.DataFrame({'zone_id':jobs.zone_id,'sector_id':jobs.sector_id})
    j = j[(j.zone_id>0)&(j.sector_id==9)]
    td = travel_data.to_frame()
    td2 = td.reset_index(level=0)
    maxtaz = td2.index.max()+1
    j30 = misc.compute_range(td, j.groupby('zone_id').size(), "travel_time", 30, agg=np.sum)
    j30 = j30.reindex(index=range(1,maxtaz),fill_value=0)
    return j30.fillna(0).astype('int')

@sim.column('zones', 'jobs_9_within_20_min', cache=True)
def jobs_9_within_20_min(jobs, travel_data):
    j = pd.DataFrame({'zone_id':jobs.zone_id,'sector_id':jobs.sector_id})
    j = j[(j.zone_id>0)&(j.sector_id==9)]
    td = travel_data.to_frame()
    td2 = td.reset_index(level=0)
    maxtaz = td2.index.max()+1
    jx = misc.compute_range(td, j.groupby('zone_id').size(), "travel_time", 20, agg=np.sum)
    jx = jx.reindex(index=range(1,maxtaz),fill_value=0)
    return jx.fillna(0).astype('int')

@sim.column('zones', 'jobs_9_within_10_min', cache=True)
def jobs_9_within_10_min(jobs, travel_data):
    j = pd.DataFrame({'zone_id':jobs.zone_id,'sector_id':jobs.sector_id})
    j = j[(j.zone_id>0)&(j.sector_id==9)]
    td = travel_data.to_frame()
    td2 = td.reset_index(level=0)
    maxtaz = td2.index.max()+1
    jx = misc.compute_range(td, j.groupby('zone_id').size(), "travel_time", 10, agg=np.sum)
    jx = jx.reindex(index=range(1,maxtaz),fill_value=0)
    return jx.fillna(0).astype('int')

@sim.column('zones', 'jobs_10_within_30_min', cache=True)
def jobs_10_within_30_min(jobs, travel_data):
    j = pd.DataFrame({'zone_id':jobs.zone_id,'sector_id':jobs.sector_id})
    j = j[(j.zone_id>0)&(j.sector_id==10)]
    td = travel_data.to_frame()
    td2 = td.reset_index(level=0)
    maxtaz = td2.index.max()+1
    j30 = misc.compute_range(td, j.groupby('zone_id').size(), "travel_time", 30, agg=np.sum)
    j30 = j30.reindex(index=range(1,maxtaz),fill_value=0)
    return j30.fillna(0).astype('int')

@sim.column('zones', 'jobs_10_within_20_min', cache=True)
def jobs_10_within_20_min(jobs, travel_data):
    j = pd.DataFrame({'zone_id':jobs.zone_id,'sector_id':jobs.sector_id})
    j = j[(j.zone_id>0)&(j.sector_id==10)]
    td = travel_data.to_frame()
    td2 = td.reset_index(level=0)
    maxtaz = td2.index.max()+1
    jx = misc.compute_range(td, j.groupby('zone_id').size(), "travel_time", 20, agg=np.sum)
    jx = jx.reindex(index=range(1,maxtaz),fill_value=0)
    return jx.fillna(0).astype('int')

@sim.column('zones', 'jobs_10_within_10_min', cache=True)
def jobs_10_within_10_min(jobs, travel_data):
    j = pd.DataFrame({'zone_id':jobs.zone_id,'sector_id':jobs.sector_id})
    j = j[(j.zone_id>0)&(j.sector_id==10)]
    td = travel_data.to_frame()
    td2 = td.reset_index(level=0)
    maxtaz = td2.index.max()+1
    jx = misc.compute_range(td, j.groupby('zone_id').size(), "travel_time", 10, agg=np.sum)
    jx = jx.reindex(index=range(1,maxtaz),fill_value=0)
    return jx.fillna(0).astype('int')
    
@sim.column('zones', 'res_units_within_30_min', cache=True)
def res_units_within_30_min(buildings, travel_data):
    b = buildings.to_frame()
    b = b[['zone_id','residential_units']].groupby('zone_id').sum()
    b = b[b.index>0]
    b = pd.Series(data=b.residential_units, index=b.index)
    td = travel_data.to_frame()
    td2 = td.reset_index(level=0)
    maxtaz = td2.index.max()+1
    ru30 = misc.compute_range(td, b, "travel_time", 30, agg=np.sum)
    ru30 = ru30.reindex(index=range(1,maxtaz),fill_value=0)
    return ru30.fillna(0).astype('int')
    
@sim.column('zones', 'res_units_within_20_min', cache=True)
def res_units_within_30_min(buildings, travel_data):
    b = buildings.to_frame()
    b = b[['zone_id','residential_units']].groupby('zone_id').sum()
    b = b[b.index>0]
    b = pd.Series(data=b.residential_units, index=b.index)
    td = travel_data.to_frame()
    td2 = td.reset_index(level=0)
    maxtaz = td2.index.max()+1
    ru20 = misc.compute_range(td, b, "travel_time", 20, agg=np.sum)
    ru20 = ru20.reindex(index=range(1,maxtaz),fill_value=0)    

    return ru20.fillna(0).astype('int')
'''
@sim.column('zones', 'volume_two_way_zn', cache=True)
def volume_two_way_zn(zones, year, settings):
    if year < 2020:
        file = settings['tdm']['main_dir'] + '2015/' + settings['tdm']['output_volume']
        z = wfrc_utils.dbf2df(file).set_index('TAZZ')
    elif year in range(2020,2028):
        file = settings['tdm']['main_dir'] + '2022/' + settings['tdm']['output_volume']
        z = wfrc_utils.dbf2df(file).set_index('TAZZ')
    elif year in range(2028,2036):
        file = settings['tdm']['main_dir'] + '2027/' + settings['tdm']['output_volume']
        z = wfrc_utils.dbf2df(file).set_index('TAZZ')
    #elif year in range(2031,2035):
    #    file = settings['tdm']['main_dir'] + '2030/' + settings['tdm']['output_volume']
    #    z = wfrc_utils.dbf2df(file).set_index('TAZZ')
    elif year in range(2036,2046):
        file = settings['tdm']['main_dir'] + '2035/' + settings['tdm']['output_volume']
        z = wfrc_utils.dbf2df(file).set_index('TAZZ')
    elif year in range(2046,2099):
        file = settings['tdm']['main_dir'] + '2045/' + settings['tdm']['output_volume']
        z = wfrc_utils.dbf2df(file).set_index('TAZZ')
    else:
        file = settings['tdm']['main_dir'] + '2011/' + settings['tdm']['output_volume']
        z = wfrc_utils.dbf2df(file).set_index('TAZZ')
    return z.VOL2WAY
'''
@sim.column('zones', 'commute_time_20', cache=True)
def commute_time_20(zones, year, settings):
    if year < 2020:
        file = settings['tdm']['main_dir'] + '2015/6_REMM/AvgTravelTime.csv'
        z = pd.read_csv(file).set_index('TAZ')
    elif year in range(2020,2028):
        file = settings['tdm']['main_dir'] + '2022/6_REMM/AvgTravelTime.csv'
        z = pd.read_csv(file).set_index('TAZ')
    elif year in range(2028,2036):
        file = settings['tdm']['main_dir'] + '2027/6_REMM/AvgTravelTime.csv'
        z = pd.read_csv(file).set_index('TAZ')
    #elif year in range(2031,2035):
    #    file = settings['tdm']['main_dir'] + '2030/' + settings['tdm']['output_volume']
    #    z = wfrc_utils.dbf2df(file).set_index('TAZZ')
    elif year in range(2036,2046):
        file = settings['tdm']['main_dir'] + '2035/6_REMM/AvgTravelTime.csv'
        z = pd.read_csv(file).set_index('TAZ')
    elif year in range(2046,2099):
        file = settings['tdm']['main_dir'] + '2045/6_REMM/AvgTravelTime.csv'
        z = pd.read_csv(file).set_index('TAZ')
    else:
        file = settings['tdm']['main_dir'] + '2011/6_REMM/AvgTravelTime.csv'
        z = pd.read_csv(file).set_index('TAZ')
    return np.absolute(z.TIMEAuto-20)

@sim.column('zones', 'commute_time', cache=True)
def commute_time(zones, year, settings):
    if year < 2020:
        file = settings['tdm']['main_dir'] + '2015/6_REMM/AvgTravelTime.csv'
        z = pd.read_csv(file).set_index('TAZ')
    elif year in range(2020,2028):
        file = settings['tdm']['main_dir'] + '2022/6_REMM/AvgTravelTime.csv'
        z = pd.read_csv(file).set_index('TAZ')
    elif year in range(2028,2036):
        file = settings['tdm']['main_dir'] + '2027/6_REMM/AvgTravelTime.csv'
        z = pd.read_csv(file).set_index('TAZ')
    #elif year in range(2031,2035):
    #    file = settings['tdm']['main_dir'] + '2030/' + settings['tdm']['output_volume']
    #    z = wfrc_utils.dbf2df(file).set_index('TAZZ')
    elif year in range(2036,2046):
        file = settings['tdm']['main_dir'] + '2035/6_REMM/AvgTravelTime.csv'
        z = pd.read_csv(file).set_index('TAZ')
    elif year in range(2046,2099):
        file = settings['tdm']['main_dir'] + '2045/6_REMM/AvgTravelTime.csv'
        z = pd.read_csv(file).set_index('TAZ')
    else:
        file = settings['tdm']['main_dir'] + '2011/6_REMM/AvgTravelTime.csv'
        z = pd.read_csv(file).set_index('TAZ')
    return z.TIMEAuto
    '''
@sim.column('zones', 'res_units_within_10_min', cache=True)
def res_units_within_30_min(buildings, travel_data):
    b = buildings.to_frame()
    b = b[['zone_id','residential_units']].groupby('zone_id').sum()
    b = b[b.index>0]
    b = pd.Series(data=b.residential_units, index=b.index)
    td = travel_data.to_frame()
    ru10 = misc.compute_range(td, b, "travel_time", 10, agg=np.sum)
    return ru10.fillna(0).astype('int')'''

    
#################
#BLOCKS VARIABLES
#################

@sim.column("blocks", "sum_land_value")
def sum_land_value(parcels, blocks):
    return parcels.land_value.groupby(parcels.block_id).sum().\
        reindex(blocks.index).fillna(0)
        
#####################
#HOUSEHOLDS VARIABLES
#####################

@sim.column("households", "building_id")
def building_id(households):
    return households.local.building_id.astype("int")
    
@sim.column('households', 'parcel_id')
def parcel_id(households, buildings):
    df = misc.reindex(buildings.parcel_id, households.building_id)
    return df.fillna(-1).astype("int")
    
@sim.column('households', 'proportion_workers', cache=True)
def proportion_workers(households):
    return households.workers / households.persons
    
@sim.column('households', 'county_id')
def county_id(households, buildings):
    df = misc.reindex(buildings.county_id, households.building_id)
    return df.fillna(-1).astype("int")
    
@sim.column("households", "zone_id")
def zone_id(households, buildings):
    df = misc.reindex(buildings.zone_id, households.building_id)
    return df.fillna(-1).astype("int")

@sim.column("households", "node_id")
def node_id(households, buildings):
    df = misc.reindex(buildings.node_id, households.building_id)
    return df.fillna(-1).astype("int")
    
@sim.column('households', 'distlrg_id')
def distlrg_id(households, buildings):
    df = misc.reindex(buildings.distlrg_id, households.building_id)
    return df.fillna(-1).astype("int")

@sim.column('households', 'distmed_id')
def distmed_id(households, buildings):
    df = misc.reindex(buildings.distmed_id, households.building_id)
    return df.fillna(-1).astype("int")
    
@sim.column('households', 'distsml_id')
def distsml_id(households, buildings):
    df = misc.reindex(buildings.distsml_id, households.building_id)
    return df.fillna(-1).astype("int")

@sim.column('households', 'income_quartile')
def income_quartile(households):
    #q = pd.Series(pd.cut(households.income, [0,33060,59300,91940,1000000], labels=[1,2,3,4],), index=households.index).fillna(1)
    q = pd.Series(pd.cut(households.income, [0, 31806, 54525, 90875, 100000000], labels=[1,2,3,4],), index=households.index).fillna(1)
    return q.astype('int')

@sim.column('households', 'b_year_built')
def b_year_built(households, buildings):
    #q = pd.Series(pd.cut(households.income, [0,33060,59300,91940,1000000], labels=[1,2,3,4],), index=households.index).fillna(1)
    df = misc.reindex(buildings.year_built, households.building_id)
    return df.fillna(-1).astype("int")

####################################
#HOUSEHOLDS FOR ESTIMATION VARIABLES
####################################

@sim.column('households_for_estimation', 'parcel_id', cache=True)
def node_id(households_for_estimation, buildings):
    return misc.reindex(buildings.parcel_id, households_for_estimation.building_id)

@sim.column('households_for_estimation', 'node_id', cache=True)
def node_id(households_for_estimation, parcels):
    return misc.reindex(parcels.node_id, households_for_estimation.parcel_id)
    
@sim.column('households_for_estimation', 'zone_id', cache=True)
def zone_id(households_for_estimation, parcels):
    return misc.reindex(parcels.zone_id, households_for_estimation.parcel_id)
    
@sim.column('households_for_estimation', 'county_id', cache=True)
def zone_id(households_for_estimation, buildings):
    return misc.reindex(buildings.county_id, households_for_estimation.parcel_id).fillna(0)
    
@sim.column('households_for_estimation', 'proportion_workers', cache=True)
def proportion_workers(households_for_estimation):
    return households_for_estimation.workers / households_for_estimation.persons

@sim.column('households_for_estimation', 'income_quartile', cache=True)
def income_quartile(households_for_estimation):
    q = pd.Series(pd.cut(households_for_estimation.income, [0,33060,59300,91940,1000000], labels=[1,2,3,4],), index=households_for_estimation.index).fillna(1)
    return q.astype('int')
    
#########################
#LARGE DISTRICT VARIABLES
#########################

@sim.column("distlrg", "distlrg_median_income")
def distlrg_median_income(households, distlrg):
    return households.income.groupby(households.distlrg_id).median().\
        reindex(distlrg.index).fillna(0)

##########################
#MEDIUM DISTRICT VARIABLES
##########################

@sim.column("distmed", "distmed_median_income")
def distmed_median_income(households, distmed):
    return households.income.groupby(households.distmed_id).median().\
        reindex(distmed.index).fillna(0)

#########################
#SMALL DISTRICT VARIABLES
#########################

@sim.column("distsml", "distsml_median_income")
def distsml_median_income(households, distsml):
    return households.income.groupby(households.distsml_id).median().\
        reindex(distsml.index).fillna(0)
    
    
###Zonal Model Variables

@sim.column("zones", "res_median_price", cache=True)
def res_median_price(buildings, zones, settings):
    pr = buildings.res_price_per_sqft[(buildings.building_type_id.\
        isin([1,2]))&(buildings.res_price_per_sqft>0)].groupby(buildings.zone_id).median().reindex(zones.index).fillna(0)
    return pr

@sim.column("zones", "ofc_median_price", cache=True)
def ofc_median_price(buildings, zones, settings):
    pr = buildings.unit_price_non_residential[(buildings.building_type_id.\
    isin([5]))&(buildings.unit_price_non_residential>0)].groupby(buildings.zone_id).median().reindex(zones.index).fillna(0)
    return pr

@sim.column("zones", "ret_median_price", cache=True)
def ret_median_price(buildings, zones, settings):
    pr = buildings.unit_price_non_residential[(buildings.building_type_id.\
    isin([4]))&(buildings.unit_price_non_residential>0)].groupby(buildings.zone_id).median().reindex(zones.index).fillna(0)
    return pr
    
@sim.column("zones", "ind_median_price", cache=True)
def ind_median_price(buildings, zones, settings):
    pr = buildings.unit_price_non_residential[(buildings.building_type_id.\
    isin([3]))&(buildings.unit_price_non_residential>0)].groupby(buildings.zone_id).median().reindex(zones.index).fillna(0)
    return pr

@sim.column('zones', 'seg_col')
def seg_col(zones):
    z = zones.to_frame('tdm_county_id')
    z['seg_col'] = 1
    return z['seg_col']

