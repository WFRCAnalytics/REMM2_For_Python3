import pandas as pd, numpy as np, pysal as ps
import time
import string
import os
import _pickle as pickle
import utils as wfrc_utils
import orca.orca as sim
from urbansim.utils import misc
from urbansim_defaults import utils
from urbansim_defaults import models
from urbansim.developer import sqftproforma, developer
import subprocess
import gc
    

def hedonic_export(hedonic_output, filepath):
    h_dict = hedonic_output.to_dict()
    sm = h_dict['models'].keys()
    h_est = pd.DataFrame(columns=['Submodel','Variable','Coefficient','T-Score','Std. Error','R-Squared','Adj. R-Squared'])
    for sub in h_dict['models'].keys():	
        params = h_dict['models'][sub]['fit_parameters']   
        vars = params['Coefficient'].keys()
        temp = pd.DataFrame(columns=(['Variable','R-Squared','Adj. R-Squared'] + params.keys()))
        for v in vars:
            row = dict()
            row['Variable'] = v
            for p in params.keys():
                row[p] = params[p][v]
            temp = temp.append(row, ignore_index=True)
        temp['Submodel'] = sub
        temp['R-Squared'].loc[0] = h_dict['models'][sub]['fit_rsquared']
        temp['Adj. R-Squared'].loc[0] = h_dict['models'][sub]['fit_rsquared_adj']
        h_est = pd.concat([h_est, temp])
    h_est.set_index(['Submodel', h_est.index], inplace=True)
    ts = string.replace(time.asctime(), ':','-')
    h_est.to_csv(filepath + ts + '.csv', columns=['R-Squared','Adj. R-Squared','Variable','Coefficient','T-Score','Std. Error'])

def lcm_export(lcm_output, filepath):
    lcm_dict = lcm_output.to_dict()
    sm = lcm_dict['models'].keys()
    lcm_est = pd.DataFrame(columns=['Submodel','Variable','Coefficient','T-Score','Std. Error','Null Log-likelihood','Log-likelihood at Convergence','Log-likelihood Ratio'])
    for sub in lcm_dict['models'].keys():	
        params = lcm_dict['models'][sub]['fit_parameters']   
        vars = params['Coefficient'].keys()
        temp = pd.DataFrame(columns=(['Variable','Null Log-likelihood','Log-likelihood at Convergence','Log-likelihood Ratio'] + params.keys()))
        for v in vars:
            row = dict()
            row['Variable'] = v
            for p in params.keys():
                row[p] = params[p][v]
            temp = temp.append(row, ignore_index=True)
        temp['Submodel'] = sub
        temp['Log-likelihood at Convergence'].loc[0] = lcm_dict['models'][sub]['log_likelihoods']['convergence']
        temp['Null Log-likelihood'].loc[0] = lcm_dict['models'][sub]['log_likelihoods']['null']
        temp['Log-likelihood Ratio'].loc[0] = lcm_dict['models'][sub]['log_likelihoods']['ratio']
        lcm_est = pd.concat([lcm_est, temp])
    lcm_est.set_index(['Submodel', lcm_est.index], inplace=True)
    lcm_est.to_csv(filepath, mode = 'a', header = False, columns=['Null Log-likelihood','Log-likelihood at Convergence','Log-likelihood Ratio','Variable','Coefficient','T-Score','Std. Error'])

@sim.step('garbage_collect')
def garbage_collect():
    gc.collect()
    
@sim.step('nrh_estimate')
def nrh_estimate(buildings_for_estimation_grouped, aggregations):
    nrh_ofc = utils.hedonic_estimate("nrh_ofc.yaml", buildings_for_estimation_grouped, aggregations)
    hedonic_export(nrh_ofc, 'configs/nrh_ofc-')
    nrh_ret = utils.hedonic_estimate("nrh_ret.yaml", buildings_for_estimation_grouped, aggregations)
    hedonic_export(nrh_ret, 'configs/nrh_ret-')
    nrh_ind = utils.hedonic_estimate("nrh_ind.yaml", buildings_for_estimation_grouped, aggregations)
    hedonic_export(nrh_ind, 'configs/nrh_ind-')
    return nrh_ofc
    return nrh_ret
    return nrh_ind
    
@sim.step('nrh_estimate_slc')
def nrh_estimate_slc(buildings_for_estimation_grouped, aggregations):
    nrh_ofc_slc = utils.hedonic_estimate("nrh_ofc_slc.yaml", buildings_for_estimation_grouped, aggregations)
    hedonic_export(nrh_ofc_slc, 'configs/nrh_ofc_slc-')
    nrh_ret_slc = utils.hedonic_estimate("nrh_ret_slc.yaml", buildings_for_estimation_grouped, aggregations)
    hedonic_export(nrh_ret_slc, 'configs/nrh_ret_slc-')
    nrh_ind_slc = utils.hedonic_estimate("nrh_ind_slc.yaml", buildings_for_estimation_grouped, aggregations)
    hedonic_export(nrh_ind_slc, 'configs/nrh_ind_slc-')
    return nrh_ofc_slc
    return nrh_ret_slc
    return nrh_ind_slc

@sim.step('nrh_estimate_davis')
def nrh_estimate_davis(buildings_for_estimation_grouped, aggregations):
    nrh_ofc_davis = utils.hedonic_estimate("nrh_ofc_davis.yaml", buildings_for_estimation_grouped, aggregations)
    hedonic_export(nrh_ofc_davis, 'configs/nrh_ofc_davis-')
    nrh_ret_davis = utils.hedonic_estimate("nrh_ret_davis.yaml", buildings_for_estimation_grouped, aggregations)
    hedonic_export(nrh_ret_davis, 'configs/nrh_ret_davis-')
    nrh_ind_davis = utils.hedonic_estimate("nrh_ind_davis.yaml", buildings_for_estimation_grouped, aggregations)
    hedonic_export(nrh_ind_davis, 'configs/nrh_ind_davis-')
    return nrh_ofc_davis
    return nrh_ret_davis
    return nrh_ind_davis
    
@sim.step('nrh_estimate_weber')
def nrh_estimate_weber(buildings_for_estimation_grouped, aggregations):
    nrh_ofc_weber = utils.hedonic_estimate("nrh_ofc_weber.yaml", buildings_for_estimation_grouped, aggregations)
    hedonic_export(nrh_ofc_weber, 'configs/nrh_ofc_weber-')
    nrh_ret_weber = utils.hedonic_estimate("nrh_ret_weber.yaml", buildings_for_estimation_grouped, aggregations)
    hedonic_export(nrh_ret_weber, 'configs/nrh_ret_weber-')
    nrh_ind_weber = utils.hedonic_estimate("nrh_ind_weber.yaml", buildings_for_estimation_grouped, aggregations)
    hedonic_export(nrh_ind_weber, 'configs/nrh_ind_weber-')
    return nrh_ofc_weber
    return nrh_ret_weber
    return nrh_ind_weber

@sim.step('nrh_estimate_utah')
def nrh_estimate_utah(buildings_for_estimation_grouped, aggregations):
    nrh_ofc_utah = utils.hedonic_estimate("nrh_ofc_utah.yaml", buildings_for_estimation_grouped, aggregations)
    hedonic_export(nrh_ofc_utah, 'configs/nrh_ofc_utah-')
    nrh_ret_utah = utils.hedonic_estimate("nrh_ret_utah.yaml", buildings_for_estimation_grouped, aggregations)
    hedonic_export(nrh_ret_utah, 'configs/nrh_ret_utah-')
    nrh_ind_utah = utils.hedonic_estimate("nrh_ind_utah.yaml", buildings_for_estimation_grouped, aggregations)
    hedonic_export(nrh_ind_utah, 'configs/nrh_ind_utah-')
    return nrh_ofc_utah
    return nrh_ret_utah
    return nrh_ind_utah
    
"""@sim.step('nrh_estimate_slc')
def nrh_estimate(buildings, aggregations):
    nrh_mu = utils.hedonic_estimate("nrh_mu_slc.yaml", buildings, aggregations)
    nrh_oth = utils.hedonic_estimate("nrh_oth_slc.yaml", buildings, aggregations)
    return nrh_mu
    return nrh_oth"""
    
@sim.step('nrh_ind_simulate')
def nrh_ind_simulate(buildings, aggregations):
    nrh_ind_davis =  utils.hedonic_simulate("nrh_ind_davis.yaml", buildings, aggregations,
                                  "unit_price_non_residential")
    nrh_ind_weber = utils.hedonic_simulate("nrh_ind_weber.yaml", buildings, aggregations,
                                  "unit_price_non_residential")
    nrh_ind_slc = utils.hedonic_simulate("nrh_ind_slc.yaml", buildings, aggregations,
                                  "unit_price_non_residential")
    nrh_ind_utah = utils.hedonic_simulate("nrh_ind_utah.yaml", buildings, aggregations,
                                  "unit_price_non_residential")
    return 1

@sim.step('nrh_ofc_simulate')
def nrh_ofc_simulate(buildings, aggregations):
    nrh_ofc_davis = utils.hedonic_simulate("nrh_ofc_davis.yaml", buildings, aggregations,
                                  "unit_price_non_residential")
    nrh_ofc_weber = utils.hedonic_simulate("nrh_ofc_weber.yaml", buildings, aggregations,
                                  "unit_price_non_residential")
    nrh_ofc_slc = utils.hedonic_simulate("nrh_ofc_slc.yaml", buildings, aggregations,
                                  "unit_price_non_residential")
    nrh_ofc_utah = utils.hedonic_simulate("nrh_ofc_utah.yaml", buildings, aggregations,
                                  "unit_price_non_residential")
    return 1
                                                                    
@sim.step('nrh_ret_simulate')
def nrh_ret_simulate(buildings, aggregations):
    nrh_ret_davis = utils.hedonic_simulate("nrh_ret_davis.yaml", buildings, aggregations,
                                  "unit_price_non_residential")
    nrh_ret_weber = utils.hedonic_simulate("nrh_ret_weber.yaml", buildings, aggregations,
                                  "unit_price_non_residential")
    nrh_ret_slc = utils.hedonic_simulate("nrh_ret_slc.yaml", buildings, aggregations,
                                  "unit_price_non_residential")
    nrh_ret_utah = utils.hedonic_simulate("nrh_ret_utah.yaml", buildings, aggregations,
                                  "unit_price_non_residential")
    return 1
                                                                    
@sim.step('nrh_mu_oth_simulate')
def nrh_simulate(buildings, aggregations):
    return utils.hedonic_simulate("nrh_mu_oth.yaml", buildings, aggregations,
                                  "unit_price_non_residential")
    
@sim.step('rsh_estimate')
def rsh_estimate(buildings, aggregations):
    rsh_sf = utils.hedonic_estimate("rsh_sf.yaml", buildings, aggregations)
    hedonic_export(rsh_sf, 'configs/rsh_sf-')
    rsh_mf = utils.hedonic_estimate("rsh_mf.yaml", buildings, aggregations)
    hedonic_export(rsh_mf, 'configs/rsh_mf-')
    return rsh_sf
    return rsh_mf
    
@sim.step('rsh_estimate_slc')
def rsh_estimate_slc(buildings, aggregations):
    rsh_sf_slc = utils.hedonic_estimate("rsh_sf_slc.yaml", buildings, aggregations)
    hedonic_export(rsh_sf_slc, 'configs/rsh_sf_slc-')
    rsh_mf_slc = utils.hedonic_estimate("rsh_mf_slc.yaml", buildings, aggregations)
    hedonic_export(rsh_mf_slc, 'configs/rsh_mf_slc-')
    return rsh_sf_slc
    return rsh_mf_slc

@sim.step('rsh_estimate_davis')
def rsh_estimate_davis(buildings, aggregations):
    rsh_sf_davis = utils.hedonic_estimate("rsh_sf_davis.yaml", buildings, aggregations)
    hedonic_export(rsh_sf_davis, 'configs/rsh_sf_davis-')
    rsh_mf_davis = utils.hedonic_estimate("rsh_mf_davis.yaml", buildings, aggregations)
    hedonic_export(rsh_mf_davis, 'configs/rsh_mf_davis-')
    return rsh_sf_davis
    return rsh_mf_davis
    
@sim.step('rsh_estimate_weber')
def rsh_estimate_weber(buildings, aggregations):
    rsh_sf_weber = utils.hedonic_estimate("rsh_sf_weber.yaml", buildings, aggregations)
    hedonic_export(rsh_sf_weber, 'configs/rsh_sf_weber-')
    rsh_mf_weber = utils.hedonic_estimate("rsh_mf_weber.yaml", buildings, aggregations)
    hedonic_export(rsh_mf_weber, 'configs/rsh_mf_weber-')
    return rsh_sf_weber
    return rsh_mf_weber    

@sim.step('rsh_estimate_utah')
def rsh_estimate_utah(buildings, aggregations):
    rsh_sf_utah = utils.hedonic_estimate("rsh_sf_utah.yaml", buildings, aggregations)
    hedonic_export(rsh_sf_utah, 'configs/rsh_sf_utah-')
    rsh_mf_utah = utils.hedonic_estimate("rsh_mf_utah.yaml", buildings, aggregations)
    hedonic_export(rsh_mf_utah, 'configs/rsh_mf_utah-')
    return rsh_sf_utah
    return rsh_mf_utah
    
@sim.step('rsh_sf_simulate')
def rsh_simulate(buildings, aggregations):
    rsh_sf_davis = utils.hedonic_simulate("rsh_sf_davis.yaml", buildings, aggregations,
                                  "res_price_per_sqft")
    rsh_sf_weber = utils.hedonic_simulate("rsh_sf_weber.yaml", buildings, aggregations,
                                  "res_price_per_sqft")
    rsh_sf_slc = utils.hedonic_simulate("rsh_sf_slc.yaml", buildings, aggregations,
                                  "res_price_per_sqft")
    rsh_sf_utah = utils.hedonic_simulate("rsh_sf_utah.yaml", buildings, aggregations,
                                  "res_price_per_sqft")
    return 1

@sim.step('rsh_mf_simulate')
def rsh_simulate(buildings, aggregations):
    rsh_mf_davis = utils.hedonic_simulate("rsh_mf_davis.yaml", buildings, aggregations,
                                  "res_price_per_sqft")
    rsh_mf_weber = utils.hedonic_simulate("rsh_mf_weber.yaml", buildings, aggregations,
                                  "res_price_per_sqft")
    rsh_mf_slc = utils.hedonic_simulate("rsh_mf_slc.yaml", buildings, aggregations,
                                  "res_price_per_sqft")
    rsh_mf_utah = utils.hedonic_simulate("rsh_mf_utah.yaml", buildings, aggregations,
                                  "res_price_per_sqft")
    return 1
    
@sim.step('elcm_estimate')
def elcm_estimate(jobs, buildings, aggregations):
    elcm = utils.lcm_estimate("elcm.yaml", jobs, "building_id",
                              buildings, aggregations)
    lcm_export(elcm, 'configs/elcm_estimation.csv')
    return elcm
    

@sim.step('hlcm_estimate')
def hlcm_estimate(households_for_estimation, buildings, aggregations):
    hlcm = utils.lcm_estimate("hlcm.yaml", households_for_estimation, "building_id",
                              buildings, aggregations)
    #lcm_export(hlcm, 'configs/hlcm_estimation.csv')
    return hlcm
    
@sim.step('hlcm_estimate_slc')
def hlcm_estimate(households, buildings, aggregations):
    hlcm = utils.lcm_estimate("hlcm_slc.yaml", households, "building_id",
                              buildings, aggregations)
    lcm_export(hlcm, 'configs/hlcm_estimation_slc.csv')
    return hlcm
                              
@sim.step('hlcm_estimate_utah')
def hlcm_estimate(households, buildings, aggregations):
    hlcm = utils.lcm_estimate("hlcm_utah.yaml", households, "building_id",
                              buildings, aggregations)
    lcm_export(hlcm, 'configs/hlcm_estimation_utah.csv')
    return hlcm
                              
@sim.step('hlcm_estimate_dw')
def hlcm_estimate(households, buildings, aggregations):
    hlcm = utils.lcm_estimate("hlcm_dw.yaml", households, "building_id",
                              buildings, aggregations)
    lcm_export(hlcm, 'configs/hlcm_estimation_dw.csv')
    return hlcm
                              
@sim.step('hlcm_simulate_slc')
def hlcm_simulate_slc(households, buildings, aggregations, settings):
    return wfrc_utils.lcm_simulate("hlcm_slc.yaml", households, buildings,
                              aggregations,
                              "building_id", "residential_units",
                              "vacant_residential_units",
                              settings.get("enable_supply_correction", None))

@sim.step('hlcm_simulate_utah')
def hlcm_simulate_utah(households, buildings, aggregations, settings):
    return wfrc_utils.lcm_simulate("hlcm_utah.yaml", households, buildings,
                              aggregations,
                              "building_id", "residential_units",
                              "vacant_residential_units",
                              settings.get("enable_supply_correction", None))
                              
@sim.step('hlcm_simulate_davis')
def hlcm_simulate_davis(households, buildings, aggregations, settings):
    return wfrc_utils.lcm_simulate("hlcm_davis.yaml", households, buildings,
                              aggregations,
                              "building_id", "residential_units",
                              "vacant_residential_units",
                              settings.get("enable_supply_correction", None))

@sim.step('hlcm_simulate_weber')
def hlcm_simulate_weber(households, buildings, aggregations, settings):
    return utils.lcm_simulate("hlcm_weber.yaml", households, buildings,
                              aggregations,
                              "building_id", "residential_units",
                              "vacant_residential_units",
                              settings.get("enable_supply_correction", None))
                              
@sim.step('elcm_simulate_slc')
def elcm_simulate_slc(jobs, buildings, aggregations):
    return wfrc_utils.lcm_simulate("elcm_slc.yaml", jobs, buildings, aggregations,
                              "building_id", "job_spaces",
                              "vacant_job_spaces")

@sim.step('elcm_simulate_utah')
def elcm_simulate_utah(jobs, buildings, aggregations):
    return wfrc_utils.lcm_simulate("elcm_utah.yaml", jobs, buildings, aggregations,
                              "building_id", "job_spaces",
                              "vacant_job_spaces")

@sim.step('elcm_simulate_davis')
def elcm_simulate_davis(jobs, buildings, aggregations):
    return wfrc_utils.lcm_simulate("elcm_davis.yaml", jobs, buildings, aggregations,
                              "building_id", "job_spaces",
                              "vacant_job_spaces")

@sim.step('elcm_simulate_weber')
def elcm_simulate_weber(jobs, buildings, aggregations):
    return wfrc_utils.lcm_simulate("elcm_weber.yaml", jobs, buildings, aggregations,
                              "building_id", "job_spaces",
                              "vacant_job_spaces")
'''
@sim.step('trend_calibration')
def trend_calibration(year):
    if year <= 2026:
        
        adjusted_step = 1
    
        hhtargetcsv = pd.read_csv('./data/calibration/LRGDISTTrendHH.csv', index_col='District')
        jobtargetcsv = pd.read_csv('./data/calibration/LRGDISTTrendJOB.csv', index_col='District')
    
        hhtargets = hhtargetcsv[str(year)].to_dict()
        jobtargets = jobtargetcsv[str(year)].to_dict()
        
        tothhtarget = sum(hhtargets.values())
        totjobtarget = sum(jobtargets.values())
        
        hhtargetweber = hhtargets[1] + hhtargets[2]
        hhtargetdavis = hhtargets[3] + hhtargets[4]
        hhtargetsl = hhtargets[5] + hhtargets[6] +hhtargets[7] + hhtargets[8] + hhtargets[9]+hhtargets[10] + hhtargets[11]
        hhtargetutah = hhtargets[12] + hhtargets[13] +hhtargets[14] + hhtargets[15]
        
        jobtargetweber = jobtargets[1] + jobtargets[2]
        jobtargetdavis = jobtargets[3] + jobtargets[4]
        jobtargetsl = jobtargets[5] + jobtargets[6] +jobtargets[7] + jobtargets[8] + jobtargets[9]+jobtargets[10] + jobtargets[11]
        jobtargetutah = jobtargets[12] + jobtargets[13] +jobtargets[14] + jobtargets[15]           
        
        
    
        households = sim.get_table("households").to_frame(["building_id","distlrg_id"]).groupby("distlrg_id").building_id.count().to_dict()
        if -1 in households: del households[-1]
        tothh = sum(households.values())
        
        jobs = sim.get_table("jobs").to_frame(["building_id","distlrg_id"]).groupby("distlrg_id").building_id.count().to_dict()
        if -1 in jobs: del jobs[-1]
        totjob = sum(jobs.values())
        
        hhweber = households[1] + households[2]
        hhdavis = households[3] + households[4]
        hhsl = households[5] + households[6] +households[7] + households[8] + households[9]+households[10] + households[11]
        hhutah = households[12] + households[13] +households[14] + households[15]
        
        jobweber = jobs[1] + jobs[2]
        jobdavis = jobs[3] + jobs[4]
        jobsl = jobs[5] + jobs[6] +jobs[7] + jobs[8] + jobs[9]+jobs[10] + jobs[11]
        jobutah = jobs[12] + jobs[13] +jobs[14] + jobs[15]        
        

        calibration_shifters = pd.read_csv('./data/calibration/distlrg_shifters.csv').set_index('distlrg_id')
        new_calib_shifters = pd.DataFrame(index = calibration_shifters.index)
        calibration_shifters = calibration_shifters.to_dict()
        res_shifters = calibration_shifters["res_price_shifter"]
        nonres_shifters = calibration_shifters["nonres_price_shifter"]
        res_new_shifters = {}
        nonres_new_shifters = {}
        res_fraction = {}
        nonres_fraction = {}

        for district in res_shifters.keys():
            ##residential    
            #shift = shifters[district]
            #hhtarget = int(hhtargets[district]*float(tothh)/tothhtarget)
            if (district ==1) | (district == 2):
                hhtarget = int(hhtargets[district]*float(hhweber)/hhtargetweber)
            elif (district == 3) | (district == 4):
                hhtarget = int(hhtargets[district]*float(hhdavis)/hhtargetdavis)
            elif (district >=5) & (district <= 11):
                hhtarget = int(hhtargets[district]*float(hhsl)/hhtargetsl)
            elif (district >= 12) & (district <= 15):
                hhtarget = int(hhtargets[district]*float(hhutah)/hhtargetutah)
            
            hhsimulated = households[district] if district in households else 0.0
            
   
            
            if hhtarget <= hhsimulated:
                #new_shift = 1 + (float(hhtarget)/hhsimulated - 1)*adjusted_step
                #new_shift = 0.0406*np.exp(3.1666*new_shift)/0.963303060258791
                #new_shift = 1/(1e-34*np.exp((2-float(hhtarget)/hhsimulated)*78.24)/0.953235623617922)
                new_shift = -np.power((1-float(hhtarget)/hhsimulated),0.1)+1
            else:
                new_shift = 1 + (float(hhtarget)/hhsimulated - 1)*adjusted_step
                new_shift = 1e-34*np.exp(new_shift*78.24)/0.953235623617922
            if (district == 7) & (year == 2011):
                new_shift = 0.5
            #if new_shift > 1.1:
            #    new_shift = 1.1
            #if new_shift < 0.9:
            #    new_shift = 0.9
            res_new_shifters[district] = new_shift
            res_fraction[district] = float(hhtarget)/hhsimulated
            #nonresidential
            #shift = shifters[district]
            #jobtarget = int(jobtargets[district]*float(totjob)/totjobtarget)
            
            if (district ==1) | (district == 2):
                jobtarget = int(jobtargets[district]*float(jobweber)/jobtargetweber)
            elif (district == 3) | (district == 4):
                jobtarget = int(jobtargets[district]*float(jobdavis)/jobtargetdavis)
            elif (district >= 5) & (district <= 11):
                jobtarget = int(jobtargets[district]*float(jobsl)/jobtargetsl)
            elif (district >= 12) & (district <= 15):
                jobtarget = int(jobtargets[district]*float(jobutah)/jobtargetutah)
                
            #if (district ==1) | (district == 2):
            #    adjusted_step=1
            #elif (district == 3) | (district == 4):
            #    adjusted_step=0.5
            #elif (district >= 5) & (district <= 11):
            #    adjusted_step=2
            #elif (district >= 12) & (district <= 15):
            #    adjusted_step=3
                
                
            jobsimulated = jobs[district] if district in jobs else 0.0
            
            if jobtarget <= jobsimulated:
                #new_shift = 1 + (float(jobtarget)/jobsimulated - 1)*adjusted_step
                #new_shift = 0.0406*np.exp(3.1666*new_shift)/0.963303060258791
                #new_shift = 1/(1e-34*np.exp((2-float(jobtarget)/jobsimulated)*78.24)/0.953235623617922)
                new_shift = -np.power((1-float(jobtarget)/jobsimulated),0.1)+1
            else:
                new_shift = 1 + (float(jobtarget)/jobsimulated - 1)*adjusted_step
                new_shift = 1e-34*np.exp(new_shift*78.24)/0.953235623617922
            #if new_shift > 1.1:
            #    new_shift = 1.1
            #if new_shift < 0.9:
            #    new_shift = 0.9
            nonres_new_shifters[district] = new_shift
            nonres_fraction[district] = float(jobtarget)/jobsimulated
    
    
        res_new_shifters = pd.Series(res_new_shifters, name = "res_price_shifter")
        res_new_shifters.index.name = 'distlrg_id'

        nonres_new_shifters = pd.Series(nonres_new_shifters, name = "nonres_price_shifter")
        nonres_new_shifters.index.name = 'distlrg_id'
        
        res_new_fraction = pd.Series(res_fraction, name = "res_fraction")
        res_new_fraction.index.name = 'distlrg_id'

        nonres_new_fraction = pd.Series(nonres_fraction, name = "nonres_fraction")
        nonres_new_fraction.index.name = 'distlrg_id'
        
        res_new_fraction.to_frame().to_csv('./data/calibration/res_fraction' + str(year) + '.csv')
        nonres_new_fraction.to_frame().to_csv('./data/calibration/nonres_fraction' + str(year) + '.csv')

        new_calib_shifters["res_price_shifter"] = res_new_shifters
        new_calib_shifters["nonres_price_shifter"] = nonres_new_shifters

    else:
        calibration_shifters = pd.read_csv('./data/calibration/distlrg_shifters.csv').set_index('distlrg_id')
        new_calib_shifters = pd.DataFrame(index = calibration_shifters.index)
        calibration_shifters = calibration_shifters.to_dict()
        res_shifters = calibration_shifters["res_price_shifter"]
        res_new_shifters = {}
        nonres_new_shifters = {}
        for district in res_shifters.keys():
            res_new_shifters[district] = 1
            nonres_new_shifters[district] = 1
         
        res_new_shifters = pd.Series(res_new_shifters, name = "res_price_shifter")
        res_new_shifters.index.name = 'distlrg_id'

        nonres_new_shifters = pd.Series(nonres_new_shifters, name = "nonres_price_shifter")
        nonres_new_shifters.index.name = 'distlrg_id'

        new_calib_shifters["res_price_shifter"] = res_new_shifters
        new_calib_shifters["nonres_price_shifter"] = nonres_new_shifters
        
    new_calib_shifters.to_csv('./data/calibration/distlrg_shifters.csv')
    new_calib_shifters.to_csv('./data/calibration/distlrg_shifters' + str(year) + '.csv')
        
        
'''
@sim.step('pipeline_projects')
def pipeline_projects(year):
    bt = sim.get_table("buildings")
    b = bt.to_frame(bt.local_columns)
    hht = sim.get_table("households")
    hh = hht.to_frame(hht.local_columns)
    jobst = sim.get_table("jobs")
    jobs = jobst.to_frame(jobst.local_columns)
    pipeline = pd.read_csv("data\pipeline_buildings.csv")
    pipeline = pipeline[(pipeline.year_built == year)]
    demolition = pipeline[(pipeline.DEVTYPE == "R") | (pipeline.DEVTYPE == "D")]
    demolition_buildings = b.parcel_id.isin(demolition.parcel_id)
    building_step_1 = b[np.logical_not(demolition_buildings)]
    delete_hh = hh.building_id.isin(b[demolition_buildings].index)
    hh.building_id[delete_hh] = -1
    delete_jobs = jobs.building_id.isin(b[demolition_buildings].index)
    jobs.building_id[delete_jobs] = -1
    construction = pipeline[(pipeline.DEVTYPE != "D")]
    construction = construction[bt.local_columns]
    maxind = np.max(building_step_1.index.values)
    construction = construction.reset_index(drop=True)
    construction.index = construction.index + maxind + 1
    building_step_2 = pd.concat([building_step_1, construction], verify_integrity=True)
    building_step_2.index.name = 'building_id'
    sim.add_table("households", hh)
    sim.add_table("jobs", jobs)
    sim.add_table("buildings", building_step_2)

@sim.step('feasibility')
def feasibility(parcels, settings,
                parcel_sales_price_sqft_func,
                parcel_is_allowed_func):
    kwargs = settings['feasibility']
    pfc = sqftproforma.SqFtProFormaConfig()
    for type in pfc.costs.keys():
        pfc.costs[type] = np.multiply(pfc.costs[type], .7284)
    for ptype in pfc.parking_cost_d.keys():
        pfc.parking_cost_d[ptype] = np.multiply(pfc.parking_cost_d[ptype], .7284)
    pfc.costs['residential'] = [90.0,110.0,120.0,140.0]
    #pfc.costs['retail'] = [14.0,17.0,20.0,23.0]
    #pfc.costs['industrial'] = [14.0,17.0,20.0,23.0]
    #pfc.costs['office'] = [14.0,17.0,20.0,23.0]
    pfc.profit_factor = 1.1
    pfc.fars = [0.005,0.01,0.05,.1, .15, .2, .25, .3, .4, .5, .75, 1.0, 1.5, 1.8, 2.0, 2.25, 2.5, 2.75,
                     3.0, 3.25, 3.5, 3.75, 4.0, 4.5,
                     5.0, 5.5, 6.0, 6.5, 7.0, 9.0, 11.0]
    wfrc_utils.run_feasibility(parcels,
                          parcel_sales_price_sqft_func,
                          parcel_is_allowed_func,
                          config = pfc,
                          **kwargs)
                          
@sim.step('residential_developer_slc')
def residential_developer(feasibility, households_slc, buildings_slc, buildings, parcels_slc, year,
                          settings, summary, form_to_btype_func,
                          add_extra_columns_func):
    kwargs = settings['residential_developer']
    new_buildings = wfrc_utils.run_developer(
        "residential",
        households_slc,
        buildings_slc,
        buildings,
        "residential_units",
        parcels_slc.shape_area,
        parcels_slc.ave_sqft_per_unit,
        parcels_slc.total_residential_units,
        feasibility,
        year=year,
        form_to_btype_callback=form_to_btype_func,
        add_more_columns_callback=add_extra_columns_func,
        **kwargs)
        
    summary.add_parcel_output(new_buildings)
        
@sim.step('residential_developer_utah')
def residential_developer(feasibility, households_utah, buildings_utah, buildings, parcels_utah, year,
                          settings, summary, form_to_btype_func,
                          add_extra_columns_func):
    kwargs = settings['residential_developer']
    new_buildings = wfrc_utils.run_developer(
        "residential",
        households_utah,
        buildings_utah,
        buildings,
        "residential_units",
        parcels_utah.shape_area,
        parcels_utah.ave_sqft_per_unit,
        parcels_utah.total_residential_units,
        feasibility,
        year=year,
        form_to_btype_callback=form_to_btype_func,
        add_more_columns_callback=add_extra_columns_func,
        **kwargs)
        
    summary.add_parcel_output(new_buildings)
        
@sim.step('residential_developer_davis')
def residential_developer(feasibility, households_davis, buildings_davis, buildings, parcels_davis, year,
                          settings, summary, form_to_btype_func,
                          add_extra_columns_func):
    kwargs = settings['residential_developer']
    new_buildings = wfrc_utils.run_developer(
        "residential",
        households_davis,
        buildings_davis,
        buildings,
        "residential_units",
        parcels_davis.shape_area,
        parcels_davis.ave_sqft_per_unit,
        parcels_davis.total_residential_units,
        feasibility,
        year=year,
        form_to_btype_callback=form_to_btype_func,
        add_more_columns_callback=add_extra_columns_func,
        **kwargs)
        
    summary.add_parcel_output(new_buildings)
            
@sim.step('residential_developer_weber')
def residential_developer(feasibility, households_weber, buildings_weber, buildings, parcels_weber, year,
                          settings, summary, form_to_btype_func,
                          add_extra_columns_func):
    kwargs = settings['residential_developer']
    new_buildings = wfrc_utils.run_developer(
        "residential",
        households_weber,
        buildings_weber,
        buildings,
        "residential_units",
        parcels_weber.shape_area,
        parcels_weber.ave_sqft_per_unit,
        parcels_weber.total_residential_units,
        feasibility,
        year=year,
        form_to_btype_callback=form_to_btype_func,
        add_more_columns_callback=add_extra_columns_func,
        **kwargs)
        
    summary.add_parcel_output(new_buildings)
        
@sim.step('non_residential_developer_slc')
def non_residential_developer(feasibility, jobs_slc, buildings_slc, buildings, parcels_slc, year,
                              settings, summary, form_to_btype_func,
                              add_extra_columns_func):

    kwargs = settings['non_residential_developer']
    new_buildings = wfrc_utils.run_developer(
        ["office", "retail", "industrial"],
        jobs_slc,
        buildings_slc,
        buildings,
        "job_spaces",
        parcels_slc.shape_area,
        parcels_slc.ave_sqft_per_unit,
        parcels_slc.total_job_spaces,
        feasibility,
        year=year,
        form_to_btype_callback=form_to_btype_func,
        add_more_columns_callback=add_extra_columns_func,
        residential=False,
        **kwargs)

    summary.add_parcel_output(new_buildings)

@sim.step('industrial_developer_slc')
def industrial_developer_slc(feasibility, jobs_slc_ind, buildings_slc_ind, buildings, parcels_slc, year,
                              settings, summary, form_to_btype_func,
                              add_extra_columns_func):

    kwargs = settings['industrial_developer']
    new_buildings = wfrc_utils.run_developer(
        ["industrial"],
        jobs_slc_ind,
        buildings_slc_ind,
        buildings,
        "job_spaces",
        parcels_slc.shape_area,
        parcels_slc.ave_sqft_per_unit,
        parcels_slc.total_job_spaces,
        feasibility,
        year=year,
        form_to_btype_callback=form_to_btype_func,
        add_more_columns_callback=add_extra_columns_func,
        residential=False,
        **kwargs)

    summary.add_parcel_output(new_buildings)
    
@sim.step('industrial_developer_utah')
def industrial_developer_utah(feasibility, jobs_utah_ind, buildings_utah_ind, buildings, parcels_utah, year,
                              settings, summary, form_to_btype_func,
                              add_extra_columns_func):

    kwargs = settings['industrial_developer']
    new_buildings = wfrc_utils.run_developer(
        ["industrial"],
        jobs_utah_ind,
        buildings_utah_ind,
        buildings,
        "job_spaces",
        parcels_utah.shape_area,
        parcels_utah.ave_sqft_per_unit,
        parcels_utah.total_job_spaces,
        feasibility,
        year=year,
        form_to_btype_callback=form_to_btype_func,
        add_more_columns_callback=add_extra_columns_func,
        residential=False,
        **kwargs)

    summary.add_parcel_output(new_buildings)
    
@sim.step('industrial_developer_davis')
def industrial_developer_davis(feasibility, jobs_davis_ind, buildings_davis_ind, buildings, parcels_davis, year,
                              settings, summary, form_to_btype_func,
                              add_extra_columns_func):

    kwargs = settings['industrial_developer']
    new_buildings = wfrc_utils.run_developer(
        ["industrial"],
        jobs_davis_ind,
        buildings_davis_ind,
        buildings,
        "job_spaces",
        parcels_davis.shape_area,
        parcels_davis.ave_sqft_per_unit,
        parcels_davis.total_job_spaces,
        feasibility,
        year=year,
        form_to_btype_callback=form_to_btype_func,
        add_more_columns_callback=add_extra_columns_func,
        residential=False,
        **kwargs)

    summary.add_parcel_output(new_buildings)
    
@sim.step('industrial_developer_weber')
def industrial_developer_weber(feasibility, jobs_weber_ind, buildings_weber_ind, buildings, parcels_weber, year,
                              settings, summary, form_to_btype_func,
                              add_extra_columns_func):

    kwargs = settings['industrial_developer']
    new_buildings = wfrc_utils.run_developer(
        ["industrial"],
        jobs_weber_ind,
        buildings_weber_ind,
        buildings,
        "job_spaces",
        parcels_weber.shape_area,
        parcels_weber.ave_sqft_per_unit,
        parcels_weber.total_job_spaces,
        feasibility,
        year=year,
        form_to_btype_callback=form_to_btype_func,
        add_more_columns_callback=add_extra_columns_func,
        residential=False,
        **kwargs)

    summary.add_parcel_output(new_buildings)
    
@sim.step('retail_developer_slc')
def retail_developer_slc(feasibility, jobs_slc_ret, buildings_slc_ret, buildings, parcels_slc, year,
                              settings, summary, form_to_btype_func,
                              add_extra_columns_func):

    kwargs = settings['retail_developer']
    new_buildings = wfrc_utils.run_developer(
        ["retail"],
        jobs_slc_ret,
        buildings_slc_ret,
        buildings,
        "job_spaces",
        parcels_slc.shape_area,
        parcels_slc.ave_sqft_per_unit,
        parcels_slc.total_job_spaces,
        feasibility,
        year=year,
        form_to_btype_callback=form_to_btype_func,
        add_more_columns_callback=add_extra_columns_func,
        residential=False,
        **kwargs)

    summary.add_parcel_output(new_buildings)
    
@sim.step('retail_developer_utah')
def retail_developer_utah(feasibility, jobs_utah_ret, buildings_utah_ret, buildings, parcels_utah, year,
                              settings, summary, form_to_btype_func,
                              add_extra_columns_func):

    kwargs = settings['retail_developer']
    new_buildings = wfrc_utils.run_developer(
        ["retail"],
        jobs_utah_ret,
        buildings_utah_ret,
        buildings,
        "job_spaces",
        parcels_utah.shape_area,
        parcels_utah.ave_sqft_per_unit,
        parcels_utah.total_job_spaces,
        feasibility,
        year=year,
        form_to_btype_callback=form_to_btype_func,
        add_more_columns_callback=add_extra_columns_func,
        residential=False,
        **kwargs)

    summary.add_parcel_output(new_buildings)
    
@sim.step('retail_developer_davis')
def retail_developer_davis(feasibility, jobs_davis_ret, buildings_davis_ret, buildings, parcels_davis, year,
                              settings, summary, form_to_btype_func,
                              add_extra_columns_func):

    kwargs = settings['retail_developer']
    new_buildings = wfrc_utils.run_developer(
        ["retail"],
        jobs_davis_ret,
        buildings_davis_ret,
        buildings,
        "job_spaces",
        parcels_davis.shape_area,
        parcels_davis.ave_sqft_per_unit,
        parcels_davis.total_job_spaces,
        feasibility,
        year=year,
        form_to_btype_callback=form_to_btype_func,
        add_more_columns_callback=add_extra_columns_func,
        residential=False,
        **kwargs)

    summary.add_parcel_output(new_buildings)
    
@sim.step('retail_developer_weber')
def retail_developer_weber(feasibility, jobs_weber_ret, buildings_weber_ret, buildings, parcels_weber, year,
                              settings, summary, form_to_btype_func,
                              add_extra_columns_func):

    kwargs = settings['retail_developer']
    new_buildings = wfrc_utils.run_developer(
        ["retail"],
        jobs_weber_ret,
        buildings_weber_ret,
        buildings,
        "job_spaces",
        parcels_weber.shape_area,
        parcels_weber.ave_sqft_per_unit,
        parcels_weber.total_job_spaces,
        feasibility,
        year=year,
        form_to_btype_callback=form_to_btype_func,
        add_more_columns_callback=add_extra_columns_func,
        residential=False,
        **kwargs)

    summary.add_parcel_output(new_buildings)
    
@sim.step('office_developer_slc')
def office_developer_slc(feasibility, jobs_slc_ofc, buildings_slc_ofc, buildings, parcels_slc, year,
                              settings, summary, form_to_btype_func,
                              add_extra_columns_func):

    kwargs = settings['office_developer']
    new_buildings = wfrc_utils.run_developer(
        ["office"],
        jobs_slc_ofc,
        buildings_slc_ofc,
        buildings,
        "job_spaces",
        parcels_slc.shape_area,
        parcels_slc.ave_sqft_per_unit,
        parcels_slc.total_job_spaces,
        feasibility,
        year=year,
        form_to_btype_callback=form_to_btype_func,
        add_more_columns_callback=add_extra_columns_func,
        residential=False,
        **kwargs)

    summary.add_parcel_output(new_buildings)
    
@sim.step('office_developer_utah')
def office_developer_utah(feasibility, jobs_utah_ofc, buildings_utah_ofc, buildings, parcels_utah, year,
                              settings, summary, form_to_btype_func,
                              add_extra_columns_func):

    kwargs = settings['office_developer']
    new_buildings = wfrc_utils.run_developer(
        ["office"],
        jobs_utah_ofc,
        buildings_utah_ofc,
        buildings,
        "job_spaces",
        parcels_utah.shape_area,
        parcels_utah.ave_sqft_per_unit,
        parcels_utah.total_job_spaces,
        feasibility,
        year=year,
        form_to_btype_callback=form_to_btype_func,
        add_more_columns_callback=add_extra_columns_func,
        residential=False,
        **kwargs)

    summary.add_parcel_output(new_buildings)
    
@sim.step('office_developer_davis')
def office_developer_davis(feasibility, jobs_davis_ofc, buildings_davis_ofc, buildings, parcels_davis, year,
                              settings, summary, form_to_btype_func,
                              add_extra_columns_func):

    kwargs = settings['office_developer']
    new_buildings = wfrc_utils.run_developer(
        ["office"],
        jobs_davis_ofc,
        buildings_davis_ofc,
        buildings,
        "job_spaces",
        parcels_davis.shape_area,
        parcels_davis.ave_sqft_per_unit,
        parcels_davis.total_job_spaces,
        feasibility,
        year=year,
        form_to_btype_callback=form_to_btype_func,
        add_more_columns_callback=add_extra_columns_func,
        residential=False,
        **kwargs)

    summary.add_parcel_output(new_buildings)
    
@sim.step('office_developer_weber')
def office_developer_weber(feasibility, jobs_weber_ofc, buildings_weber_ofc, buildings, parcels_weber, year,
                              settings, summary, form_to_btype_func,
                              add_extra_columns_func):

    kwargs = settings['office_developer']
    new_buildings = wfrc_utils.run_developer(
        ["office"],
        jobs_weber_ofc,
        buildings_weber_ofc,
        buildings,
        "job_spaces",
        parcels_weber.shape_area,
        parcels_weber.ave_sqft_per_unit,
        parcels_weber.total_job_spaces,
        feasibility,
        year=year,
        form_to_btype_callback=form_to_btype_func,
        add_more_columns_callback=add_extra_columns_func,
        residential=False,
        **kwargs)

    summary.add_parcel_output(new_buildings)
    
@sim.step('non_residential_developer_utah')
def non_residential_developer(feasibility, jobs_utah, buildings_utah, buildings, parcels_utah, year,
                              settings, summary, form_to_btype_func,
                              add_extra_columns_func):

    kwargs = settings['non_residential_developer']
    new_buildings = wfrc_utils.run_developer(
        ["office", "retail", "industrial"],
        jobs_utah,
        buildings_utah,
        buildings,
        "job_spaces",
        parcels_utah.shape_area,
        parcels_utah.ave_sqft_per_unit,
        parcels_utah.total_job_spaces,
        feasibility,
        year=year,
        form_to_btype_callback=form_to_btype_func,
        add_more_columns_callback=add_extra_columns_func,
        residential=False,
        **kwargs)

    summary.add_parcel_output(new_buildings) 
    
@sim.step('non_residential_developer_davis')
def non_residential_developer(feasibility, jobs_davis, buildings_davis, buildings, parcels_davis, year,
                              settings, summary, form_to_btype_func,
                              add_extra_columns_func):

    kwargs = settings['non_residential_developer']
    new_buildings = wfrc_utils.run_developer(
        ["office", "retail", "industrial"],
        jobs_davis,
        buildings_davis,
        buildings,
        "job_spaces",
        parcels_davis.shape_area,
        parcels_davis.ave_sqft_per_unit,
        parcels_davis.total_job_spaces,
        feasibility,
        year=year,
        form_to_btype_callback=form_to_btype_func,
        add_more_columns_callback=add_extra_columns_func,
        residential=False,
        **kwargs)

    summary.add_parcel_output(new_buildings)
    
@sim.step('non_residential_developer_weber')
def non_residential_developer(feasibility, jobs_weber, buildings_weber, buildings, parcels_weber, year,
                              settings, summary, form_to_btype_func,
                              add_extra_columns_func):

    kwargs = settings['non_residential_developer']
    new_buildings = wfrc_utils.run_developer(
        ["office", "retail", "industrial"],
        jobs_weber,
        buildings_weber,
        buildings,
        "job_spaces",
        parcels_weber.shape_area,
        parcels_weber.ave_sqft_per_unit,
        parcels_weber.total_job_spaces,
        feasibility,
        year=year,
        form_to_btype_callback=form_to_btype_func,
        add_more_columns_callback=add_extra_columns_func,
        residential=False,
        **kwargs)

    summary.add_parcel_output(new_buildings)
    
@sim.step("indicator_export")
def indicator_export(households, buildings, jobs, parcels, zones, distlrg, distmed, distsml, year, summary, run_number, settings):

#    if year > 2011:
#        f = sim.get_table('feasibility')
#        f = f['residential']
#        f.to_csv("runs/feasibilityyear" + str(year) + ".csv")
#        b = buildings.to_frame()
#        b.to_csv("runs/buildingyear" + str(year) + ".csv")
        

    #if year >= 2011:

    # if year in [2015,2020,2030,2040,2050]:
    if year in settings['remm']['indicator_years']:

        households = households.to_frame()
        buildings = buildings.to_frame()
        b1 = buildings[buildings.building_type_id==1]
        b2 = buildings[buildings.building_type_id==2]
        b3 = buildings[buildings.building_type_id==3]
        b4 = buildings[buildings.building_type_id==4]
        b5 = buildings[buildings.building_type_id==5]
        b1['improvement_value'] = np.multiply(b1.res_price_per_sqft, b1.building_sqft)
        b2['improvement_value'] = np.multiply(b2.res_price_per_sqft, b2.building_sqft)
        b3['improvement_value'] = np.multiply(b3.unit_price_non_residential, b3.non_residential_sqft)
        b4['improvement_value'] = np.multiply(b4.unit_price_non_residential, b4.non_residential_sqft)
        b5['improvement_value'] = np.multiply(b5.unit_price_non_residential, b5.non_residential_sqft)
        jobs = jobs.to_frame()
        #parcels = parcels.to_frame(["county_id","zone_id"])
        zones = zones.to_frame()
        distlrg = distlrg.to_frame()
        distmed = distmed.to_frame()
        distsml = distsml.to_frame()
        uphh = households[households.building_id==-1]
        upj = jobs[jobs.building_id==-1]
        counties = buildings.groupby('county_id').agg({'residential_units':'sum','job_spaces':'sum','non_residential_sqft':'sum'})
        bnew = buildings[buildings.note == 'simulated']
        #zones.to_csv("E:/REMMRun/run" + str(summary.run_num) + "year" + str(year) + "zonalvariables.csv")
        zones.to_csv("REMMRun/run" + str(summary.run_num) + "year" + str(year) + "zonalvariables.csv")
        
        

        #bnew.to_csv("E:/REMMRun/run" + str(summary.run_num) + "year" + str(year) + "newbuildings.csv")
        bnew.to_csv("REMMRun/run" + str(summary.run_num) + "year" + str(year) + "newbuildings.csv")
        buildings['households'] = households.groupby("building_id").building_id.count()

        buildings['population'] = households.groupby("building_id").persons.sum()
        buildings['jobs'] = jobs.groupby("building_id").building_id.count()
        buildings['jobs1'] = jobs[jobs.sector_id==1].groupby("building_id").building_id.count()
        buildings['jobs2'] = jobs[jobs.sector_id==2].groupby("building_id").building_id.count()
        buildings['jobs3'] = jobs[jobs.sector_id==3].groupby("building_id").building_id.count()
        buildings['jobs4'] = jobs[jobs.sector_id==4].groupby("building_id").building_id.count()
        buildings['jobs5'] = jobs[jobs.sector_id==5].groupby("building_id").building_id.count()
        buildings['jobs6'] = jobs[jobs.sector_id==6].groupby("building_id").building_id.count()
        buildings['jobs7'] = jobs[jobs.sector_id==7].groupby("building_id").building_id.count()
        buildings['jobs8'] = jobs[jobs.sector_id==8].groupby("building_id").building_id.count()
        buildings['jobs9'] = jobs[jobs.sector_id==9].groupby("building_id").building_id.count()
        buildings['jobs10'] = jobs[jobs.sector_id==10].groupby("building_id").building_id.count()
        buildings = buildings.fillna(0)
        #buildings.to_csv("REMMRun/run" + str(summary.run_num) + "year" + str(year) + "allbuildings.csv")
        buildings.to_csv("REMMRun/run" + str(summary.run_num) + "year" + str(year) + "allbuildings.csv")
            
        
        def geog_export(summary_geog, households, jobs, buildings, uphh, upj, year, run_number):
            id_col = summary_geog.index.name
            
            summary_geog['households'] = households.groupby(id_col).building_id.count()
            summary_geog['population'] = households.groupby(id_col).persons.sum()
            summary_geog['avg_hh_income'] = households.groupby(id_col).income.median()
            summary_geog['hh_inc1'] = households[households.income_quartile==1].groupby(id_col).building_id.count()
            summary_geog['hh_inc2'] = households[households.income_quartile==2].groupby(id_col).building_id.count()
            summary_geog['hh_inc3'] = households[households.income_quartile==3].groupby(id_col).building_id.count()
            summary_geog['hh_inc4'] = households[households.income_quartile==4].groupby(id_col).building_id.count()
        
            summary_geog['jobs'] = jobs.groupby(id_col).building_id.count()
            summary_geog['jobs1'] = jobs[jobs.sector_id==1].groupby(id_col).building_id.count()
            #summary_geog['jobs1_prop'][summary_geog.county_id==1] = np.divide(summary_geog.jobs1, jobs[(jobs.sector_id==1)&(jobs.county_id==1)].sector_id.count())
            summary_geog['jobs2'] = jobs[jobs.sector_id==2].groupby(id_col).building_id.count()
            summary_geog['jobs3'] = jobs[jobs.sector_id==3].groupby(id_col).building_id.count()
            summary_geog['jobs4'] = jobs[jobs.sector_id==4].groupby(id_col).building_id.count()
            summary_geog['jobs5'] = jobs[jobs.sector_id==5].groupby(id_col).building_id.count()
            summary_geog['jobs6'] = jobs[jobs.sector_id==6].groupby(id_col).building_id.count()
            summary_geog['jobs7'] = jobs[jobs.sector_id==7].groupby(id_col).building_id.count()
            summary_geog['jobs8'] = jobs[jobs.sector_id==8].groupby(id_col).building_id.count()
            summary_geog['jobs9'] = jobs[jobs.sector_id==9].groupby(id_col).building_id.count()
            summary_geog['jobs10'] = jobs[jobs.sector_id==10].groupby(id_col).building_id.count()
        
            summary_geog['residential_units'] = buildings.groupby(id_col).residential_units.sum()
            summary_geog['res_price1'] = np.divide(b1[(b1.res_price_per_sqft>0)&(b1.res_price_per_sqft<300)].groupby(id_col).improvement_value.sum(),b1[(b1.res_price_per_sqft>0)&(b1.res_price_per_sqft<300)].groupby(id_col).building_sqft.sum())
            summary_geog['res_price2'] = np.divide(b2[(b2.res_price_per_sqft>0)&(b2.res_price_per_sqft<300)].groupby(id_col).improvement_value.sum(),b2[(b2.res_price_per_sqft>0)&(b2.res_price_per_sqft<300)].groupby(id_col).building_sqft.sum())
            summary_geog['nonres_price3'] = np.divide(b3[(b3.unit_price_non_residential>0)&(b3.unit_price_non_residential <300)].groupby(id_col).improvement_value.sum(),b3[(b3.unit_price_non_residential>0)&(b3.unit_price_non_residential<300)].groupby(id_col).non_residential_sqft.sum())
            summary_geog['nonres_price4'] = np.divide(b4[(b4.unit_price_non_residential>0)&(b4.unit_price_non_residential <300)].groupby(id_col).improvement_value.sum(),b4[(b4.unit_price_non_residential>0)&(b4.unit_price_non_residential<300)].groupby(id_col).non_residential_sqft.sum())
            summary_geog['nonres_price5'] = np.divide(b5[(b5.unit_price_non_residential>0)&(b5.unit_price_non_residential <300)].groupby(id_col).improvement_value.sum(),b5[(b5.unit_price_non_residential>0)&(b5.unit_price_non_residential<300)].groupby(id_col).non_residential_sqft.sum())
            summary_geog['job_spaces'] = buildings.groupby(id_col).job_spaces.sum()
            summary_geog['non_residential_sqft'] = buildings.groupby(id_col).non_residential_sqft.sum()
            summary_geog[id_col] = summary_geog.index
            summary_geog = summary_geog[[id_col,'households','residential_units','population','res_price1','res_price2','avg_hh_income','hh_inc1','hh_inc2','hh_inc3','hh_inc4','jobs','jobs1','jobs2','jobs3','jobs4','jobs5','jobs6','jobs7','jobs8','jobs9','jobs10','job_spaces','non_residential_sqft','nonres_price3','nonres_price4','nonres_price5']]
            summary_geog = summary_geog.fillna(0)
            if id_col == 'zone_id':
                #ADJUSTPOPULATION
                eq = pd.read_csv("data/TAZCTYEQ.csv", index_col="Z")
                summary_geog['COUNTY'] = eq.COUNTY.reindex(summary_geog.index).fillna(0)
                pop_control = pd.read_csv("./data/population_controls.csv")
                pop_control = pop_control[pop_control.year == year]
                summary_geog['pop_adjust'] = 0
                summary_geog.pop_adjust[(summary_geog.households > 0) & (summary_geog.population/summary_geog.households > 1.5)]= 1
                zadjust = summary_geog[summary_geog.pop_adjust == 1]
                znoadjust = summary_geog[summary_geog.pop_adjust == 0]
                cadjust = zadjust.groupby("COUNTY").population.sum()
                cnoadjust = znoadjust.groupby("COUNTY").population.sum()
                adjust1 = (pop_control[pop_control.cid == 3].number_of_population.iloc[0] - cnoadjust[1])/cadjust[1]
                adjust2 = (pop_control[pop_control.cid == 1].number_of_population.iloc[0] - cnoadjust[2])/cadjust[2]
                adjust3 = (pop_control[pop_control.cid == 2].number_of_population.iloc[0] - cnoadjust[3])/cadjust[3]
                adjust4 = (pop_control[pop_control.cid == 4].number_of_population.iloc[0] - cnoadjust[4])/cadjust[4]
                zafteradjust = summary_geog.copy()
                zafteradjust.population[(zafteradjust.pop_adjust == 1) & (zafteradjust.COUNTY == 1)] =     zafteradjust.population[(zafteradjust.pop_adjust == 1) & (zafteradjust.COUNTY == 1)]*adjust1
                zafteradjust.population[(zafteradjust.pop_adjust == 1) & (zafteradjust.COUNTY == 2)] = zafteradjust.population[(zafteradjust.pop_adjust == 1) & (zafteradjust.COUNTY == 2)]*adjust2
                zafteradjust.population[(zafteradjust.pop_adjust == 1) & (zafteradjust.COUNTY == 3)] = zafteradjust.population[(zafteradjust.pop_adjust == 1) & (zafteradjust.COUNTY == 3)]*adjust3
                zafteradjust.population[(zafteradjust.pop_adjust == 1) & (zafteradjust.COUNTY == 4)] = zafteradjust.population[(zafteradjust.pop_adjust == 1) & (zafteradjust.COUNTY == 4)]*adjust4
                summary_geog = zafteradjust
                #ENDADJUSTPOPULATION  

            file_name = "%s_indicators_%d_%d.csv" % (id_col[:-3], run_number, year)
            #efile = os.path.join(misc.runs_dir(), file_name)
            # efile = os.path.join("E:/REMMRun", file_name)
            efile = os.path.join("REMMRun", file_name)
            summary_geog.to_csv(efile, index=False)

        geog_export(zones, households, jobs, buildings, uphh, upj, year, summary.run_num)
        #zones['TAZID'] = zones.index
        #summary.add_zone_output(zones, "indicator_export", year)
        #summary.write_zone_output()
        #geog_export(distlrg, households, jobs, buildings, uphh, upj, year, summary.run_num)
        #geog_export(distmed, households, jobs, buildings, uphh, upj, year, summary.run_num)
        #geog_export(distsml, households, jobs, buildings, uphh, upj, year, summary.run_num)
        geog_export(counties, households, jobs, buildings, uphh, upj, year, summary.run_num)
        #buildings.to_csv(r"data/buildings11202015.csv")
        #geog_export(zone_id_2011, households, jobs, buildings, uphh, upj, year, summary.run_num)        

@sim.step('travel_model_export_no_construction')
def travel_model_export_no_constuction(year, settings, jobs, households, buildings, parcels):
    households = households.to_frame()
    jobs = jobs.to_frame()
    tdm_output = pd.read_csv("data/tdm_template.csv",index_col = ";TAZID")
    tdm_output['TOTHH'] = households.groupby("zone_id").building_id.count()
    tdm_output['HHPOP'] = households.groupby("zone_id").persons.sum()
    tdm_output['RETL'] = jobs[jobs.sector_id==9].groupby("zone_id").building_id.count()
    tdm_output['FOOD'] = jobs[jobs.sector_id==1].groupby("zone_id").building_id.count()
    tdm_output['MANU'] = jobs[jobs.sector_id==5].groupby("zone_id").building_id.count()
    tdm_output['WSLE'] = jobs[jobs.sector_id==10].groupby("zone_id").building_id.count()
    tdm_output['OFFI'] = jobs[jobs.sector_id==6].groupby("zone_id").building_id.count()
    tdm_output['GVED'] = jobs[jobs.sector_id==3].groupby("zone_id").building_id.count()
    tdm_output['HLTH'] = jobs[jobs.sector_id==4].groupby("zone_id").building_id.count()
    tdm_output['OTHR'] = jobs[jobs.sector_id==7].groupby("zone_id").building_id.count()
    tdm_output = tdm_output.fillna(0)
    pop_control = pd.read_csv("data/population_controls.csv")
    pop_control = pop_control[pop_control.year == year]
    tdm_output['pop_adjust'] = 0
    tdm_output.pop_adjust[(tdm_output.TOTHH > 0) & (tdm_output.HHPOP/tdm_output.TOTHH > 1.5)]= 1
    zadjust = tdm_output[tdm_output.pop_adjust == 1]
    znoadjust = tdm_output[tdm_output.pop_adjust == 0]
    cadjust = zadjust.groupby("CO_FIPS").HHPOP.sum()
    cnoadjust = znoadjust.groupby("CO_FIPS").HHPOP.sum()
    adjust57 = (pop_control[pop_control.cid == 3].number_of_population.iloc[0] - cnoadjust[57])*1.0/cadjust[57]
    adjust11 = (pop_control[pop_control.cid == 1].number_of_population.iloc[0] - cnoadjust[11])*1.0/cadjust[11]
    adjust35 = (pop_control[pop_control.cid == 2].number_of_population.iloc[0] - cnoadjust[35])*1.0/cadjust[35]
    adjust49 = (pop_control[pop_control.cid == 4].number_of_population.iloc[0] - cnoadjust[49])*1.0/cadjust[49]            
    zafteradjust = tdm_output.copy()
    zafteradjust.HHPOP[(zafteradjust.pop_adjust == 1) & (zafteradjust.CO_FIPS == 57)] = zafteradjust.HHPOP[(zafteradjust.pop_adjust == 1) & (zafteradjust.CO_FIPS == 57)]*adjust57
    zafteradjust.HHPOP[(zafteradjust.pop_adjust == 1) & (zafteradjust.CO_FIPS == 11)] = zafteradjust.HHPOP[(zafteradjust.pop_adjust == 1) & (zafteradjust.CO_FIPS == 11)]*adjust11
    zafteradjust.HHPOP[(zafteradjust.pop_adjust == 1) & (zafteradjust.CO_FIPS == 35)] = zafteradjust.HHPOP[(zafteradjust.pop_adjust == 1) & (zafteradjust.CO_FIPS == 35)]*adjust35
    zafteradjust.HHPOP[(zafteradjust.pop_adjust == 1) & (zafteradjust.CO_FIPS == 49)] = zafteradjust.HHPOP[(zafteradjust.pop_adjust == 1) & (zafteradjust.CO_FIPS == 49)]*adjust49
    tdm_output = zafteradjust.copy()
    
    employment_control = pd.read_csv("data/employment_controls.csv")
    #Home-based Job
    hbj = employment_control[(employment_control.year == year)&(employment_control.sector_id == 12)]
    zhbj = tdm_output[(tdm_output.TOTHH > 0)]
    c_hbj_adjust = zhbj.groupby("CO_FIPS").TOTHH.sum()
    #first adjustment
    hbj_adjust57 =  (hbj[hbj.cid == 3].number_of_jobs.iloc[0])*1.0/c_hbj_adjust[57]
    hbj_adjust11 =  (hbj[hbj.cid == 1].number_of_jobs.iloc[0])*1.0/c_hbj_adjust[11]
    hbj_adjust35 =  (hbj[hbj.cid == 2].number_of_jobs.iloc[0])*1.0/c_hbj_adjust[35]
    hbj_adjust49 =  (hbj[hbj.cid == 4].number_of_jobs.iloc[0])*1.0/c_hbj_adjust[49]
    tdm_output["HBJ"] = 0
    tdm_output.HBJ[tdm_output.CO_FIPS == 57] = tdm_output.TOTHH[tdm_output.CO_FIPS == 57]*hbj_adjust57
    tdm_output.HBJ[tdm_output.CO_FIPS == 11] = tdm_output.TOTHH[tdm_output.CO_FIPS == 11]*hbj_adjust11
    tdm_output.HBJ[tdm_output.CO_FIPS == 35] = tdm_output.TOTHH[tdm_output.CO_FIPS == 35]*hbj_adjust35
    tdm_output.HBJ[tdm_output.CO_FIPS == 49] = tdm_output.TOTHH[tdm_output.CO_FIPS == 49]*hbj_adjust49
    tdm_output.HBJ = np.round(tdm_output.HBJ)
    #second adjustment
    c_hbj_adjust = tdm_output.groupby("CO_FIPS").HBJ.sum()
    hbj_adjust57 =  (hbj[hbj.cid == 3].number_of_jobs.iloc[0])*1.0/c_hbj_adjust[57]
    hbj_adjust11 =  (hbj[hbj.cid == 1].number_of_jobs.iloc[0])*1.0/c_hbj_adjust[11]
    hbj_adjust35 =  (hbj[hbj.cid == 2].number_of_jobs.iloc[0])*1.0/c_hbj_adjust[35]
    hbj_adjust49 =  (hbj[hbj.cid == 4].number_of_jobs.iloc[0])*1.0/c_hbj_adjust[49]
    tdm_output.HBJ[tdm_output.CO_FIPS == 57] = tdm_output.HBJ[tdm_output.CO_FIPS == 57]*hbj_adjust57
    tdm_output.HBJ[tdm_output.CO_FIPS == 11] = tdm_output.HBJ[tdm_output.CO_FIPS == 11]*hbj_adjust11
    tdm_output.HBJ[tdm_output.CO_FIPS == 35] = tdm_output.HBJ[tdm_output.CO_FIPS == 35]*hbj_adjust35
    tdm_output.HBJ[tdm_output.CO_FIPS == 49] = tdm_output.HBJ[tdm_output.CO_FIPS == 49]*hbj_adjust49

    #Agriculture Job
    agj = employment_control[(employment_control.year == year)&(employment_control.sector_id == 11)]
    p = parcels.to_frame(['total_residential_units','total_job_spaces','zone_id','agriculture','shape_area'])
    pa = p[(p.agriculture == 1) & (p.total_residential_units == 0) & (p.total_residential_units == 0) & (p.zone_id != 2870)]
    tdm_output['agr_sqft'] = pa.groupby("zone_id").shape_area.sum()
    tdm_output = tdm_output.fillna(0)
    zagj = tdm_output[(tdm_output.agr_sqft > 0)]
    c_agj_adjust = zagj.groupby("CO_FIPS").agr_sqft.sum()
    #first adjustment
    agj_adjust57 =  (agj[agj.cid == 3].number_of_jobs.iloc[0])*1.0/c_agj_adjust[57]
    agj_adjust11 =  (agj[agj.cid == 1].number_of_jobs.iloc[0])*1.0/c_agj_adjust[11]
    agj_adjust35 =  (agj[agj.cid == 2].number_of_jobs.iloc[0])*1.0/c_agj_adjust[35]
    agj_adjust49 =  (agj[agj.cid == 4].number_of_jobs.iloc[0])*1.0/c_agj_adjust[49]
    tdm_output['FM_AGRI'] = 0
    tdm_output.FM_AGRI[tdm_output.CO_FIPS == 57] = tdm_output.agr_sqft[tdm_output.CO_FIPS == 57]*agj_adjust57
    tdm_output.FM_AGRI[tdm_output.CO_FIPS == 11] = tdm_output.agr_sqft[tdm_output.CO_FIPS == 11]*agj_adjust11
    tdm_output.FM_AGRI[tdm_output.CO_FIPS == 35] = tdm_output.agr_sqft[tdm_output.CO_FIPS == 35]*agj_adjust35
    tdm_output.FM_AGRI[tdm_output.CO_FIPS == 49] = tdm_output.agr_sqft[tdm_output.CO_FIPS == 49]*agj_adjust49
    tdm_output.FM_AGRI = np.round(tdm_output.FM_AGRI)
    #secondadjustment
    c_agj_adjust = tdm_output.groupby("CO_FIPS").FM_AGRI.sum()
    agj_adjust57 =  (agj[agj.cid == 3].number_of_jobs.iloc[0])*1.0/c_agj_adjust[57]
    agj_adjust11 =  (agj[agj.cid == 1].number_of_jobs.iloc[0])*1.0/c_agj_adjust[11]
    agj_adjust35 =  (agj[agj.cid == 2].number_of_jobs.iloc[0])*1.0/c_agj_adjust[35]
    agj_adjust49 =  (agj[agj.cid == 4].number_of_jobs.iloc[0])*1.0/c_agj_adjust[49]  
    tdm_output.FM_AGRI[tdm_output.CO_FIPS == 57] = tdm_output.FM_AGRI[tdm_output.CO_FIPS == 57]*agj_adjust57
    tdm_output.FM_AGRI[tdm_output.CO_FIPS == 11] = tdm_output.FM_AGRI[tdm_output.CO_FIPS == 11]*agj_adjust11
    tdm_output.FM_AGRI[tdm_output.CO_FIPS == 35] = tdm_output.FM_AGRI[tdm_output.CO_FIPS == 35]*agj_adjust35
    tdm_output.FM_AGRI[tdm_output.CO_FIPS == 49] = tdm_output.FM_AGRI[tdm_output.CO_FIPS == 49]*agj_adjust49                
                
            
    #Mining Job
    mij = employment_control[(employment_control.year == year)&(employment_control.sector_id == 8)]
    c_mij_adjust = tdm_output.groupby("CO_FIPS").FM_MING.sum()
    mij_adjust57 =  (mij[mij.cid == 3].number_of_jobs.iloc[0])*1.0/c_mij_adjust[57]
    mij_adjust11 =  (mij[mij.cid == 1].number_of_jobs.iloc[0])*1.0/c_mij_adjust[11]
    mij_adjust35 =  (mij[mij.cid == 2].number_of_jobs.iloc[0])*1.0/c_mij_adjust[35]
    mij_adjust49 =  (mij[mij.cid == 4].number_of_jobs.iloc[0])*1.0/c_mij_adjust[49]
    tdm_output.FM_MING[tdm_output.CO_FIPS == 57] = tdm_output.FM_MING[tdm_output.CO_FIPS == 57]*mij_adjust57
    tdm_output.FM_MING[tdm_output.CO_FIPS == 11] = tdm_output.FM_MING[tdm_output.CO_FIPS == 11]*mij_adjust11
    tdm_output.FM_MING[tdm_output.CO_FIPS == 35] = tdm_output.FM_MING[tdm_output.CO_FIPS == 35]*mij_adjust35
    tdm_output.FM_MING[tdm_output.CO_FIPS == 49] = tdm_output.FM_MING[tdm_output.CO_FIPS == 49]*mij_adjust49
                
    #tdmoutput
    tdm_output['HHSIZE'] = 0
    tdm_output.HHSIZE[tdm_output.TOTHH > 0] = tdm_output.HHPOP[tdm_output.TOTHH > 0]/tdm_output.TOTHH[tdm_output.TOTHH > 0]    
        
    inputdir = settings['tdm']['input_dir'] # come back to this when gui is farther along
    filename = "SE_WF_" + str(year) + ".csv"
    filepath = os.path.join(inputdir, filename)
            
    tdm_output.to_csv(filepath)
    
    buildings = buildings.to_frame(['residential_units','job_spaces','zone_id'])
    unitfilename =  "UNITS_JOBSPACES_" + str(year) + ".csv"
    unitfilepath = os.path.join(inputdir, unitfilename)
    unit_output = tdm_output[['TOTHH','HHPOP']]
    unit_output['REMMEMP'] = jobs.groupby("zone_id").building_id.count()
    unit_output['residential_units'] = buildings.groupby("zone_id").residential_units.sum()
    unit_output['job_spaces'] = buildings.groupby("zone_id").job_spaces.sum()
    unit_output = unit_output.fillna(0)
    unit_output.to_csv(unitfilepath)
    
 
@sim.step('travel_model_export_add_construction')
def travel_model_export_add_constuction(year, settings, jobs, households, buildings, parcels):
    
    buildings = buildings.to_frame()
    inputdir = settings['tdm']['input_dir']
    filename = "SE_WF_" + str(year) + ".csv"
    filepath = os.path.join(inputdir, filename)
    
    tdm_output = pd.read_csv(filepath,index_col = ";TAZID")
                 
    employment_control = pd.read_csv("data/employment_controls.csv")
            
    #Construction Job
    coj = employment_control[(employment_control.year == year)&(employment_control.sector_id == 2)]
    cbuilding = buildings[buildings.year_built == year]
    tdm_output['new_building_sqft'] = cbuilding.groupby("zone_id").building_sqft.sum()
    tdm_output = tdm_output.fillna(0)
    zcoj = tdm_output[(tdm_output.new_building_sqft > 0)]
    c_coj_adjust = zcoj.groupby("CO_FIPS").new_building_sqft.sum()
    #firstadjustment

    if 57 in c_coj_adjust.index.tolist():
        coj_adjust57 = (coj[coj.cid == 3].number_of_jobs.iloc[0])*1.0/c_coj_adjust[57]
    else:
        coj_adjust57 = 0

    if 11 in c_coj_adjust.index.tolist():
        coj_adjust11 =  (coj[coj.cid == 1].number_of_jobs.iloc[0])*1.0/c_coj_adjust[11]
    else:
        coj_adjust11 = 0

    if 35 in c_coj_adjust.index.tolist():
        coj_adjust35 =  (coj[coj.cid == 2].number_of_jobs.iloc[0])*1.0/c_coj_adjust[35]
    else:
        coj_adjust35 = 0

    if 49 in c_coj_adjust.index.tolist():
        coj_adjust49 =  (coj[coj.cid == 4].number_of_jobs.iloc[0])*1.0/c_coj_adjust[49]
    else:
        coj_adjust49 = 0

    tdm_output['FM_CONS'] = 0
    tdm_output.FM_CONS[tdm_output.CO_FIPS == 57] = tdm_output.new_building_sqft[tdm_output.CO_FIPS == 57]*coj_adjust57
    tdm_output.FM_CONS[tdm_output.CO_FIPS == 11] = tdm_output.new_building_sqft[tdm_output.CO_FIPS == 11]*coj_adjust11
    tdm_output.FM_CONS[tdm_output.CO_FIPS == 35] = tdm_output.new_building_sqft[tdm_output.CO_FIPS == 35]*coj_adjust35
    tdm_output.FM_CONS[tdm_output.CO_FIPS == 49] = tdm_output.new_building_sqft[tdm_output.CO_FIPS == 49]*coj_adjust49
    tdm_output.FM_CONS = np.round(tdm_output.FM_CONS)
    #secondadjustment
    c_coj_adjust = tdm_output.groupby("CO_FIPS").FM_CONS.sum()
    coj_adjust57 =  (coj[coj.cid == 3].number_of_jobs.iloc[0])*1.0/c_coj_adjust[57]
    coj_adjust11 =  (coj[coj.cid == 1].number_of_jobs.iloc[0])*1.0/c_coj_adjust[11]
    coj_adjust35 =  (coj[coj.cid == 2].number_of_jobs.iloc[0])*1.0/c_coj_adjust[35]
    coj_adjust49 =  (coj[coj.cid == 4].number_of_jobs.iloc[0])*1.0/c_coj_adjust[49]
    tdm_output.FM_CONS[tdm_output.CO_FIPS == 57] = tdm_output.FM_CONS[tdm_output.CO_FIPS == 57]*coj_adjust57
    tdm_output.FM_CONS[tdm_output.CO_FIPS == 11] = tdm_output.FM_CONS[tdm_output.CO_FIPS == 11]*coj_adjust11
    tdm_output.FM_CONS[tdm_output.CO_FIPS == 35] = tdm_output.FM_CONS[tdm_output.CO_FIPS == 35]*coj_adjust35
    tdm_output.FM_CONS[tdm_output.CO_FIPS == 49] = tdm_output.FM_CONS[tdm_output.CO_FIPS == 49]*coj_adjust49        

    tdm_output['ALLEMP'] = tdm_output['RETL']+tdm_output['FOOD']+tdm_output['MANU']+tdm_output['WSLE']+tdm_output['OFFI']+tdm_output['GVED']+tdm_output['HLTH']+tdm_output['OTHR']+tdm_output['FM_AGRI']+tdm_output['FM_MING']+tdm_output['FM_CONS']+tdm_output['HBJ']
    
    tdm_output['RETEMP'] = tdm_output['RETL'] + tdm_output['FOOD']
    tdm_output['INDEMP'] = tdm_output['MANU'] + tdm_output['WSLE']
    tdm_output['OTHEMP'] = tdm_output['OFFI'] + tdm_output['GVED']+ tdm_output['HLTH'] + tdm_output['OTHR']
    tdm_output['TOTEMP'] = tdm_output['RETEMP'] + tdm_output['INDEMP']+ tdm_output['OTHEMP']
    
    
    tdm_output = tdm_output[["CO_TAZID","TOTHH","HHPOP","HHSIZE","TOTEMP","RETEMP","INDEMP","OTHEMP","ALLEMP","RETL","FOOD","MANU","WSLE","OFFI","GVED","HLTH","OTHR","FM_AGRI","FM_MING","FM_CONS","HBJ","AVGINCOME","Enrol_Elem","Enrol_Midl","Enrol_High","CO_FIPS","CO_NAME"]]
    
    inputdir = settings['tdm']['input_dir']
    filename = "SE_WF_" + str(year) + ".csv"
    filepath = os.path.join(inputdir, filename)
            
    tdm_output.to_csv(filepath)    
    
    

@sim.step('travel_time_import')
def travel_time_import(settings, year):
    if year in settings['tdm']['run_years']:
        mdir = settings['tdm']['main_dir'] + str(year)
        ttfile = settings['tdm']['output_traveltime']
        logsumfile = settings['tdm']['output_logsum']
        ttfile_path = os.path.join(mdir, ttfile)
        logsumfile_path = os.path.join(mdir, logsumfile)
        ttdbf = ps.lib.io.open(ttfile_path)
        ttd = {col: ttdbf.by_col(col) for col in ["I", "J", "MINAUTO","MINTRANSIT"]}
        td_new =  pd.DataFrame(ttd)
        logsumdbf = ps.lib.io.open(logsumfile_path)
        logsumd = {col: logsumdbf.by_col(col) for col in ["I", "J", "LOG0","LOG1","LOG2"]}
        logsum = pd.DataFrame(logsumd)
        td_new = td_new.rename(columns={'I':'from_zone_id','J':'to_zone_id','MINAUTO':'travel_time','MINTRANSIT':'travel_time_transit'})
        td_new.from_zone_id = td_new.from_zone_id.astype('int')
        td_new.to_zone_id = td_new.to_zone_id.astype('int')
        logsum = logsum.rename(columns={'I':'from_zone_id','J':'to_zone_id','LOG0':'log0','LOG1':'log1','LOG2':'log2'})
        logsum.from_zone_id = logsum.from_zone_id.astype('int')
        logsum.to_zone_id = logsum.to_zone_id.astype('int')
        td_new.set_index(['from_zone_id','to_zone_id'], inplace=True)
        logsum.set_index(['from_zone_id','to_zone_id'], inplace=True)
        td_all = pd.merge(td_new, logsum, left_index=True, right_index=True)
        
        sim.add_table("travel_data", td_all, cache=True)

    elif year == 2050:
        travel_time_reset(settings, year)


@sim.step('travel_time_reset')
def travel_time_reset(settings, year):
    mdir = settings['tdm']['main_dir'] + '2015'
    ttfile = settings['tdm']['output_traveltime']
    logsumfile = settings['tdm']['output_logsum']
    ttfile_path = os.path.join(mdir, ttfile)
    logsumfile_path = os.path.join(mdir, logsumfile)
    ttdbf = ps.lib.io.open(ttfile_path)
    ttd = {col: ttdbf.by_col(col) for col in ["I", "J", "MINAUTO","MINTRANSIT"]}
    td_new =  pd.DataFrame(ttd)
    logsumdbf = ps.lib.io.open(logsumfile_path)
    logsumd = {col: logsumdbf.by_col(col) for col in ["I", "J", "LOG0","LOG1","LOG2"]}
    logsum = pd.DataFrame(logsumd)
    td_new = td_new.rename(columns={'I':'from_zone_id','J':'to_zone_id','MINAUTO':'travel_time','MINTRANSIT':'travel_time_transit'})
    td_new.from_zone_id = td_new.from_zone_id.astype('int')
    td_new.to_zone_id = td_new.to_zone_id.astype('int')
    logsum = logsum.rename(columns={'I':'from_zone_id','J':'to_zone_id','LOG0':'log0','LOG1':'log1','LOG2':'log2'})
    logsum.from_zone_id = logsum.from_zone_id.astype('int')
    logsum.to_zone_id = logsum.to_zone_id.astype('int')
    td_new.set_index(['from_zone_id','to_zone_id'], inplace=True)
    logsum.set_index(['from_zone_id','to_zone_id'], inplace=True)
    td_all = pd.merge(td_new, logsum, left_index=True, right_index=True)
    sim.add_table("travel_data", td_all, cache=True)
    

        
@sim.step('run_cube')
def run_cube(year, settings,store):
    REMMdir = os.getcwd()
    if year in settings['tdm']['run_years']:
        mdir = settings['tdm']['main_dir'] + str(year)
        batdir = settings['tdm']['bat_dir']
        bat_file = "_Run_Hailmary_" + str(year) + ".bat"
        file_path = os.path.join(batdir, bat_file)
        os.chdir(batdir)
        if os.path.exists(batdir):
            subprocess.call(file_path)
            outdir = os.path.join(mdir,"6_REMM\Ro")
            if os.path.exists(outdir):
                os.chdir(outdir)
                subprocess.call("VolumeCalculate.bat")
            os.chdir(REMMdir)
            travel_time_import(settings, year)
        else:
            print("Batch file is missing")
        os.chdir(REMMdir)
        
        
@sim.step('run_arcpy') # update this to use built in arcpy module
def run_arcpy(year, settings,store):
    if year%3 == 2:
    #if year > 0:
        REMMdir = os.getcwd()
        b = sim.get_table('parcels').to_frame(['x','y','total_residential_units','total_job_spaces','county_id','gridID'])
        bdev = b[((b.total_residential_units >= 1) | (b.total_job_spaces >= 1))&(b.county_id == 4)]
        bdev.to_csv('utahdevelopedparcels.csv')
        f = open('YEAR.txt', 'w')
        f.write(str(year))
        f.close()
        os.chdir(os.path.join(REMMdir,"UtilityRestriction"))
        subprocess.call(r"UtilityRestriction.bat")
        os.chdir(REMMdir)
        
# this if the function for mapping a specific building that we build to a
# specific building type
@sim.injectable("form_to_btype_func", autocall=False)
def form_to_btype_func(building):
    settings = sim.settings
    form = building.form
    if form is None or form == "residential":
        zoningbaseline = sim.get_table("zoning_baseline")
        zbl = zoningbaseline.to_frame(["type1","type2"])
        buildingzoning = zbl.loc[int(building.parcel_id)]
        if buildingzoning.type1 == 't' and buildingzoning.type2 == 'f':
            return 1
        elif buildingzoning.type1 == 'f' and buildingzoning.type2 == 't':
            return 2
        else:
            dua = building.residential_units / (building.parcel_size / 43560.0)
            # precise mapping of form to building type for residential
            if dua <= 7:
                return 1
            elif dua > 7:
                return 2
    return settings["form_to_btype"][form][0]

    
@sim.injectable("add_extra_columns_func", autocall=False)
def add_extra_columns(df):
    for col in ["residential_price", "non_residential_price","improvement_value","year_built","unit_price_residential","unit_price_non_residential","shape_area","res_price_per_sqft","res_price_per_sqft_old"]:
        df[col] = 0
    return df


####Zonal Models


@sim.step('zone_res_estimate')
def zone_res_estimate(zones):
    zone_res = utils.hedonic_estimate("zonal_res_price.yaml", zones, [])
    
@sim.step('zone_ofc_estimate')
def zone_ofc_estimate(zones):
    zone_ofc = utils.hedonic_estimate("zonal_ofc_price.yaml", zones, [])
    
@sim.step('zone_ret_estimate')
def zone_ret_estimate(zones):
    zone_ret = utils.hedonic_estimate("zonal_ret_price.yaml", zones, [])
    
@sim.step('zone_ind_estimate')
def zone_ind_estimate(zones):
    zone_ind = utils.hedonic_estimate("zonal_ind_price.yaml", zones, [])
