
import datasources, models, variables, utils
import orca.orca as sim
import pandas as pd
import numpy as np
import os

#sim.run(["travel_time_reset"])
sim.run([
        "clear_cache",
        #"run_arcpy",
        "travel_time_import",
        "neighborhood_vars",      # neighborhood variables
        "households_transition",  # households transition
        "jobs_transition",        # jobs transition
        "feasibility",  # compute feasibility
        "residential_developer_slc",  # build buildings Salt Lake County
        "residential_developer_davis",  # build buildings Davis County
        "residential_developer_weber",  # build buildings Weber County
        "residential_developer_utah",  # build buildings Utah County
        "office_developer_slc",
        "office_developer_utah",
        "office_developer_davis",
        "office_developer_weber",
        "retail_developer_slc",
        "retail_developer_utah",
        "retail_developer_davis",
        "retail_developer_weber",
        "industrial_developer_slc",
        "industrial_developer_utah",
        "industrial_developer_davis",
        "industrial_developer_weber",
        "nrh_ind_simulate",  # industrial price model
        "nrh_ofc_simulate",  # office price model
        "nrh_ret_simulate",  # retail price model
        "nrh_mu_oth_simulate",  # non-residential mixed-use, other model
        "rsh_sf_simulate",  # single-family residential price model
        "rsh_mf_simulate",  # multi-family residential price model
        "hlcm_simulate_weber",    # households location choice Weber County
        "hlcm_simulate_slc",      # households location choice Salt Lake County
        "hlcm_simulate_utah",     # households location choice Utah County
        "hlcm_simulate_davis",    # households location choice Davis County
        "elcm_simulate_slc",      # employment location choice Salt Lake County
        "elcm_simulate_utah",     # employment location choice Utah County
        "elcm_simulate_davis",    # employment location choice Davis County
        "elcm_simulate_weber",  # employment location choice Weber County
        #"clear_cache",
        "indicator_export",
        "travel_model_export_no_construction",  # export travel model inputs at TAZ level in specified years
        "garbage_collect",
        "travel_model_export_add_construction",  # export travel model inputs at TAZ level in specified years
        #"run_cube",               # call Cube and import travel times in specified years
    ], iter_vars=range(2015,2061)) #this can be determined using indicator_years_from setting.yaml
