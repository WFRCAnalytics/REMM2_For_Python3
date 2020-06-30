import numpy as np
import pandas as pd
import pysal as ps
import os
import orca_wfrc.orca as sim
from urbansim_wfrc.utils import misc
from urbansim_wfrc.developer import sqftproforma, developer
from urbansim_wfrc.models import SegmentedMNLLocationChoiceModel
from urbansim_defaults import utils
#import WFRCDeveloper
import json


def get_run_no():
    if 'run_no' not in sim.list_injectables():
        sim.add_injectable("run_no", misc.get_run_number())
    return sim.get_injectable("run_no")

def get_run_filename():
    if 'run_no' not in sim.list_injectables():
        get_run_no()
    return os.path.join(misc.runs_dir(), "run%d.h5" % sim.get_injectable("run_no"))
    
def df2dbf(df, dbf_path):
    type2spec = {int: ('N', 20, 0),
                 np.int64: ('N', 20, 0),
                 np.int32: ('N', 20, 0),
                 float: ('N', 36, 15),
                 np.float32: ('N', 36, 15),
                 np.float64: ('N', 36, 15),
                 str: ('C', 14, 0)
                 }
    types = [type(df[i].iloc[0]) for i in df.columns]
    specs = [type2spec[t] for t in types]
    db = ps.open(dbf_path, 'w')
    db.header = list(df.columns)
    db.field_spec = specs
    for i, row in df.T.iteritems():
        db.write(row)
    db.close()


def dbf2df(dbf_path, index=None, cols=False, incl_index=False):
    db = ps.lib.io.open(dbf_path)
    if cols:
        if incl_index:
            cols.append(index)
        vars_to_read = cols
    else:
        vars_to_read = db.header
    data = dict([(var, db.by_col(var)) for var in vars_to_read])
    if index:
        index = db.by_col(index)
        db.close()
        return pd.DataFrame(data, index=index)
    else:
        db.close()
        return pd.DataFrame(data)
    
def lcm_simulate(cfg, choosers, buildings, join_tbls, out_fname,
                 supply_fname, vacant_fname,
                 enable_supply_correction=None):
    """
    Simulate the location choices for the specified choosers

    Parameters
    ----------
    cfg : string
        The name of the yaml config file from which to read the location
        choice model
    choosers : DataFrameWrapper
        A dataframe of agents doing the choosing
    buildings : DataFrameWrapper
        A dataframe of buildings which the choosers are locating in and which
        have a supply
    join_tbls : list of strings
        A list of land use dataframes to give neighborhood info around the
        buildings - will be joined to the buildings using existing broadcasts.
    out_fname : string
        The column name to write the simulated location to
    supply_fname : string
        The string in the buildings table that indicates the amount of
        available units there are for choosers, vacant or not
    vacant_fname : string
        The string in the buildings table that indicates the amount of vacant
        units there will be for choosers
    enable_supply_correction : Python dict
        Should contain keys "price_col" and "submarket_col" which are set to
        the column names in buildings which contain the column for prices and
        an identifier which segments buildings into submarkets
    """
    cfg = misc.config(cfg)

    choosers_df = utils.to_frame(choosers, [], cfg, additional_columns=[out_fname])
    
    additional_columns = [supply_fname, vacant_fname]
    if enable_supply_correction is not None and \
            "submarket_col" in enable_supply_correction:
        additional_columns += [enable_supply_correction["submarket_col"]]
    if enable_supply_correction is not None and \
            "price_col" in enable_supply_correction:
        additional_columns += [enable_supply_correction["price_col"]]
    locations_df = utils.to_frame(buildings, join_tbls, cfg,
                            additional_columns=additional_columns)
    
    
    available_units = buildings[supply_fname]
    vacant_units = buildings[vacant_fname]

    print("There are %d total available units" % available_units.sum())
    print("    and %d total choosers" % len(choosers))
    print("    but there are %d overfull buildings" %
          len(vacant_units[vacant_units < 0]))

    vacant_units = vacant_units[vacant_units > 0]

    # sometimes there are vacant units for buildings that are not in the
    # locations_df, which happens for reasons explained in the warning below
    indexes = np.repeat(vacant_units.index.values,
                        vacant_units.values.astype('int'))
    isin = pd.Series(indexes).isin(locations_df.index)
    missing = len(isin[isin == False])
    indexes = indexes[isin.values]
    units = locations_df.loc[indexes].reset_index()
    utils.check_nas(units)

    print("    for a total of %d temporarily empty units" % vacant_units.sum())
    print("    in %d buildings total in the region" % len(vacant_units))

    if missing > 0:
        print("WARNING: %d indexes aren't found in the locations df -" %
              missing)
        print("    this is usually because of a few records that don't join ")
        print("    correctly between the locations df and the aggregations tables")

    movers = choosers_df[choosers_df[out_fname] == -1]
    print("There are %d total movers for this LCM" % len(movers))

    if enable_supply_correction is not None:
        assert isinstance(enable_supply_correction, dict)
        assert "price_col" in enable_supply_correction
        price_col = enable_supply_correction["price_col"]
        assert "submarket_col" in enable_supply_correction
        submarket_col = enable_supply_correction["submarket_col"]

        lcm = utils.yaml_to_class(cfg).from_yaml(str_or_buffer=cfg)

        if enable_supply_correction.get("warm_start", False) is True:
            raise NotImplementedError()

        multiplier_func = enable_supply_correction.get("multiplier_func", None)
        if multiplier_func is not None:
            multiplier_func = sim.get_injectable(multiplier_func)

        kwargs = enable_supply_correction.get('kwargs', {})
        new_prices, submarkets_ratios = supply_and_demand(
            lcm,
            movers,
            units,
            submarket_col,
            price_col,
            base_multiplier=None,
            multiplier_func=multiplier_func,
            **kwargs)

        # we will only get back new prices for those alternatives
        # that pass the filter - might need to specify the table in
        # order to get the complete index of possible submarkets
        submarket_table = enable_supply_correction.get("submarket_table", None)
        if submarket_table is not None:
            submarkets_ratios = submarkets_ratios.reindex(
                sim.get_table(submarket_table).index).fillna(1)
            # write final shifters to the submarket_table for use in debugging
            sim.get_table(submarket_table)["price_shifters"] = submarkets_ratios

        print("Running supply and demand")
        print("Simulated Prices")
        print(buildings[price_col].describe())
        print("Submarket Price Shifters")
        print(submarkets_ratios.describe())
        # we want new prices on the buildings, not on the units, so apply
        # shifters directly to buildings and ignore unit prices
        sim.add_column(buildings.name,
                       price_col+"_hedonic", buildings[price_col])
        new_prices = buildings[price_col] * \
            submarkets_ratios.loc[buildings[submarket_col]].values
        buildings.update_col_from_series(price_col, new_prices)
        print("Adjusted Prices")
        print(buildings[price_col].describe())

    #if len(movers) > vacant_units.sum():
    #    print "WARNING: Not enough locations for movers"
    #    print "    reducing locations to size of movers for performance gain"
    #    movers = movers.head(vacant_units.sum())

    new_units, _ = utils.yaml_to_class(cfg).predict_from_cfg(movers, units, cfg)
    # new_units returns nans when there aren't enough units,
    # get rid of them and they'll stay as -1s
    new_units = new_units.dropna()

    # go from units back to buildings
    new_buildings = pd.Series(units.loc[new_units.values][out_fname].values,
                              index=new_units.index)

    choosers.update_col_from_series(out_fname, new_buildings)
    utils._print_number_unplaced(choosers, out_fname)

    if enable_supply_correction is not None:
        new_prices = buildings[price_col]
        if "clip_final_price_low" in enable_supply_correction:
            new_prices = new_prices.clip(lower=enable_supply_correction[
                "clip_final_price_low"])
        if "clip_final_price_high" in enable_supply_correction:
            new_prices = new_prices.clip(upper=enable_supply_correction[
                "clip_final_price_high"])
        buildings.update_col_from_series(price_col, new_prices)

    vacant_units = buildings[vacant_fname]
    print("    and there are now %d empty units" % vacant_units.sum())
    print("    and %d overfull buildings" % len(vacant_units[vacant_units < 0]))


def run_developer(forms, agents, buildings, buildings_all, supply_fname, parcel_size,
                  ave_unit_size, total_units, feasibility, year=None,
                  target_vacancy=.1, form_to_btype_callback=None,
                  add_more_columns_callback=None, max_parcel_size=34647265,
                  residential=True, bldg_sqft_per_job=400.0,
                  min_unit_size=400, remove_developed_buildings=True,
                  unplace_agents=['households', 'jobs']):
    """
    Run the developer model to pick and build buildings

    Parameters
    ----------
    forms : string or list of strings
        Passed directly dev.pick
    agents : DataFrame Wrapper
        Used to compute the current demand for units/floorspace in the area
    buildings : DataFrame Wrapper
        Used to compute the current supply of units/floorspace in the area
    buildings_all:
        Buildings for the entire region, used to write back to buildings table
    supply_fname : string
        Identifies the column in buildings which indicates the supply of
        units/floorspace
    parcel_size : Series
        Passed directly to dev.pick
    ave_unit_size : Series
        Passed directly to dev.pick - average residential unit size
    total_units : Series
        Passed directly to dev.pick - total current residential_units /
        job_spaces
    feasibility : DataFrame Wrapper
        The output from feasibility above (the table called 'feasibility')
    year : int
        The year of the simulation - will be assigned to 'year_built' on the
        new buildings
    target_vacancy : float
        The target vacancy rate - used to determine how much to build
    form_to_btype_callback : function
        Will be used to convert the 'forms' in the pro forma to
        'building_type_id' in the larger model
    add_more_columns_callback : function
        Takes a dataframe and returns a dataframe - is used to make custom
        modifications to the new buildings that get added
    max_parcel_size : float
        Passed directly to dev.pick - max parcel size to consider
    min_unit_size : float
        Passed directly to dev.pick - min unit size that is valid
    residential : boolean
        Passed directly to dev.pick - switches between adding/computing
        residential_units and job_spaces
    bldg_sqft_per_job : float
        Passed directly to dev.pick - specified the multiplier between
        floor spaces and job spaces for this form (does not vary by parcel
        as ave_unit_size does)
    remove_redeveloped_buildings : optional, boolean (default True)
        Remove all buildings on the parcels which are being developed on
    unplace_agents : optional : list of strings (default ['households', 'jobs'])
        For all tables in the list, will look for field building_id and set
        it to -1 for buildings which are removed - only executed if
        remove_developed_buildings is true

    Returns
    -------
    Writes the result back to the buildings table and returns the new
    buildings with available debugging information on each new building
    """

    dev = developer.Developer(feasibility.to_frame())
    #dev = WFRCDeveloper.WFRCDeveloper(feasibility.to_frame())
    target_units = dev.\
        compute_units_to_build(len(agents),
                               buildings[supply_fname].sum(),
                               target_vacancy)

    print("{:,} feasible buildings before running developer".format(
        len(dev.feasibility)))

    new_buildings = dev.pick(forms,
                             target_units,
                             parcel_size,
                             ave_unit_size,
                             total_units,
                             max_parcel_size=max_parcel_size,
                             min_unit_size=min_unit_size,
                             drop_after_build=True,
                             residential=residential,
                             bldg_sqft_per_job=bldg_sqft_per_job)

    sim.add_table("feasibility", dev.feasibility)
    year = sim.get_injectable('year')
    if new_buildings is None:
        return

    if len(new_buildings) == 0:
        return new_buildings
    
    if not isinstance(forms, list):
        # form gets set only if forms is a list
        new_buildings["form"] = forms

    if form_to_btype_callback is not None:
        new_buildings["building_type_id"] = new_buildings.\
            apply(form_to_btype_callback, axis=1)

    new_buildings["stories"] = new_buildings.stories.apply(np.ceil)
    new_buildings["note"] = "simulated"
    
    ret_buildings = new_buildings
    if add_more_columns_callback is not None:
        new_buildings = add_more_columns_callback(new_buildings)
        
    if year is not None:
        new_buildings["year_built"] = year

    print("Adding {:,} buildings with {:,} {}".
          format(len(new_buildings),
                 int(new_buildings[supply_fname].sum()),
                 supply_fname))

    print("{:,} feasible buildings after running developer".format(
        len(dev.feasibility)))

    old_buildings = buildings.to_frame(buildings.local_columns)
    old_buildings_all = buildings_all.to_frame(buildings.local_columns)
    new_buildings = new_buildings[buildings.local_columns]
    
    if remove_developed_buildings:
        redev_buildings = old_buildings.parcel_id.isin(new_buildings.parcel_id)
        redev_buildings_all = old_buildings_all.parcel_id.isin(new_buildings.parcel_id)
        l = len(old_buildings)
        drop_buildings = old_buildings[redev_buildings]
        drop_buildings_all = old_buildings_all[redev_buildings_all]
        old_buildings = old_buildings[np.logical_not(redev_buildings)]
        old_buildings_all = old_buildings_all[np.logical_not(redev_buildings_all)]
        l2 = len(old_buildings)
        print("before dropped l:" + str(l))
        print("after dropped l2: " + str(l2))
        #print redev_buildings
        #print drop_buildings
        if l2-l > 0:
            print("Dropped {} buildings because they were redeveloped".
                  format(l2 - l))

        for tbl in unplace_agents:
            agents = sim.get_table(tbl)
            agents = agents.to_frame(agents.local_columns)
            #displaced_agents = agents.building_id.isin(drop_buildings.index)
            displaced_agents = agents.building_id.isin(drop_buildings_all.index)
            print("Unplaced {} before: {}".format(tbl, len(agents.query(
                "building_id == -1"))))
            agents.building_id[displaced_agents] = -1
            print("Unplaced {} after: {}".format(tbl, len(agents.query(
                "building_id == -1"))))
            sim.add_table(tbl, agents)
    
    all_buildings = dev.merge(old_buildings_all, new_buildings)
    
    sim.add_table("buildings", all_buildings)

    return ret_buildings
    
def compute_range(travel_data, attr, travel_time_attr, dist, agg=np.sum):
    """
    Compute a zone-based accessibility query using the urbansim format
    travel data dataframe.

    Parameters
    ----------
    travel_data : dataframe
        The dataframe of urbansim format travel data.  Has from_zone_id as
        first index, to_zone_id as second index, and different impedances
        between zones as columns.
    attr : series
        The attr to aggregate.  Should be indexed by zone_id and the values
        will be aggregated.
    travel_time_attr : string
        The column name in travel_data to use as the impedance.
    dist : float
        The max distance to aggregate up to
    agg : function, optional, np.sum by default
        The numpy function to use for aggregation
    """
    travel_data = travel_data.reset_index(level=1)
    td_ind = travel_data.groupby('to_zone_id').sum()
    travel_data = travel_data[travel_data[travel_time_attr] < dist]
    travel_data["attr"] = attr[travel_data.to_zone_id].values
    travel_data = travel_data.groupby(level=0).attr.apply(agg)
    return pd.merge(td_ind, pd.DataFrame(travel_data), how='left', left_index=True, right_index=True).attr.fillna(0)
    
class SimulationSummaryData(object):
    """
    Keep track of zone-level and parcel-level output for use in the
    simulation explorer.  Writes the correct format and filenames that the
    simulation explorer expects.

    Parameters
    ----------
    run_number : int
        The run number for this run
    zone_indicator_file : optional, str
        A template for the zone_indicator_filename - use {} notation and the
        run_number will be substituted.  Should probably not be modified if
        using the simulation explorer.
    parcel_indicator_file : optional, str
        A template for the parcel_indicator_filename - use {} notation and the
        run_number will be substituted.  Should probably not be modified if
        using the simulation explorer.
    """

    def __init__(self,
                 run_number,
                 zone_indicator_file="runs/run{}_simulation_output.json",
                 parcel_indicator_file="runs/run{}_parcel_output.csv"):
        self.run_num = run_number
        self.zone_indicator_file = zone_indicator_file.format(run_number)
        self.parcel_indicator_file = \
            parcel_indicator_file.format(run_number)
        self.parcel_output = None
        self.zone_output = None

    def add_zone_output(self, zones_df, name, year, round=2):
        """
        Pass in a dataframe and this function will store the results in the
        simulation state to write out at the end (to describe how the simulation
        changes over time)

        Parameters
        ----------
        zones_df : DataFrame
            dataframe of indicators whose index is the zone_id and columns are
            indicators describing the simulation
        name : string
            The name of the dataframe to use to differentiate all the sources of
            the indicators
        year : int
            The year to associate with these indicators
        round : int
            The number of decimal places to round to in the output json

        Returns
        -------
        Nothing
        """
        # this creates a hierarchical json data structure to encapsulate
        # zone-level indicators over the simulation years.  "index" is the ids
        # of the shapes that this will be joined to and "year" is the list of
        # years. Each indicator then get put under a two-level dictionary of
        # column name and then year.  this is not the most efficient data
        # structure but since the number of zones is pretty small, it is a
        # simple and convenient data structure
        if self.zone_output is None:
            d = {
                "index": [int(x) for x in list(zones_df.index)],
                "years": []
            }
        else:
            d = self.zone_output

        assert d["index"] == [int(x) for x in list(zones_df.index)], "Passing in zones " \
            "dataframe that is not aligned on the same index as a previous " \
            "dataframe"

        if year not in d["years"]:
            d["years"].append(year)

        for col in zones_df.columns:
            d.setdefault(col, {})
            d[col]["original_df"] = name
            s = zones_df[col]
            dtype = s.dtype
            if dtype == "float64" or dtype == "float32":
                s = s.fillna(0)
                d[col][year] = [float(x) for x in list(s.round(round))]
            elif dtype == "int64" or dtype == "int32":
                s = s.fillna(0)
                d[col][year] = [int(x) for x in list(s)]
            else:
                d[col][year] = list(s)

        self.zone_output = d

    def add_parcel_output(self, new_parcel_output):
        """
        Add new parcel-level indicators to the parcel output.

        Parameters
        ----------
        new_parcel_output : DataFrame
            Adds a new set of parcel data for output exploration - this data
            is merged with previous data that has been added.  This data is
            generally used to capture new developments that UrbanSim has
            predicted, thus it doesn't override previous years' indicators

        Returns
        -------
        Nothing
        """
        if new_parcel_output is None:
            return

        if self.parcel_output is not None:
            # merge with old  parcel output
            self.parcel_output = \
                pd.concat([self.parcel_output, new_parcel_output]).\
                reset_index(drop=True)
        else:
            self.parcel_output = new_parcel_output

    def write_parcel_output(self,
                            add_xy=None):
        """
        Write the parcel-level output to a csv file

        Parameters
        ----------
        add_xy : dictionary (optional)
            Used to add x, y values to the output - an example dictionary is
            pasted below - the parameters should be fairly self explanatory.
            Note that from_epsg and to_epsg can be omitted in which case the
            coordinate system is not changed.  NOTE: pyproj is required
            if changing coordinate systems::

                {
                    "xy_table": "parcels",
                    "foreign_key": "parcel_id",
                    "x_col": "x",
                    "y_col": "y",
                    "from_epsg": 3740,
                    "to_epsg": 4326
                }


        Returns
        -------
        Nothing
        """
        if self.parcel_output is None:
            return

        po = self.parcel_output
        if add_xy is not None:
            x_name, y_name = add_xy["x_col"], add_xy["y_col"]
            xy_joinname = add_xy["foreign_key"]
            xy_df = sim.get_table(add_xy["xy_table"])
            po[x_name] = misc.reindex(xy_df[x_name], po[xy_joinname])
            po[y_name] = misc.reindex(xy_df[y_name], po[xy_joinname])

            if "from_epsg" in add_xy and "to_epsg" in add_xy:
                import pyproj
                p1 = pyproj.Proj('+init=epsg:%d' % add_xy["from_epsg"])
                p2 = pyproj.Proj('+init=epsg:%d' % add_xy["to_epsg"])
                x2, y2 = pyproj.transform(p1, p2,
                                          po[x_name].values,
                                          po[y_name].values)
                po[x_name], po[y_name] = x2, y2

        po.to_csv(self.parcel_indicator_file, index_label="development_id")

    def write_zone_output(self):
        """
        Write the zone-level output to a file.
        """
        if self.zone_output is None:
            return
        outf = open(self.zone_indicator_file, "w")
        json.dump(self.zone_output, outf)
        outf.close()
        
def run_feasibility(parcels, parcel_price_callback,
                    parcel_use_allowed_callback, residential_to_yearly=True,
                    parcel_filter=None, only_built=True, forms_to_test=None,
                    config=None, pass_through=[]):
    """
    Execute development feasibility on all parcels

    Parameters
    ----------
    parcels : DataFrame Wrapper
        The data frame wrapper for the parcel data
    parcel_price_callback : function
        A callback which takes each use of the pro forma and returns a series
        with index as parcel_id and value as yearly_rent
    parcel_use_allowed_callback : function
        A callback which takes each form of the pro forma and returns a series
        with index as parcel_id and value and boolean whether the form
        is allowed on the parcel
    residential_to_yearly : boolean (default true)
        Whether to use the cap rate to convert the residential price from total
        sales price per sqft to rent per sqft
    parcel_filter : string
        A filter to apply to the parcels data frame to remove parcels from
        consideration - is typically used to remove parcels with buildings
        older than a certain date for historical preservation, but is
        generally useful
    only_built : boolean
        Only return those buildings that are profitable - only those buildings
        that "will be built"
    forms_to_test : list of strings (optional)
        Pass the list of the names of forms to test for feasibility - if set to
        None will use all the forms available in ProFormaConfig
    config : SqFtProFormaConfig configuration object.  Optional.  Defaults to
        None
    pass_through : list of strings
        Will be passed to the feasibility lookup function - is used to pass
        variables from the parcel dataframe to the output dataframe, usually
        for debugging

    Returns
    -------
    Adds a table called feasibility to the sim object (returns nothing)
    """

    pf = sqftproforma.SqFtProForma(config) if config \
        else sqftproforma.SqFtProForma()

    df = parcels.to_frame()
    if parcel_filter:
        df = df.query(parcel_filter)
    # add prices for each use
    for use in pf.config.uses:
        # assume we can get the 80th percentile price for new development
        df[use] = parcel_price_callback(use)

    # convert from cost to yearly rent
    if residential_to_yearly:
        df["residential"] *= pf.config.cap_rate

    print("Describe of the yearly rent by use")
    print(df[pf.config.uses].describe())

    d = {}

    forms = forms_to_test or pf.config.forms
    for form in forms:
        print("Computing feasibility for form %s" % form)
        allowed = parcel_use_allowed_callback(form).loc[df.index]
        d[form] = pf.lookup(form, df[allowed], only_built=only_built,
                            pass_through=pass_through)
        if residential_to_yearly and "residential" in pass_through:
            d[form]["residential"] /= pf.config.cap_rate

    far_predictions = pd.concat(d.values(), keys=d.keys(), axis=1)
    far_predictions['residential'].max_profit = far_predictions['residential'].max_profit / np.power(far_predictions['residential'].max_profit_far*far_predictions['residential'].shape_area,1)
    far_predictions['industrial'].max_profit = far_predictions['industrial'].max_profit / np.power(far_predictions['industrial'].max_profit_far*far_predictions['industrial'].shape_area,1)
    far_predictions['retail'].max_profit = far_predictions['retail'].max_profit / np.power(far_predictions['retail'].max_profit_far*far_predictions['retail'].shape_area,1)
    far_predictions['office'].max_profit = far_predictions['office'].max_profit / np.power(far_predictions['office'].max_profit_far*far_predictions['office'].shape_area,1)
    #far_predictions['residential'].max_profit = np.divide(far_predictions['residential'].max_profit,far_predictions['residential'].max_dua)
    #far_predictions['residential'].max_profit[far_predictions['residential'].max_profit==-np.inf] = np.nan

    sim.add_table("feasibility", far_predictions)
