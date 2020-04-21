#REMM Utility Constraint for Utah County, Developed by Xiao Li WFRC & Tim Hereth MAG.
import arcpy
print "Utility Restriction start"

arcpy.env.overwriteOutput = True

devbuilding = r"..\utahdevelopedparcels.csv"
tabledir = r"UtilityRestriction.gdb"
tableall = r"UtilityRestriction.gdb\utahdevelopeparcels"
table = "utahdevelopeparcels"

arcpy.TableToTable_conversion(devbuilding, tabledir, table)

spRef = r"projection.prj"

point = "pointlyr"
pointfeature = r"UtilityRestriction.gdb\utahdevpoint"
arcpy.MakeXYEventLayer_management(tableall, "x", "y", point, spRef)

arcpy.CopyFeatures_management(point,pointfeature)


gridSum  = r"UtilityRestriction.gdb\gridSum"
arcpy.Statistics_analysis(pointfeature, gridSum, [["total_residential_units", "SUM"],["total_job_spaces", "SUM"]], "gridID")

gridShape = r"UtilityRestriction.gdb\UtahGrid"
gridlayer = "grid_lyr"
arcpy.MakeFeatureLayer_management(gridShape, gridlayer)
arcpy.AddJoin_management(gridlayer, "GRIDID", gridSum, "gridID", "KEEP_ALL")
arcpy.CalculateField_management(gridlayer,"UtahGrid.AdjustedUnits","!gridSum.SUM_total_residential_units! * !UtahGrid.BufferFriction!","PYTHON_9.3")
arcpy.SelectLayerByAttribute_management(gridlayer, "NEW_SELECTION", 'UtahGrid.AdjustedUnits >= 10')

f = open(r'..\YEAR.txt', 'r')
year = str(f.read()) 
f.close()

resdevbuffer = r"UtilityRestriction.gdb\resdevbuffer_" + year
arcpy.Buffer_analysis(gridlayer, resdevbuffer, "0.5 Miles", "FULL", "ROUND", "ALL")

utahparcels = r"UtilityRestriction.gdb\utahparcelspoint"
utahparcelslyr = "utahparcellyr"
arcpy.MakeFeatureLayer_management(utahparcels, utahparcelslyr)

arcpy.SelectLayerByLocation_management (utahparcelslyr, "INTERSECT", resdevbuffer)
arcpy.SelectLayerByLocation_management (utahparcelslyr, None, None, "", "SWITCH_SELECTION")

arcpy.TableToTable_conversion (utahparcelslyr, r"..\data", 'developableparcels.dbf')

print "Utility Restriction end"



