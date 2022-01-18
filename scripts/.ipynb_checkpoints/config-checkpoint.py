
#Define directory for IO
data_root = '/network/group/aopp/predict/TIP016_PAXTON_RPSPEEDY/ML4L/'
xdata = 'xdata.nc'

#CDS

#A dictionary of CDS features
xvariables = {
#sfc
134:'surface_pressure',
165: '10m_u_component_of_wind',
166: '10m_v_component_of_wind',
167: '2m_temperature',
#sfc_skin: 
243: 'forecast_albedo',
169: 'surface_solar_radiation_downwards',
175: 'surface_thermal_radiation_downwards',
228: 'total_precipitation',
141: 'snow_depth',
168: '2m_dewpoint_temperature'
}

#Other parameters not availabel from ERA-5 month mean 
# 151 - Mean sea level pressure. 
# 15 - UV visible albedo for direct radiation 
# 16 - UV visible albedo for diffuse radiation. 
# 17 - Near IR albedo for direct radiation. 
#18: - Near IR albedo for diffuse radiation
# 26 - Lake cover. 
# 27 - Low vegitation cover. 
# 28 - High vegitation cover. 
# 35 - Ice temperature layer 1. 
# 36 - Ice temperature layer 2. 
# 43 - Soil type. 
# 74 - Standard deviation of filtered subgrid orography. 
# 129 - Geopotential. 
# 160 - Standard deviation of orography. 
# 161: - Anisotropy of sub-gridscale orography
# 162: - Angle of sub-gridscale orography
# 163: - Slope of sub-gridscale orography
#Available at model level 137 (ml_skin): 
# 130 - temperature
# 131 -  U component of wind
# 133 - V component of wind
#172: - Land-sea mask 

#MODIS
wget_file = 'tmp/get_MODIS_data.txt'


#Time selection
min_year = 2011
max_year = 2019
years = [str(i) for i in range(min_year,max_year+1)]
months = [f"{i:02d}" for i in range(1,12+1)] #every month
times= ["00:00"]


#Clean data file
clean_data = data_root + 'cleaned_data.pkl'

#Specify times to split your cleaned data on
training_limit   = '20171201' 
validation_limit = '20181201'

#Load from remote for x/y data
load_x_data_from_remote = False
load_y_data_from_remote = True




