# Importing libs

from datetime import datetime
from math import sqrt

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor

from Cdb import CdbFactory
from Imports import ImportHelper
from knnRegressor import MultiRegressor

#%%
#Getting Measurements

df_meds = pd.read_csv("./dataIn/meds_pandas.csv")
targets = ['rssi_1_1', 'rssi_1_2', 'rssi_1_3', 'rssi_2_1', 'rssi_2_2', 'rssi_2_3', 'rssi_3_1', 'rssi_3_2', 'rssi_3_3']
features = ['dist_1', 'dist_2', 'dist_3']


#%%
#Separate training and test
#Obs.: Only outdoor to this case

df_out = df_meds[df_meds['indoor'] == False]
print(df_out.shape)  # 3064 x 43

#%%
# Search area

print("Latitude Min:{0} Max:{1}".format(df_out['lat'].min(), df_out['lat'].max()))
print("Longitude Min:{0} Max:{1}".format(df_out['lon'].min(), df_out['lon'].max()))

#%%
#Separate features and targets

train, test = train_test_split(df_out, test_size = 0.2)
X_train = train[features] # dist1, 2  e 3 (column selection)
X_test = test[features]
Y_train = train[targets]
Y_test = test[targets]
#%%
#Create K-NN Modelfor each target (BTS)

cv_parms = {'n_neighbors': list(range(1, 24))}
model = MultiRegressor(cv_parms=cv_parms, estimator=KNeighborsRegressor(weights='distance'))
model.fit(X_train, Y_train)

#%%
# Doing predictions

y_preds = model.predict(X_test)
for col in range(0, y_preds.shape[1]):
    y_p = y_preds[:, col].reshape(-1, 1)
    y_real = Y_test.values[:, col].reshape(-1, 1)
    print("RMSE Modelo {0}:{1} dB".format(col, sqrt(mean_squared_error(y_p, y_real))))
    #Root mean square error RMSE
    #Including values on the X_test
    col_pred = 'RSSI_pred_{0}'.format(col)
    col_real = 'RSSI_real_{0}'.format(col)
    X_test[col_pred] = y_p
    X_test[col_real] = y_real
#%%
# Putting geographic coordinates
X_test['lat'] = test['lat']
X_test['lon'] = test['lon']

print(X_test.head())
#%%
# Create a ".csv" file
file_name = "./dataOut/test_{0}.csv".format(datetime.now().strftime("%d_%b_%Y_%I_%M_%S_%p"))
print(file_name)
X_test.to_csv(file_name, index=True)

#%%
#Create a grid to execute predictions
#Each point has a RSSI prediction
bts = ImportHelper.GetBts()
grid = CdbFactory.CreateCdb(df_meds["lat"].min(), df_meds["lon"].min(),
                            df_meds["lat"].max(),
                            df_meds["lon"].max(), 20)

#%%
#Mapping grid snd put on a file
gridComDist = ImportHelper.PutDistancesFromBtss(bts, grid)
coverMaps_arr = model.predict(gridComDist[features])
coverMaps = pd.DataFrame(data=coverMaps_arr, columns=targets)
coverMapsFinal = pd.concat([coverMaps, gridComDist], axis = 1)
coverMapsFinal.to_pickle("./dataOut/coverMapsFinal.pkl")

#%%
#Create a file to coverage with best BTS server
map_data = pd.read_pickle("./dataOut/coverMapsFinal.pkl")
map_data["best_server"] = map_data.drop(["dist_1","dist_2","dist_3","lat","lon"],axis=1).max(axis=1)

#%%
#Define coverage levels
def cover_level(rssi):
    if rssi >= -65:
        return "Excellent"
    elif rssi >= -75:
        return "Good"
    elif rssi >= -85:
        return "Regular"
    elif rssi >= -90:
        return "Bad"
    else:
        return "Shadow"

#%%
#Add coverage levels of previous step
map_data["nivel_cobertura"] = map_data["best_server"].apply(cover_level)

#%%
#GEnerate the coverage plot

#Build a box to plt results on the georeferenced map
BBox = ((map_data["lon"].min()-0.002, map_data["lon"].max()+0.002, map_data["lat"].min()-0.002,map_data["lat"].max()+0.002))
mapGeoRef = plt.imread('./figs/map4.png')

#plot the results over the georeferenced map
fig, ax = plt.subplots(figsize = (8,7))
sns.scatterplot(x = "lon", y = "lat", alpha= 0.75, ax =ax, legend='brief', zorder=1, data = map_data, hue = map_data["nivel_cobertura"].tolist())
ax.set_title('Coverage Map by RSSI level')
ax.set_xlim(BBox[0],BBox[1])
ax.set_ylim(BBox[2],BBox[3])
ax.imshow(mapGeoRef, zorder=0, extent = BBox, aspect= 'equal')
plt.show()

#%%

#Creating Dataframe to RSSI analyses
X_test.to_pickle("./dataOut/rssiReal.pkl")
df_Rssi = pd.read_pickle("./dataOut/rssiReal.pkl")
df_Rssi["index"] = range(1,614)
df_Rssi["error"] = df_Rssi["RSSI_pred_8"] -  df_Rssi["RSSI_real_8"]
#%%
# style
plt.style.use('seaborn-darkgrid')

# create a color palette
palette = plt.get_cmap('Set1')

# multiple line plot
num = 0
for column in df_Rssi.drop(["index","dist_1","dist_2","dist_3", "lat", "lon",
                            "RSSI_pred_0","RSSI_real_0","RSSI_pred_1","RSSI_real_1",
                            "RSSI_pred_2","RSSI_real_2","RSSI_pred_3","RSSI_real_3",
                            "RSSI_pred_4","RSSI_real_4","RSSI_pred_5","RSSI_real_5",
                            "RSSI_pred_6","RSSI_real_6","RSSI_pred_7","RSSI_real_7", "RSSI_pred_8","RSSI_real_8"], axis = 1):
    num += 1
    plt.plot(df_Rssi["index"], df_Rssi[column], marker='', color=palette(num), linewidth=1, alpha=0.9, label=column)

# Add legend
plt.legend(loc=2, ncol=2)

# Add titles
plt.title("Error between Predict RSSI and Real RSSI ", loc='left', fontsize=12, fontweight=0, color='orange')
plt.xlabel("Test Points")
plt.ylabel("RSSI Level")
plt.show()

