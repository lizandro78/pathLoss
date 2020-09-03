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
X_test.to_csv(file_name, index=False)

#%%
#Create a grid to execute predictions
#Each point has a RSSI prediction
bts = ImportHelper.GetBts()
grid = CdbFactory.CreateCdb(df_meds["lat"].min(), df_meds["lon"].min(),
                            df_meds["lat"].max(),
                            df_meds["lon"].max(), 20)

#%%
gridComDist = ImportHelper.PutDistancesFromBtss(bts, grid)
coverMaps_arr = model.predict(gridComDist[features])
coverMaps = pd.DataFrame(data=coverMaps_arr, columns=targets)
coverMapsFinal = pd.concat([coverMaps, gridComDist], axis = 1)
coverMapsFinal.to_pickle("./dataOut/coverMapsFinal.pkl")

#%%
map_data = pd.read_pickle("./dataOut/coverMapsFinal.pkl")
map_data["best_server"] = map_data.drop(["dist_1","dist_2","dist_3","lat","lon"],axis=1).max(axis=1)

#%%
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

map_data["nivel_cobertura"] = map_data["best_server"].apply(cover_level)

#%%

BBox = ((map_data["lon"].min(), map_data["lon"].max(), map_data["lat"].min(),map_data["lat"].max()))
#BBox = ((map_data["lon"].min()-0.002, map_data["lon"].max()+0.002, map_data["lat"].min()-0.002,map_data["lat"].max()+0.002))
mapGeoRef = plt.imread('./figs/map4.png')

fig, ax = plt.subplots(figsize = (8,7))
sns.scatterplot(x = "lon", y = "lat", alpha= 0.85, ax =ax, legend='brief', zorder=1, data = map_data, hue = map_data["nivel_cobertura"].tolist())
#sns.scatterplot(x = "lon", y = "lat", alpha= 0.55, ax =ax, legend='brief',shade=True, cmap="viridis",shade_lowest=False, zorder=1, data = map_data)
#shade=True, cmap="viridis",shade_lowest=False,


ax.set_title('Cover Map by RSSI level')
ax.set_xlim(BBox[0],BBox[1])
ax.set_ylim(BBox[2],BBox[3])
ax.imshow(mapGeoRef, zorder=0, extent = BBox, aspect= 'equal')
plt.show()
