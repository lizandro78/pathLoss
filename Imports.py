import numpy as np
import pandas as pd
import os
from GeoUtils import GeoUtils
from Cdb import CdbFactory


class ImportHelper(object):

    @staticmethod
    def CriarDataset(meds_path, cdb_path):
        btss = ImportHelper.GetBts()
        if (not os.path.isfile(meds_path)):
            print("Importing Bts...")
            # print(btss.head(9))
            print("Impoting field data...")
            meds = ImportHelper.GetMedicoes()
            print("Taking RSSI")
            meds = ImportHelper.PutRssiColumns(btss, meds)
            print("Calc Deltas")
            meds = ImportHelper.PutDeltas(meds)
            print("Calc dists and az...")
            meds = ImportHelper.PutDistancesFromBtss(btss, meds)
            meds = ImportHelper.PutAzimuthsFromBtss(btss, meds)
            print("Calc...delay")
            meds = ImportHelper.PutDelay(meds)
            meds.to_csv(meds_path)
        else:
            meds = pd.read_csv(meds_path)
        if (not os.path.isfile(cdb_path)):
            print("Creating...Cdbs")
            cdb = ImportHelper.CreateCdbs(meds, btss)
            print("Exporting to Csv...")
            cdb.to_csv(cdb_path)
        else:
            cdb = pd.read_csv(cdb_path)
        return meds, cdb

    @staticmethod
    def GetBts():
        dir = os.path.dirname(os.path.abspath(__file__))
        df_bts = pd.read_csv(f"{dir}\\dataIn\\Bts.csv")
        return df_bts

    @staticmethod
    def GetMedicoes():
        dir = os.path.dirname(os.path.abspath(__file__))
        df_meds = pd.read_csv(f"{dir}\\meds_full2.csv")
        df_meds['indoor'] = df_meds['local'] != 'Outdoor'
        return df_meds

    @staticmethod
    def PutRssiColumns(df_bts, df_meds):
        df_bts['cchx'] = 'X' + df_bts['cch'].apply(str)
        df_resp = df_meds[df_bts['cchx']]
        df_resp.columns = np.array(df_bts['RssiId'])
        df = df_resp.join(df_meds[['lat', 'lon', 'indoor']])
        return df

    @staticmethod
    def PutDeltas(meds):
        # Bts1
        meds['delta_1_12'] = meds['rssi_1_1'] - meds['rssi_1_2']
        meds['delta_1_13'] = meds['rssi_1_1'] - meds['rssi_1_3']
        meds['delta_1_23'] = meds['rssi_1_2'] - meds['rssi_1_3']
        # Bts2
        meds['delta_2_12'] = meds['rssi_2_1'] - meds['rssi_2_2']
        meds['delta_2_13'] = meds['rssi_2_1'] - meds['rssi_2_3']
        meds['delta_2_23'] = meds['rssi_2_2'] - meds['rssi_2_3']
        # Bts3
        meds['delta_3_12'] = meds['rssi_3_1'] - meds['rssi_3_2']
        meds['delta_3_13'] = meds['rssi_3_1'] - meds['rssi_3_3']
        meds['delta_3_23'] = meds['rssi_3_2'] - meds['rssi_3_3']
        return meds

    @staticmethod
    def PutDistancesFromBtss(df_bts, df_meds):
        lat1, lon1 = df_bts['lat'][0], df_bts['lon'][0]
        lat2, lon2 = df_bts['lat'][3], df_bts['lon'][3]
        lat3, lon3 = df_bts['lat'][6], df_bts['lon'][6]
        lats, lons = np.array(df_meds[['lat']]), np.array(df_meds[['lon']])
        print("put dist...1/3")
        df_meds['dist_1'] = GeoUtils.distanceInKm(lat1, lon1, lats, lons)
        print("put dist...2/3")
        df_meds['dist_2'] = GeoUtils.distanceInKm(lat2, lon2, lats, lons)
        print("put dist...3/3")
        df_meds['dist_3'] = GeoUtils.distanceInKm(lat3, lon3, lats, lons)
        return df_meds

    @staticmethod
    def PutAzimuthsFromBtss(df_bts, df_meds):
        lat1, lon1 = df_bts['lat'][0], df_bts['lon'][0]
        lat2, lon2 = df_bts['lat'][3], df_bts['lon'][3]
        lat3, lon3 = df_bts['lat'][6], df_bts['lon'][6]
        lats, lons = np.array(df_meds[['lat']]), np.array(df_meds[['lon']])
        print("put azim...1/3")
        df_meds['ang_1'] = GeoUtils.AzimuthAtoB(lat1, lon1, lats, lons)
        df_meds['cos_1'] = np.cos(np.deg2rad(df_meds['ang_1']))
        df_meds['sin_1'] = np.sin(np.deg2rad(df_meds['ang_1']))
        df_meds['tg_1'] = 1 / (1 + np.tan(np.deg2rad(df_meds['ang_1'])))  # relu(softmax)
        print("put azim...2/3")
        df_meds['ang_2'] = GeoUtils.AzimuthAtoB(lat2, lon2, lats, lons)
        df_meds['cos_2'] = np.cos(np.deg2rad(df_meds['ang_2']))
        df_meds['sin_2'] = np.sin(np.deg2rad(df_meds['ang_2']))
        df_meds['tg_2'] = 1 / (1 + np.tan(np.deg2rad(df_meds['ang_2'])))  ##relu(softmax)
        print("put azim...3/3")
        df_meds['ang_3'] = GeoUtils.AzimuthAtoB(lat3, lon3, lats, lons)
        df_meds['cos_3'] = np.cos(np.deg2rad(df_meds['ang_3']))
        df_meds['sin_3'] = np.sin(np.deg2rad(df_meds['ang_3']))
        df_meds['tg_3'] = 1 / (1 + np.tan(np.deg2rad(df_meds['ang_3'])))  # relu(softmax)

    @staticmethod
    def PutDelay(meds):

        meds["delay_1"] = meds["dist_1"].apply(lambda x: int(np.round(x / 0.234)))
        meds["delay_2"] = meds["dist_2"].apply(lambda x: int(np.round(x / 0.234)))
        meds["delay_3"] = meds["dist_3"].apply(lambda x: int(np.round(x / 0.234)))
        meds["delay_12"] = meds["delay_1"] - meds["delay_2"]
        meds["delay_13"] = meds["delay_1"] - meds["delay_3"]
        meds["delay_23"] = meds["delay_2"] - meds["delay_3"]
        return meds

    @staticmethod
    def CreateCdbs(meds, bts):
        # Define Limits
        latMax, latMin = meds['lat'].max(), meds['lat'].min()
        lonMax, lonMin = meds['lon'].max(), meds['lon'].min()
        # Creating the cdb
        cdb = CdbFactory.CreateCdb(latMin, lonMin, latMax, lonMax, 20)
        cdb = ImportHelper.PutDistancesFromBtss(bts, cdb)
        cdb = ImportHelper.PutAzimuthsFromBtss(bts, cdb)
        cdb = ImportHelper.PutDelay(cdb)
        return cdb
