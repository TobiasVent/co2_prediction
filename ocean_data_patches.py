import pandas as pd
import numpy as np
import pickle

# Angenommen, df ist bereits mit den Daten geladen
# Erstellen Sie ein Beispiel-DataFrame, wenn Sie keine CSV-Datei haben
# df = pd.read_csv('path_to_your_file.csv')
df = pd.read_pickle('/data/stu231428/master_project/andi_processed_ORCA025.L46.LIM2vp.CFCSF6.MOPS.JRA.LP04-KLP002.hind_2017_df.pkl')
df = df[df["tmask"]==1]
#df = df.head(200000)

# df_current_month = df[df["time_centered"] == "2017-01-16 12:00:00"]
# df_next_month =  df[df["time_centered"] == "2017-03-16 12:00:00"]


# Beispiel-Daten erstellen (Ersetzen Sie dies durch das Laden Ihrer tatsächlichen Daten)
# data = {
#     'lat': np.repeat(np.linspace(-90, 90, 100), 100),
#     'lon': np.tile(np.linspace(-180, 180, 100), 100),
#     'value': np.random.random(10000)
# }
# df = pd.DataFrame(data)

# Stellen Sie sicher, dass die Latitude- und Longitude-Spalten vorhanden sind
latitudes = df['nav_lat'].unique()
longitudes = df['nav_lon'].unique()
time_centered = df["time_centered"].unique()
print("unique latitudes: "+ str(len(latitudes)))
print("unique longitudes: "+ str(len(longitudes)))

# Definieren Sie die Größe des Ausschnitts
patch_size = 31

# Funktion zum Extrahieren von 32x32-Ausschnitten
def extract_patches(df, latitudes, longitudes, patch_size):
    patches = []
    labels = []
    
        # if i == 0:
        #     df_current_month = df[df["time_centered"] == "2017-01-16 12:00:00"]
        #     df_next_month =  df[df["time_centered"] == "2017-02-16 12:00:00"]

    for timestep in range(len(time_centered)-1):

        for i in range(0, len(latitudes) - patch_size + 1, patch_size):
            for j in range(0, len(longitudes) - patch_size + 1, patch_size):
                if i%961 or j%961 == 0:
                    print("i: "+ str(i))
                    print("j: "+ str(j))
                lat_slice = latitudes[i:i + patch_size]
                lon_slice = longitudes[j:j + patch_size]
                
                patch = df[(df['nav_lat'].isin(lat_slice)) & (df['nav_lon'].isin(lon_slice))& (df["time_centered"]== time_centered[timestep])]
                # print(patch)
                # print(len(patch))
                if len(patch) == patch_size * patch_size:
                    center_lat = lat_slice[patch_size // 2]
                    center_lon = lon_slice[patch_size // 2]
                    label = df[(df['nav_lat'] == center_lat) & (df['nav_lon']==center_lon)& (df["time_centered"]== time_centered[timestep+1])]
                    labels.append(label["co2flux"].values)

                
                    patches.append(patch)
                        
        return patches,labels



def extract_patches2(df, latitudes, longitudes, time_centered, patch_size):
    #Create a lookup dictionary for faster access
#     data_dict = {}
#     data_list = []
#     #for idx, row in df.iterrows():
#     for row in df.itertuples(index=True):

#         #data_dict[(row['nav_lat'], row['nav_lon'], row['time_centered'])] = row
#         key = (row.nav_lat, row.nav_lon, row.time_centered)
#         data_dict[key] = (row[7],row[8],row[9],row[10],row[11],row[12],row[13],row[14],row[15],row[16],row[17],row[18],row[19],row[20],row[21])
# #        df.drop(index=row.Index, inplace=True)
#         # data_list.append((row["nav_lat"],row["nav_lon"],row["time_centered"]))
        
    # with open("data_dct.pkl", "wb") as f:
    #     pickle.dump(data_dict, f)
    patches = []
    labels = []

    data_dict = pickle.load(open("/data/stu231428/master_project/vision_transformer/data_dict.pkl","rb"))


    for timestep in range(len(time_centered) - 1):
        current_time = time_centered[timestep]
        next_time = time_centered[timestep + 1]

        for i in range(0, len(latitudes) - patch_size + 1, patch_size):
            if i%961 == 0:
                    print("i: "+ str(i))
                    #print("j: "+ str(j))
            for j in range(0, len(longitudes) - patch_size + 1, patch_size):
                lat_slice = latitudes[i:i + patch_size]
                lon_slice = longitudes[j:j + patch_size]
                
                patch = []
                for lat in lat_slice:
                    for lon in lon_slice:
                        key = (lat, lon, current_time)
                        if key in data_dict:
                            patch.append(data_dict[key])
                            #patch = df[(df['nav_lat'].isin(lat_slice)) & (df['nav_lon'].isin(lon_slice))& (df["time_centered"]== time_centered[timestep])]
                        else:
                            patch = []
                            break
                    if not patch:
                        break
                
                if patch and len(patch) == patch_size * patch_size:
                    center_lat = lat_slice[patch_size // 2]
                    center_lon = lon_slice[patch_size // 2]
                    label_key = (center_lat, center_lon, next_time)
                    if label_key in data_dict:
                        #label = df[(df['nav_lat'] == center_lat) & (df['nav_lon']==center_lon)& (df["time_centered"]== time_centered[timestep+1])]
                        try:
                            label = data_dict[label_key][13]
                        except:
                            break
                        patches.append(patch)
                        labels.append(label)

    return patches, labels


# Extrahieren Sie die Patches
patches,labels =  extract_patches2(df, latitudes, longitudes,time_centered, patch_size)
#patches = extract_patches(df, latitudes, longitudes, patch_size)
# patches_array = np.array([patch.values for patch in patches])
# labels_array = np.array(labels)
# np.savez('debug.npz', patches=patches_array, labels=labels_array)
# Anzeigen des ersten Patches
with open("train_list.pkl", "wb") as f:
    pickle.dump(patches, f)

with open("train_labels_list_.pkl", "wb") as f:
    pickle.dump(labels, f)

