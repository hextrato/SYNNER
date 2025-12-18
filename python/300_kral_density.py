import argparse
import json
import numpy as np
from scipy.spatial import cKDTree
import pandas as pd
import math

'''
Example: 
python 300_kral_density.py --config kaggle-fedesoriano-stroke-prediction-dataset.config.json --maxradius 0.25 --minneighs 16  --maxneighs 32  --kdims 32 --variation C016-032
python 300_kral_density.py --config kaggle-fedesoriano-stroke-prediction-dataset.config.json --maxradius 0.25 --minneighs 32  --maxneighs 64  --kdims 32 --variation C032-064
python 300_kral_density.py --config kaggle-fedesoriano-stroke-prediction-dataset.config.json --maxradius 0.25 --minneighs 64  --maxneighs 128 --kdims 32 --variation C064-128
python 300_kral_density.py --config kaggle-fedesoriano-stroke-prediction-dataset.config.json --maxradius 0.25 --minneighs 128 --maxneighs 256 --kdims 32 --variation C128-256
'''

print("")
print("=================================================================")

# ==========
# Parameters
# ==========

print("")
print("-------------------------")
print("Parameters...")

# Initialize parser
parser = argparse.ArgumentParser()
parser.add_argument('--config', help='Config file', required=True)
# parser.add_argument('--embeddings', help='JSON Embeddings', required=True)
parser.add_argument('--kdims', type=int, help='Config file', required=False)
parser.add_argument('--maxradius', type=float, help='Max radius (min=0.05,max=1.0)', required=True)
parser.add_argument('--minneighs', type=int, help='Min neighbors', required=True)
parser.add_argument('--maxneighs', type=int, help='Max neighbors', required=True)
parser.add_argument('--variation', help='Variation name', required=True)

args = parser.parse_args()

P_CONFIG_FILE     = open("../config/"+args.config)
P_CONFIG_DATA     = json.load(P_CONFIG_FILE)
P_KDIMS           = args.kdims
#P_FILE_EMBEDDINGS = args.embeddings
P_MAX_RADIUS      = args.maxradius
P_MIN_NEIHGS      = args.minneighs
P_MAX_NEIHGS      = args.maxneighs
P_VARIATION       = args.variation
P_CONFIG_FILE.close()

P_DATA_PATH   = P_CONFIG_DATA["datapath"]
P_EMBD_PATH   = P_CONFIG_DATA["embdpath"]
P_DATA_FOLDER = P_CONFIG_DATA["datafolder"]
P_DATA_SET    = P_CONFIG_DATA["dataset"]
P_DROP_COLS   = P_CONFIG_DATA["drop-columns"]
P_LABELS      = P_CONFIG_DATA["labels"]
P_TL_VALUES   = P_CONFIG_DATA["target-label-values"]
P_SUBJECT     = P_CONFIG_DATA["subject-column"]
P_SUBJECT_TYPE= P_CONFIG_DATA["subject-type"]

print("Data Path   = ",P_DATA_PATH)
print("Data Folder = ",P_DATA_FOLDER)
print("Data Set    = ",P_DATA_SET)
print("Drop        = ",P_DROP_COLS)
print("Labels      = ",P_LABELS)
print("Max(radius) = ",P_MAX_RADIUS)
P_STEP_RADIUS = 0.005
print("Step(radius)= ",P_STEP_RADIUS)

P_FILE_EMBEDDINGS = P_EMBD_PATH+"/"+P_DATA_FOLDER+"/KRAL_"+P_DATA_SET+"_D"+str(P_KDIMS)+"-MODEL-head-train.json"
print("Embeddings  = ",P_FILE_EMBEDDINGS)
f = open(P_FILE_EMBEDDINGS)
P_JSON_EMBEDDINGS = json.load(f)
f.close()
print("Instances   = ",len(P_JSON_EMBEDDINGS))

#for kid in P_JSON_EMBEDDINGS:
#    print(kid,P_JSON_EMBEDDINGS[kid])

NP_EMBEDDINGS = np.array(list(P_JSON_EMBEDDINGS[kid] for kid in P_JSON_EMBEDDINGS))
print("k-Shape     = ",NP_EMBEDDINGS.shape)

KDTREE = cKDTree(NP_EMBEDDINGS)

#########################
# Neighbors Analysis    #
#########################

DF_NEIGH_ANALYSIS = pd.DataFrame()
for radius in np.arange(P_STEP_RADIUS, P_MAX_RADIUS + P_STEP_RADIUS, P_STEP_RADIUS):
    L_radius = "{:.3f}".format(radius)
    print ("Radius",str(L_radius))
    numNeighbors = []  # List to store the number of neighbors for each data point
    namNeighbors = []  # List to store the neighbors' names
    # Loop through each point in the dataframe
    for neigh in P_JSON_EMBEDDINGS:
        # print(neigh,type(neigh))
        # Get points within radius "i" of the current point
        # print(P_JSON_EMBEDDINGS[neigh])
        neigh_array = np.array(P_JSON_EMBEDDINGS[neigh])
        # print(neigh_array)
        neighbors = KDTREE.query_ball_point(neigh_array, radius)
        numNeighbors.append(len(neighbors) - 1)  # Subtract 1 from count, point can't be its own neighbor
        namNeighbors.append(neigh)

    # Create a pandas series with the neighbor counts for the current radius
    # Essentially a one-dimensional array with an axis label
    numNeighborSeries = pd.Series(numNeighbors, name=f"R:{L_radius}", index=namNeighbors)
    # print(numNeighborSeries)
    # Concatenate the new series to the existing DataFrame along the x-axis
    DF_NEIGH_ANALYSIS = pd.concat([DF_NEIGH_ANALYSIS, numNeighborSeries], axis=1)
    # print(DF_NEIGH_ANALYSIS)

print(DF_NEIGH_ANALYSIS.shape)
print(DF_NEIGH_ANALYSIS)

P_FILE_NEIGHBORS = P_EMBD_PATH+"/"+P_DATA_FOLDER+"/KRAL_"+P_DATA_SET+"_D"+str(P_KDIMS)+"-MODEL-head-train-"+P_VARIATION+"-NEIGHS.csv"
print("Neighbors = ",P_FILE_NEIGHBORS)
DF_NEIGH_ANALYSIS.to_csv(P_FILE_NEIGHBORS)  

DF_RADIUS_ANALYSIS = pd.DataFrame()

for seriename, serie in DF_NEIGH_ANALYSIS.items():
    print(seriename)
    datStats = []  
    namStats = ["Zeros(%)","%Outliers(<"+str(P_MIN_NEIHGS)+")","%Density(>"+str(P_MAX_NEIHGS)+")","Min","Avg","Max"]
    
    #print("zeros",len(serie[serie == 0]))
    #print("min",np.min(serie[serie > 0]))
    #print("avg",round(np.mean(serie[serie > 0]),1))
    #print("max",np.max(serie[serie > 0]))
    datStats.append((100 * len(serie[serie == 0]) / NP_EMBEDDINGS.shape[0]))
    datStats.append((100 * len(serie[serie < P_MIN_NEIHGS]) / NP_EMBEDDINGS.shape[0]))
    datStats.append((100 * len(serie[serie > P_MAX_NEIHGS]) / NP_EMBEDDINGS.shape[0]))
    datStats.append(np.min(serie[serie > 0]))
    datStats.append((np.mean(serie[serie > 0])))
    datStats.append(np.max(serie[serie > 0]))
    statsSeries = pd.Series(datStats, name=seriename, index=namStats)
    # DF_RADIUS_ANALYSIS = pd.concat([DF_RADIUS_ANALYSIS, statsSeries.astype(int)], axis=1)
    DF_RADIUS_ANALYSIS = pd.concat([DF_RADIUS_ANALYSIS, statsSeries], axis=1)
    
print(DF_RADIUS_ANALYSIS.shape)
print(DF_RADIUS_ANALYSIS)

P_FILE_RADIUS = P_EMBD_PATH+"/"+P_DATA_FOLDER+"/KRAL_"+P_DATA_SET+"_D"+str(P_KDIMS)+"-MODEL-head-train-"+P_VARIATION+"-RADIUS.csv"
print("Radius Analysis = ",P_FILE_RADIUS)
DF_RADIUS_ANALYSIS.to_csv(P_FILE_RADIUS)  
                
################################################################################################################################################################
exit(0)
################################################################################################################################################################

