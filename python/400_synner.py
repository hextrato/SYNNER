import argparse
import json
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.distance import pdist
import pandas as pd
import math
import random
from hextrato import synner

'''
Example:
python 400_synner.py --config kaggle-fedesoriano-stroke-prediction-dataset.config.json      --maxradius 0.155 --minneighs 16  --maxneighs 32   --kdims 32  --variation S016-032
python 400_synner.py --config kaggle-fedesoriano-stroke-prediction-dataset.config.json      --maxradius 0.170 --minneighs 32  --maxneighs 64   --kdims 32  --variation S032-064
python 400_synner.py --config kaggle-fedesoriano-stroke-prediction-dataset.config.json      --maxradius 0.185 --minneighs 64  --maxneighs 128  --kdims 32  --variation S064-128
python 400_synner.py --config kaggle-fedesoriano-stroke-prediction-dataset.config.json      --maxradius 0.195 --minneighs 128 --maxneighs 256  --kdims 32  --variation S128-256
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
parser.add_argument('--kdims', type=int, help='Config file', required=False)
parser.add_argument('--maxradius', type=float, help='Max radius (min=0.05,max=1.0)', required=True)
parser.add_argument('--minneighs', type=int, help='Min neighbors', required=True)
parser.add_argument('--maxneighs', type=int, help='Max neighbors', required=True)
parser.add_argument('--variation', help='Variation name', required=True)

args = parser.parse_args()

P_CONFIG_FILE     = open("../config/"+args.config)
P_CONFIG_DATA     = json.load(P_CONFIG_FILE)
P_KDIMS           = args.kdims
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
if "synner" in P_CONFIG_DATA:
    P_SYNNER_COLS = P_CONFIG_DATA["synner"]
    print(P_SYNNER_COLS)
#else:
#    P_SYNNER_COLS = {}
#    print("Missing 'synner' entry in config file")
#    exit(1)

print("Data Path   = ",P_DATA_PATH)
print("Data Folder = ",P_DATA_FOLDER)
print("Data Set    = ",P_DATA_SET)
print("Drop        = ",P_DROP_COLS)
print("Labels      = ",P_LABELS)
print("Max(radius) = ",P_MAX_RADIUS)

###############
##  DATASET  ##
###############

P_FILE_DATASET = P_DATA_PATH+"/"+P_DATA_FOLDER+"/"+P_DATA_SET+"-train.csv"
print("Dataset     = ",P_FILE_DATASET)
f = open(P_FILE_DATASET)
P_DF_TRAIN = pd.read_csv(f)
f.close()
print("Columns     = ",len(P_DF_TRAIN),"(dataset)")
#if P_DROP_COLS:
#    print("Drop columns  = ",P_DROP_COLS)
#    P_DF_TRAIN = P_DF_TRAIN.drop(columns=P_DROP_COLS,axis=1) 
print(P_DF_TRAIN.columns)

################
## EMBEDDINGS ##
################

P_FILE_EMBEDDINGS = P_EMBD_PATH+"/"+P_DATA_FOLDER+"/KRAL_"+P_DATA_SET+"_D"+str(P_KDIMS)+"-MODEL-head-train.json"
print("Embeddings  = ",P_FILE_EMBEDDINGS)
f = open(P_FILE_EMBEDDINGS)
P_JSON_EMBEDDINGS = json.load(f)
f.close()
print("Instances   = ",len(P_JSON_EMBEDDINGS),"(embeddings)")

#################
##    KDTREE   ##
#################

NP_EMBEDDINGS = np.array(list(P_JSON_EMBEDDINGS[kid] for kid in P_JSON_EMBEDDINGS))
NP_INSTANCES = list(kid for kid in P_JSON_EMBEDDINGS)

#print(NP_INSTANCES[0])
#print(NP_EMBEDDINGS[0])
#print(P_JSON_EMBEDDINGS[NP_INSTANCES[0]])

print("k-Shape     = ",NP_EMBEDDINGS.shape)
KDTREE = cKDTree(NP_EMBEDDINGS)
print("KDTREE.shape = ",KDTREE.data.shape,"(shape)")

#################
##  MAIN LOOP  ##
#################

source_instance_count = 0
instance_count = 0
synnerDataFrame = P_DF_TRAIN.iloc[0:0] # pd.DataFrame()
# synnerDataFrame["_source_subject_"]=""
synSeriesSet = []
for neigh in P_JSON_EMBEDDINGS:
    source_instance_count += 1
    neigh_vector = np.array(P_JSON_EMBEDDINGS[neigh])
    print("###############################################")
    print("#",source_instance_count,neigh,neigh_vector[0:5],"...")
    # local_cluster = KDTREE.query_ball_point(x = neigh_vector, k = P_MAX_NEIHGS, distance_upper_bound  = P_MAX_RADIUS)
    # local_cluster = KDTREE.query(x = neigh_vector, k = P_MAX_NEIHGS, distance_upper_bound  = P_MAX_RADIUS)
    # cluster_dists,cluster_inds = KDTREE.query(x = neigh_vector, k = P_MAX_NEIHGS+1, distance_upper_bound  = P_MAX_RADIUS)
    # cluster_inds = KDTREE.query_ball_point(x = neigh_vector, r = P_MAX_RADIUS)
    cluster_dists,cluster_inds = KDTREE.query(x = neigh_vector, k = P_MAX_NEIHGS+1, distance_upper_bound = P_MAX_RADIUS)
    # print(cluster_dists)
    # print(cluster_inds)
    cluster_neigh_id = []
    cluster_neigh_l2 = []
    cluster_neigh_ws = []
    cluster_neigh_count = 0
    for idx in range(len(cluster_inds)):
        i = cluster_inds[idx]
        if idx > 0 and i < KDTREE.data.shape[0]: # and str(cluster_dists[idx]).isnumeric():
            # print("Neigh >>>",idx,i,"dist=",cluster_dists[idx])
            # print(NP_INSTANCES[i],NP_EMBEDDINGS[i][0:4],"...","dist=",pdist([neigh_vector,NP_EMBEDDINGS[i]]),np.linalg.norm(neigh_vector-NP_EMBEDDINGS[i]))
            # count
            cluster_neigh_count += 1
            # list of IDs
            cluster_neigh_id.append(NP_INSTANCES[i])
            # list of distances (l2-norm)
            l2_norm_dist = np.linalg.norm(neigh_vector-NP_EMBEDDINGS[i])
            cluster_neigh_l2.append(l2_norm_dist)
            # list of weights [0,P_MAX_RADIUS]
            cluster_neigh_ws.append((P_MAX_RADIUS - l2_norm_dist)/P_MAX_RADIUS)
            
    # print("ID >>> ",cluster_neigh_id)
    # print("L2 >>> ",cluster_neigh_l2)
    # print("Ws >>> ",cluster_neigh_ws)
    # print("cluster_neigh_count =",cluster_neigh_count)
       
    
    if cluster_neigh_count < P_MIN_NEIHGS: # P_MIN_NEIHGS is min required
        print("(outlier)",cluster_neigh_count,"/",P_MIN_NEIHGS)
    else:
        searchForSubject = neigh[len(P_SUBJECT_TYPE)+1:]
        searchForDatatype = P_DF_TRAIN.dtypes[P_SUBJECT] 
        if "int" in str(searchForDatatype):
            searchForSubject = int(searchForSubject)
        instance = P_DF_TRAIN.loc[P_DF_TRAIN[P_SUBJECT] == searchForSubject]
        if instance.shape[0] == 0:
            print("ERR:","subject",P_SUBJECT_TYPE,searchForSubject,"not found")
            exit(1)
        if instance.shape[0] > 1:
            print("ERR:","subject",P_SUBJECT_TYPE,searchForSubject,"is not unique")
            exit(1)

        instance_count += 1;
        synColNames = ["_source_subject_"]  # Column names
        synColValue = [searchForSubject]    # Column values
        synNewRow = {"_source_subject_":searchForSubject}
        synNewRowIndex = searchForSubject
        # Process each dataset feature/column
        anyNewMode = False
        for dcol in P_DF_TRAIN.columns:
            if instance_count == 1 and "synner" not in P_CONFIG_DATA:
                print(dcol)
            if dcol not in P_SYNNER_COLS:
                dcolMode = "none"
                P_SYNNER_COLS[dcol] = "none"
                anyNewMode = True
            else:
                dcolMode = P_SYNNER_COLS[dcol]
            #print("col",dcol,"mode",dcolMode)

        if instance_count == 1 and anyNewMode:
            print("P_SYNNER_COLS",P_SYNNER_COLS)
            P_CONFIG_DATA["synner"] = P_SYNNER_COLS
            out_file = open("../config/"+args.config, "w")
            json.dump(P_CONFIG_DATA, out_file, indent = 4)
            out_file.close()
            exit(0)
        
        for dcol in P_DF_TRAIN.columns:
            # any mode?
            dcolModeIdx = 0
            dcolMode = P_SYNNER_COLS[dcol][dcolModeIdx]
            # print("dcol",dcol,"dcolMode",dcolMode)
            isReplicated = False
            if dcolMode == "replicate":
                dcolRatio = P_SYNNER_COLS[dcol][dcolModeIdx+1]
                if random.random() <= dcolRatio:
                    instance_col_value = instance[dcol].item()
                    synNewRow[dcol] = instance_col_value
                    isReplicated = True
            if not isReplicated:
                dcolModeIdx = 2
                dcolMode = P_SYNNER_COLS[dcol][dcolModeIdx]
                if dcolMode == "link_random_uuid":
                    instance_col_value = instance[dcol].item()
                    synNewRow[dcol] = synner.gen_random_uuid(instance_col_value, dcol)
                elif dcolMode == "continuous_weighted_average":
                    decimals = P_SYNNER_COLS[dcol][dcolModeIdx+1]
                    # cluster_neigh_id and cluster_neigh_ws = []
                    cluster_neigh_vl = []
                    for neigh_id in cluster_neigh_id:
                        searchForNeigh = neigh_id[len(P_SUBJECT_TYPE)+1:]
                        searchForDatatype = P_DF_TRAIN.dtypes[P_SUBJECT] 
                        if "int" in str(searchForDatatype):
                            searchForNeigh = int(searchForNeigh)
                        neigh_inst = P_DF_TRAIN.loc[P_DF_TRAIN[P_SUBJECT] == searchForNeigh]
                        cluster_neigh_vl.append(neigh_inst[dcol].item())
                    _result = synner.continuous_weighted_average(cluster_neigh_vl, cluster_neigh_ws, decimals=decimals)
                    if np.isnan(_result):
                        synNewRow[dcol] = ""
                    else:
                        synNewRow[dcol] = _result
                        if synNewRow[dcol] < 0 and "positive" in P_SYNNER_COLS[dcol]:
                            synNewRow[dcol] = 0
                elif dcolMode == "categorical_weighted_random_choice":
                    # cluster_neigh_id and cluster_neigh_ws = []
                    cluster_neigh_vl = []
                    for neigh_id in cluster_neigh_id:
                        searchForNeigh = neigh_id[len(P_SUBJECT_TYPE)+1:]
                        searchForDatatype = P_DF_TRAIN.dtypes[P_SUBJECT] 
                        if "int" in str(searchForDatatype):
                            searchForNeigh = int(searchForNeigh)
                        neigh_inst = P_DF_TRAIN.loc[P_DF_TRAIN[P_SUBJECT] == searchForNeigh]
                        # print(">>>>>>>>>>>>>>>>>>>>")
                        # print("loc",P_SUBJECT,"==",searchForNeigh)
                        # print(neigh_inst)
                        # print(neigh_inst[dcol])
                        #print(neigh_inst[dcol].tolist())
                        if len(neigh_inst[dcol].tolist()) == 0:
                            cluster_neigh_vl.append("")
                        else:                     
                            cluster_neigh_vl.append(neigh_inst[dcol].item())
                    # print(">>>>>>>>>>>>>>>>>>>>")
                    # print(cluster_neigh_vl)
                    # print(cluster_neigh_ws)
                    # exit(0)
                    synNewRow[dcol] = synner.categorical_weighted_random_choice(cluster_neigh_vl, cluster_neigh_ws)
                else:
                    print("Invalid synthetic generator mode:",dcolMode)
                    exit(1)

        # print(instance)
        # print("synColNames",synColNames)
        # print("synColValue",synColValue)
        # synSeries = pd.Series(synColValue, name=instance_count, index=synColNames)
        # synSeries = pd.Series(synColValue, index=synColNames)
        # synSeries = synSeries.to_frame().T
        # synSeriesSet.append(synSeries.copy())
        # print(synSeries)
        # print(synSeries)

        # synnerDataFrame = synnerDataFrame._append(synSeries, ignore_index=True)
        # synnerDataFrame = synnerDataFrame._append(synSeries,ignore_index=True)
        # synnerDataFrame = pd.concat([synnerDataFrame, synSeries],ignore_index=True)
        # synnerDataFrame[instance_count] = synSeries
        synNewIndex = len(synnerDataFrame)
        synnerDataFrame.loc[synNewRowIndex] = synNewRow
        for i in range(len(synColNames)):
            # synnerDataFrame[synColNames[i]][synNewIndex] = synColValue[i]
            # synnerDataFrame.loc[synNewIndex, synColNames[i]] = synColValue[i]
            # synnerDataFrame.at[synNewIndex, synColNames[i]] = synColValue[i]
            None
        
    # abort after min-neigh instances processed (for quick test only)
    if False and instance_count >= P_MIN_NEIHGS:
        # print("--------------------------------------------------------------------------------")
        # # print(synSeriesSet)
        # #synnerDataFrame = pd.DataFrame(synSeriesSet,  columns = synColNames)
        # print(synnerDataFrame.head())
        break
 
#######################
##  SYNNER DATASETs  ##
#######################

P_FILE_SYNNER = P_DATA_PATH+"/"+P_DATA_FOLDER+"/"+P_DATA_SET+"-train-SYNNER-"+P_VARIATION+".csv"
print("Synner Dataset = ",P_FILE_SYNNER)
f = open(P_FILE_SYNNER,"w")
synnerDataFrame.to_csv(f,index=False)
f.close()

P_FILE_SYNNER = P_DATA_PATH+"/"+P_DATA_FOLDER+"/"+P_DATA_SET+"-train-SYNNER-"+P_VARIATION+"-link.csv"
print("Synner Dataset = ",P_FILE_SYNNER)
f = open(P_FILE_SYNNER,"w")
synnerDataFrame.to_csv(f)
f.close()
