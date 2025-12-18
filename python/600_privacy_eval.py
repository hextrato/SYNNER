import numpy as np
import argparse
import json
import pandas as pd
import pickle
import math
import xgboost as xgb
import statistics as stats
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_curve
import _dataset_ as ds

print("")
print("=================================================================")

# ==========
# Parameters
# ==========

print("")
print("-------------------------")
print("Parameters...")

'''
Example:
python 600_privacy_eval.py --config kaggle-fedesoriano-stroke-prediction-dataset.config.json --kdims 32
'''

print("")
print("-------------------------")
print("Parameters...")

# Initialize parser
parser = argparse.ArgumentParser()
parser.add_argument('--config', help='Config file', required=True)
parser.add_argument('--kdims', type=int, help='number of dimensions', required=False)
parser.add_argument('--variation', help='synthetic variation', required=False)
args = parser.parse_args()

P_CONFIG_FILE = open("../config/"+args.config)
P_CONFIG_DATA = json.load(P_CONFIG_FILE)
P_KDIMS       = args.kdims
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

P_VARIATION = "FINAL-LINKED"
if args.variation:
    P_VARIATION = args.variation

print("Data Path   = ",P_DATA_PATH)
print("Data Folder = ",P_DATA_FOLDER)
print("Data Set    = ",P_DATA_SET)
print("Drop        = ",P_DROP_COLS)
print("Labels      = ",P_LABELS)
print("Variation   = ",P_VARIATION)

print("")
print("-------------------------")
print("Features...")

# feature types
if "types" not in P_CONFIG_DATA:
    P_CONFIG_DATA["types"] = {}
kg_entity_types = P_CONFIG_DATA["types"]

# continuous 
if "continuous-features" not in P_CONFIG_DATA:
    P_CONFIG_DATA["continuous-features"] = []
kg_continuous_features = P_CONFIG_DATA["continuous-features"]

if "continuous-margin" not in P_CONFIG_DATA:
    P_CONFIG_DATA["continuous-margin"] = 0.0
kg_continuous_margin = P_CONFIG_DATA["continuous-margin"]

if "continuous-ranges" not in P_CONFIG_DATA:
    P_CONFIG_DATA["continuous-ranges"] = {}
kg_continuous_ranges = P_CONFIG_DATA["continuous-ranges"]

print("")
print("-------------------------")
print("KDIMS...")

print("P_KDIMS",P_KDIMS)
if not P_KDIMS:
    print("ERROR: KDIMS not defined in config file...")
    exit(1)

print("")
print("-------------------------")
print("Privacy Labels...")

if "privacy-labels" in P_CONFIG_DATA:
    P_PRIVACY_LABELS = P_CONFIG_DATA["privacy-labels"]
else:
    P_PRIVACY_LABELS = {}
if len(P_PRIVACY_LABELS) == 0:
    print("ERROR: No 'privacy-labels' found in config file...")
    exit(1)

# =============== #
# Core Input Data #
# =============== #

v_data_full_path = P_DATA_PATH + "/" + P_DATA_FOLDER + "/" + P_DATA_SET
v_embd_full_path = P_EMBD_PATH + "/" + P_DATA_FOLDER

'''
print("Opening embeddings...",v_embd_full_path)
v_Model_Name = "KRAL_"+P_DATA_SET+"_D"+str(P_KDIMS)
print("v_Model_Name",v_Model_Name)
X_train, X_test, Y_train, Y_test = ds.open_embeddings(P_DATA_FULL_PATH = v_data_full_path, P_EMBD_FULL_PATH = v_embd_full_path, P_EMBD_MODEL_NAME = v_Model_Name, P_SPLIT_LABELS = P_LABELS, P_SUBJECT_COLUMN = P_SUBJECT, P_SUBJECT_TYPE = P_SUBJECT_TYPE)
X_train = X_train.drop(columns="head",axis=1)
X_test  = X_test.drop(columns="head",axis=1)
#print(X_train.head())
#print(Y_train.head())
'''

# for label in P_LABELS:
#     P_DROP_COLS.append(label)
# print("P_LABELS",P_LABELS)
# print("P_DROP_COLS",P_DROP_COLS)

# ============== #
# Privacy Labels #
# ============== #
    
J_EVAL_SCORES = {}

for PRIV_LABEL in P_PRIVACY_LABELS:
    PRIV_LABEL_DEF = P_PRIVACY_LABELS[PRIV_LABEL]
    print("PRIV_LABEL",PRIV_LABEL,"=",PRIV_LABEL_DEF)

    J_EVAL_SCORES[PRIV_LABEL] = {"label": PRIV_LABEL, "label-def": PRIV_LABEL_DEF}

    for XGB_MAX_DEPTH in [10]: # [8,10,12,15,20,25]:

        J_EVAL_SCORES[PRIV_LABEL][XGB_MAX_DEPTH] = {}
    
        #
        # (A) DATASET ONLY
        #

        # XGB_MAX_DEPTH  = 8
        
        if True:
            # print("Opening dataset...",v_data_full_path)
            X_train, X_test, Y_train, Y_test = ds.open_dataset(P_DATA_FULL_PATH = v_data_full_path, P_DROP_COLS = P_DROP_COLS, P_SPLIT_LABELS = [], P_SUBJECT_COLUMN = P_SUBJECT, P_CONTINUOUS = kg_continuous_features)
            X_train = X_train.drop(columns=P_SUBJECT,axis=1)
            # X_test  = X_test.drop(columns=P_SUBJECT,axis=1)
            # Y_train = Y_train.drop(columns=P_SUBJECT,axis=1) 
            # Y_test  = Y_test.drop(columns=P_SUBJECT,axis=1)

            Y_train = X_train[PRIV_LABEL].copy()
            X_train = X_train.drop(columns=PRIV_LABEL,axis=1)
            # set categorical
            for cat_attrib in X_train.columns:
                if cat_attrib not in kg_continuous_features:
                    X_train[cat_attrib] = X_train[cat_attrib].astype('category')
                    #X_tune[cat_attrib] = X_tune[cat_attrib].astype('category')
                    #X_test[cat_attrib] = X_test[cat_attrib].astype('category')
            X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, test_size=0.20, random_state=72)
            X_train, X_tune, Y_train, Y_tune = train_test_split(X_train, Y_train, test_size=0.15, random_state=24)

            print("X_train.columns",X_train.columns)
            print("Y_train.shape",Y_train.shape)

            if len(PRIV_LABEL_DEF) == 2: 

                # ----------- #
                # CATEGORICAL #
                # ----------- #

                J_EVAL_SCORES[PRIV_LABEL][XGB_MAX_DEPTH]["label-value"] = PRIV_LABEL_DEF[0]
                J_EVAL_SCORES[PRIV_LABEL][XGB_MAX_DEPTH]["label-ratio"] = PRIV_LABEL_DEF[1]

                # TARGET LABEL 
                
                target_Y_train = Y_train.copy()
                target_Y_tune  = Y_tune .copy()
                target_Y_test  = Y_test .copy()
                target_Y_train = target_Y_train.apply(lambda x: 1 if x == PRIV_LABEL_DEF[0] else 0)
                target_Y_tune  = target_Y_tune .apply(lambda x: 1 if x == PRIV_LABEL_DEF[0] else 0)
                target_Y_test  = target_Y_test .apply(lambda x: 1 if x == PRIV_LABEL_DEF[0] else 0)
                print("Train labels", target_Y_train.value_counts())
                print("Tune labels", target_Y_tune.value_counts())
                print("Test labels", target_Y_test.value_counts())

                print(X_train.head())
                print(target_Y_train.head())
               
                # XGBOOST: train
                xgBoost = xgb.XGBClassifier(max_depth=XGB_MAX_DEPTH, enable_categorical=True) # (max_iter = int(XGB_MAX_DEPTH)) # , verbose = 1)
                xgBoost.fit( X_train , target_Y_train )

                # XGBOOST: tune
                y_predicted = xgBoost.predict_proba( X_tune )
                y_predicted = y_predicted[:,1]
                y_reference = target_Y_tune
                average_precision_TUNE = average_precision_score(y_reference, y_predicted)
                print("average_precision_TUNE",average_precision_TUNE)

                ttune_pred_auprc = np.round(y_predicted,3)
                P, R, T = precision_recall_curve(y_reference, ttune_pred_auprc)
                for i in range(len(P)):
                    if P[i] <= 0.00001 and R[i] <= 0.00001:
                        P[i] = 1.0
                F1index, = np.where( (2*(P*R)/(P+R)) == max((2*(P*R)/(P+R))))
                F1index = F1index[0]
                if F1index > 0:
                    TH_avg = (T[F1index-1]+T[F1index])/2.0
                else:
                    TH_avg = T[F1index]

                for idx in range(len(y_predicted)):
                    if y_predicted[idx] < TH_avg:
                        y_predicted[idx] = 0
                    else:
                        y_predicted[idx] = 1
                f1_tune = f1_score (target_Y_tune , y_predicted) # (?) .detach() )
                print("max F1 from curve",max((2*(P*R)/(P+R))))
                print("TH_avg",TH_avg)
                print("f1_tune",f1_tune)

                # XGBoost: test
                y_predicted = xgBoost.predict_proba( X_test )
                y_predicted = y_predicted[:,1]
                y_reference = target_Y_test
                average_precision_TEST  = average_precision_score(y_reference, y_predicted)

                print("average_precision_TEST",average_precision_TEST)
                # score["average_precision_TEST"] = average_precision_TEST

                y_predicted = np.round(y_predicted,3)
                P, R, T = precision_recall_curve(y_reference, y_predicted)
                for i in range(len(P)):
                    if P[i] <= 0.00001 and R[i] <= 0.00001:
                        P[i] = 1.0
                #print(P.tolist())
                #print(R.tolist())

                for idx in range(len(y_predicted)):
                    if y_predicted[idx] < TH_avg:
                        y_predicted[idx] = 0
                    else:
                        y_predicted[idx] = 1
                f1_test = f1_score (target_Y_test , y_predicted) # (?) .detach() )
                print("f1_TEST",float(f1_test))

                J_EVAL_SCORES[PRIV_LABEL][XGB_MAX_DEPTH]["f1-score"] = {"DS":float(f1_test)}

                #score["tes_f1"] = float(f1_test)

            elif len(PRIV_LABEL_DEF) == 4:
            
                # ---------- #
                # CONTINUOUS #
                # ---------- #

                J_EVAL_SCORES[PRIV_LABEL][XGB_MAX_DEPTH]["label-stdev"] = PRIV_LABEL_DEF[3]

                # TARGET LABEL 
                
                target_Y_train = Y_train.copy()
                target_Y_tune  = Y_tune .copy()
                target_Y_test  = Y_test .copy()
                
                mean_Y_train = np.nanmean(Y_train)
                mean_Y_tune  = np.nanmean(Y_tune )
                mean_Y_test  = np.nanmean(Y_test )
                
                target_Y_train = target_Y_train.apply(lambda x: mean_Y_train if math.isnan(x) else x)
                target_Y_tune  = target_Y_tune .apply(lambda x: mean_Y_tune  if math.isnan(x) else x)
                target_Y_test  = target_Y_test .apply(lambda x: mean_Y_test  if math.isnan(x) else x)
                #print("Train labels", target_Y_train.value_counts())
                #print("Tune labels", target_Y_tune.value_counts())
                #print("Test labels", target_Y_test.value_counts())
                print("Train labels", len(target_Y_train))
                print("Tune labels", len(target_Y_tune))
                print("Test labels", len(target_Y_test))

                #print(X_train.head())
                #print(target_Y_train.head())

                print(">>>")
                print(">>>",target_Y_train)
                print(">>>")
                # target_Y_stdev = stats.pstdev(target_Y_train)
                target_Y_stdev = np.nanstd(target_Y_train) 
                print("*** target_Y_stdev",target_Y_stdev)
               
                # XGBOOST: train
                xgBoost = xgb.XGBRegressor(max_depth=XGB_MAX_DEPTH, enable_categorical=True) # (max_iter = int(XGB_MAX_DEPTH)) # , verbose = 1)
                xgBoost.fit( X_train , target_Y_train )

                # XGBOOST: tune
                y_predicted = xgBoost.predict( X_tune )
                # y_predicted = y_predicted[:,1]
                # y_reference = target_Y_tune
                y_predicted_MRR = stats.mean ( 1 / ( 1 + abs(y_predicted - target_Y_tune) / (target_Y_stdev * 1.0) ) )
                print ("y_predicted_MRR (tune)",y_predicted_MRR)
                #average_precision_TUNE = average_precision_score(y_reference, y_predicted)
                #print("average_precision_TUNE",average_precision_TUNE)

                '''
                ttune_pred_auprc = np.round(y_predicted,3)
                P, R, T = precision_recall_curve(y_reference, ttune_pred_auprc)
                for i in range(len(P)):
                    if P[i] <= 0.00001 and R[i] <= 0.00001:
                        P[i] = 1.0
                F1index, = np.where( (2*(P*R)/(P+R)) == max((2*(P*R)/(P+R))))
                F1index = F1index[0]
                if F1index > 0:
                    TH_avg = (T[F1index-1]+T[F1index])/2.0
                else:
                    TH_avg = T[F1index]

                for idx in range(len(y_predicted)):
                    if y_predicted[idx] < TH_avg:
                        y_predicted[idx] = 0
                    else:
                        y_predicted[idx] = 1
                f1_tune = f1_score (target_Y_tune , y_predicted) # (?) .detach() )
                print("max F1 from curve",max((2*(P*R)/(P+R))))
                print("TH_avg",TH_avg)
                print("f1_tune",f1_tune)
                '''
                
                # XGBoost: test
                y_predicted = xgBoost.predict( X_test )
                #y_predicted = y_predicted[:,1]
                #y_reference = target_Y_test
                y_predicted_MRR = stats.mean ( 1 / ( 1 + abs(y_predicted - target_Y_test) / (target_Y_stdev * 1.0) ) )
                print ("y_predicted_MRR (test)",y_predicted_MRR)
                # print("average_precision_TEST",average_precision_TEST)
                # score["average_precision_TEST"] = average_precision_TEST

                '''
                y_predicted = np.round(y_predicted,3)
                P, R, T = precision_recall_curve(y_reference, y_predicted)
                for i in range(len(P)):
                    if P[i] <= 0.00001 and R[i] <= 0.00001:
                        P[i] = 1.0
                #print(P.tolist())
                #print(R.tolist())

                for idx in range(len(y_predicted)):
                    if y_predicted[idx] < TH_avg:
                        y_predicted[idx] = 0
                    else:
                        y_predicted[idx] = 1
                f1_test = f1_score (target_Y_test , y_predicted) # (?) .detach() )
                print("f1_TEST",float(f1_test))
                '''

                J_EVAL_SCORES[PRIV_LABEL][XGB_MAX_DEPTH]["mrr-score"] = {"DS":float(y_predicted_MRR)}

                #score["tes_f1"] = float(f1_test)
            
            # exit(0)
                
        #
        # (B) DATASET + SYNTHETIC
        #

        # XGB_MAX_DEPTH  = 12

        if True:
            # print("Opening SYNTHETIC dataset...",v_data_full_path)
            del X_train
            del X_test
            del Y_train
            del Y_test

            X_train      , X_test      , Y_train      , Y_test       = ds.open_dataset(P_DATA_FULL_PATH = v_data_full_path, P_DROP_COLS = P_DROP_COLS, P_SPLIT_LABELS = [], P_SUBJECT_COLUMN = P_SUBJECT, P_CONTINUOUS = kg_continuous_features)
            X_train_synth, X_test_synth, Y_train_synth, Y_test_synth = ds.open_dataset(P_DATA_FULL_PATH = v_data_full_path, P_DROP_COLS = P_DROP_COLS, P_SPLIT_LABELS = [], P_SUBJECT_COLUMN = P_SUBJECT, P_CONTINUOUS = kg_continuous_features, synner_variation=P_VARIATION)
            
            X_merge_train = pd.merge(X_train, X_train_synth, left_on=P_SUBJECT, right_on='LINK_COL', how='outer')
            X_merge_train = X_merge_train.drop(columns=[P_SUBJECT+"_x",P_SUBJECT+"_y","LINK_COL"],axis=1)

            Y_train = X_merge_train[PRIV_LABEL+"_x"].copy()
            X_train = X_merge_train.drop(columns=PRIV_LABEL+"_x",axis=1)
            # set categorical
            for cat_attrib in X_train.columns:
                if cat_attrib not in kg_continuous_features:
                    X_train[cat_attrib] = X_train[cat_attrib].astype('category')
                    #X_tune[cat_attrib] = X_tune[cat_attrib].astype('category')
                    #X_test[cat_attrib] = X_test[cat_attrib].astype('category')
            X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, test_size=0.20, random_state=72)
            X_train, X_tune, Y_train, Y_tune = train_test_split(X_train, Y_train, test_size=0.15, random_state=24)

            print("X_train.columns",X_train.columns)
            print("X_tune .columns",X_train.columns)
            print("X_test .columns",X_train.columns)
            print("Y_train.shape",Y_train.shape)

            # print("====================================")
            # print("X_train     .columns",X_train.columns)
            # print("X_tune      .columns",X_tune .columns)
            # print("X_test      .columns",X_test .columns)
            # print("X_train     .shape",X_train.shape)
            # print("X_tune      .shape",X_tune .shape)
            # print("X_test      .shape",X_test .shape)

            if len(PRIV_LABEL_DEF) == 2: 

                # ----------- #
                # CATEGORICAL #
                # ----------- #

                J_EVAL_SCORES[PRIV_LABEL][XGB_MAX_DEPTH]["label-value"] = PRIV_LABEL_DEF[0]
                J_EVAL_SCORES[PRIV_LABEL][XGB_MAX_DEPTH]["label-ratio"] = PRIV_LABEL_DEF[1]

                # TARGET LABEL 
                
                target_Y_train = Y_train.copy()
                target_Y_tune  = Y_tune .copy()
                target_Y_test  = Y_test .copy()
                target_Y_train = target_Y_train.apply(lambda x: 1 if x == PRIV_LABEL_DEF[0] else 0)
                target_Y_tune  = target_Y_tune .apply(lambda x: 1 if x == PRIV_LABEL_DEF[0] else 0)
                target_Y_test  = target_Y_test .apply(lambda x: 1 if x == PRIV_LABEL_DEF[0] else 0)
                print("Train labels", target_Y_train.value_counts())
                print("Tune labels", target_Y_tune.value_counts())
                print("Test labels", target_Y_test.value_counts())

                print(X_train.head())
                print(target_Y_train.head())
               
                # XGBOOST: train
                xgBoost = xgb.XGBClassifier(max_depth=XGB_MAX_DEPTH, enable_categorical=True) # (max_iter = int(XGB_MAX_DEPTH)) # , verbose = 1)
                xgBoost.fit( X_train , target_Y_train )

                # XGBOOST: tune
                y_predicted = xgBoost.predict_proba( X_tune )
                y_predicted = y_predicted[:,1]
                y_reference = target_Y_tune
                average_precision_TUNE = average_precision_score(y_reference, y_predicted)
                print("average_precision_TUNE",average_precision_TUNE)

                ttune_pred_auprc = np.round(y_predicted,3)
                P, R, T = precision_recall_curve(y_reference, ttune_pred_auprc)
                for i in range(len(P)):
                    if P[i] <= 0.00001 and R[i] <= 0.00001:
                        P[i] = 1.0
                F1index, = np.where( (2*(P*R)/(P+R)) == max((2*(P*R)/(P+R))))
                F1index = F1index[0]
                if F1index > 0:
                    TH_avg = (T[F1index-1]+T[F1index])/2.0
                else:
                    TH_avg = T[F1index]

                for idx in range(len(y_predicted)):
                    if y_predicted[idx] < TH_avg:
                        y_predicted[idx] = 0
                    else:
                        y_predicted[idx] = 1
                f1_tune = f1_score (target_Y_tune , y_predicted) # (?) .detach() )
                print("max F1 from curve",max((2*(P*R)/(P+R))))
                print("TH_avg",TH_avg)
                print("f1_tune",f1_tune)

                # XGBoost: test
                y_predicted = xgBoost.predict_proba( X_test )
                y_predicted = y_predicted[:,1]
                y_reference = target_Y_test
                average_precision_TEST  = average_precision_score(y_reference, y_predicted)

                print("average_precision_TEST",average_precision_TEST)
                # score["average_precision_TEST"] = average_precision_TEST

                y_predicted = np.round(y_predicted,3)
                P, R, T = precision_recall_curve(y_reference, y_predicted)
                for i in range(len(P)):
                    if P[i] <= 0.00001 and R[i] <= 0.00001:
                        P[i] = 1.0
                #print(P.tolist())
                #print(R.tolist())

                for idx in range(len(y_predicted)):
                    if y_predicted[idx] < TH_avg:
                        y_predicted[idx] = 0
                    else:
                        y_predicted[idx] = 1
                f1_test = f1_score (target_Y_test , y_predicted) # (?) .detach() )
                print("f1_TEST",float(f1_test))

                J_EVAL_SCORES[PRIV_LABEL][XGB_MAX_DEPTH]["f1-score"]["DS-Syn"] = float(f1_test)

                #score["tes_f1"] = float(f1_test)

            elif len(PRIV_LABEL_DEF) == 4:
            
                # ---------- #
                # CONTINUOUS #
                # ---------- #

                J_EVAL_SCORES[PRIV_LABEL][XGB_MAX_DEPTH]["label-stdev"] = PRIV_LABEL_DEF[3]

                # TARGET LABEL 
                
                target_Y_train = Y_train.copy()
                target_Y_tune  = Y_tune .copy()
                target_Y_test  = Y_test .copy()
                
                mean_Y_train = np.nanmean(Y_train)
                mean_Y_tune  = np.nanmean(Y_tune )
                mean_Y_test  = np.nanmean(Y_test )
                
                target_Y_train = target_Y_train.apply(lambda x: mean_Y_train if math.isnan(x) else x)
                target_Y_tune  = target_Y_tune .apply(lambda x: mean_Y_tune  if math.isnan(x) else x)
                target_Y_test  = target_Y_test .apply(lambda x: mean_Y_test  if math.isnan(x) else x)
                #print("Train labels", target_Y_train.value_counts())
                #print("Tune labels", target_Y_tune.value_counts())
                #print("Test labels", target_Y_test.value_counts())
                print("Train labels", len(target_Y_train))
                print("Tune labels", len(target_Y_tune))
                print("Test labels", len(target_Y_test))

                #print(X_train.head())
                #print(target_Y_train.head())

                print(">>>")
                print(">>>",target_Y_train)
                print(">>>")
                # target_Y_stdev = stats.pstdev(target_Y_train)
                target_Y_stdev = np.nanstd(target_Y_train) 
                print("*** target_Y_stdev",target_Y_stdev)
               
                # XGBOOST: train
                xgBoost = xgb.XGBRegressor(max_depth=XGB_MAX_DEPTH, enable_categorical=True) # (max_iter = int(XGB_MAX_DEPTH)) # , verbose = 1)
                xgBoost.fit( X_train , target_Y_train )

                # XGBOOST: tune
                y_predicted = xgBoost.predict( X_tune )
                # y_predicted = y_predicted[:,1]
                # y_reference = target_Y_tune
                y_predicted_MRR = stats.mean ( 1 / ( 1 + abs(y_predicted - target_Y_tune) / (target_Y_stdev * 1.0) ) )
                print ("y_predicted_MRR (tune)",y_predicted_MRR)
                #average_precision_TUNE = average_precision_score(y_reference, y_predicted)
                #print("average_precision_TUNE",average_precision_TUNE)

                # XGBoost: test
                y_predicted = xgBoost.predict( X_test )
                #y_predicted = y_predicted[:,1]
                #y_reference = target_Y_test
                y_predicted_MRR = stats.mean ( 1 / ( 1 + abs(y_predicted - target_Y_test) / (target_Y_stdev) ) )
                print ("y_predicted_MRR (test)",y_predicted_MRR)
                # print("average_precision_TEST",average_precision_TEST)
                # score["average_precision_TEST"] = average_precision_TEST

                J_EVAL_SCORES[PRIV_LABEL][XGB_MAX_DEPTH]["mrr-score"]["DS-Syn"] = float(y_predicted_MRR)

            # break # 1 feature only

        #
        # (C) DATASET + EMBEDDINGS
        #

        # XGB_MAX_DEPTH  = 12

        if True:
            v_Model_Name = "KRAL_"+P_DATA_SET+"_D"+str(P_KDIMS)
            # print("Opening SYNTHETIC dataset...",v_data_full_path)
            del X_train
            del X_test
            del Y_train
            del Y_test
            X_train      , X_test      , Y_train      , Y_test       = ds.open_dataset(P_DATA_FULL_PATH = v_data_full_path, P_DROP_COLS = P_DROP_COLS, P_SPLIT_LABELS = [], P_SUBJECT_COLUMN = P_SUBJECT, P_CONTINUOUS = kg_continuous_features)
            # X_train_synth, X_test_synth, Y_train_synth, Y_test_synth = ds.open_dataset(P_DATA_FULL_PATH = v_data_full_path, P_DROP_COLS = P_DROP_COLS, P_SPLIT_LABELS = [], P_SUBJECT_COLUMN = P_SUBJECT, P_CONTINUOUS = kg_continuous_features, synner_variation=P_VARIATION)
            X_train_embed, _ , _ , _                              = ds.open_embeddings(P_DATA_FULL_PATH = v_data_full_path, P_EMBD_FULL_PATH = v_embd_full_path, P_EMBD_MODEL_NAME = v_Model_Name, P_SPLIT_LABELS = P_LABELS, P_SUBJECT_COLUMN = P_SUBJECT, P_SUBJECT_TYPE = P_SUBJECT_TYPE)
            X_train_embed['head'] = X_train_embed['head'].apply(lambda x: x.replace(P_SUBJECT+":","") )
            
            # X_merge_train = pd.merge(X_train, X_train_synth, left_on=P_SUBJECT, right_on='LINK_COL', how='outer')
            
            # print("X_merge_train",X_merge_train)
            # print("X_train_embed",X_train_embed)
            # print("X_merge_train.columns",X_merge_train.columns)
            # print("X_train_embed.columns",X_train_embed.columns)
            # print("X_merge_train.shape",X_merge_train.shape)
            # print("X_train_embed.shape",X_train_embed.shape)

            X_merge_train = pd.merge(X_train, X_train_embed, left_on=P_SUBJECT, right_on='head', how='outer')
            print("X_merge_train.shape",X_merge_train.shape)

            X_merge_train = X_merge_train.drop(columns=[P_SUBJECT,"head"],axis=1)
            print(X_merge_train.columns)

            Y_train = X_merge_train[PRIV_LABEL].copy()
            X_train = X_merge_train.drop(columns=PRIV_LABEL,axis=1)
            # set categorical
            for cat_attrib in X_train.columns:
                if cat_attrib not in kg_continuous_features:
                    X_train[cat_attrib] = X_train[cat_attrib].astype('category')
                    #X_tune[cat_attrib] = X_tune[cat_attrib].astype('category')
                    #X_test[cat_attrib] = X_test[cat_attrib].astype('category')
            X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, test_size=0.20, random_state=72)
            X_train, X_tune, Y_train, Y_tune = train_test_split(X_train, Y_train, test_size=0.15, random_state=24)

            print("X_train.columns",X_train.columns)
            print("X_tune .columns",X_train.columns)
            print("X_test .columns",X_train.columns)
            print("Y_train.shape",Y_train.shape)

            #X_train = X_train.drop(columns="head",axis=1)
            #X_test  = X_test.drop(columns="head",axis=1)

            if len(PRIV_LABEL_DEF) == 2: 

                # ----------- #
                # CATEGORICAL #
                # ----------- #

                J_EVAL_SCORES[PRIV_LABEL][XGB_MAX_DEPTH]["label-value"] = PRIV_LABEL_DEF[0]
                J_EVAL_SCORES[PRIV_LABEL][XGB_MAX_DEPTH]["label-ratio"] = PRIV_LABEL_DEF[1]

                # TARGET LABEL 
                
                target_Y_train = Y_train.copy()
                target_Y_tune  = Y_tune .copy()
                target_Y_test  = Y_test .copy()
                target_Y_train = target_Y_train.apply(lambda x: 1 if x == PRIV_LABEL_DEF[0] else 0)
                target_Y_tune  = target_Y_tune .apply(lambda x: 1 if x == PRIV_LABEL_DEF[0] else 0)
                target_Y_test  = target_Y_test .apply(lambda x: 1 if x == PRIV_LABEL_DEF[0] else 0)
                print("Train labels", target_Y_train.value_counts())
                print("Tune labels", target_Y_tune.value_counts())
                print("Test labels", target_Y_test.value_counts())

                print(X_train.head())
                print(target_Y_train.head())
               
                # XGBOOST: train
                xgBoost = xgb.XGBClassifier(max_depth=XGB_MAX_DEPTH, enable_categorical=True) # (max_iter = int(XGB_MAX_DEPTH)) # , verbose = 1)
                xgBoost.fit( X_train , target_Y_train )

                # XGBOOST: tune
                y_predicted = xgBoost.predict_proba( X_tune )
                y_predicted = y_predicted[:,1]
                y_reference = target_Y_tune
                average_precision_TUNE = average_precision_score(y_reference, y_predicted)
                print("average_precision_TUNE",average_precision_TUNE)

                ttune_pred_auprc = np.round(y_predicted,3)
                P, R, T = precision_recall_curve(y_reference, ttune_pred_auprc)
                for i in range(len(P)):
                    if P[i] <= 0.00001 and R[i] <= 0.00001:
                        P[i] = 1.0
                F1index, = np.where( (2*(P*R)/(P+R)) == max((2*(P*R)/(P+R))))
                F1index = F1index[0]
                if F1index > 0:
                    TH_avg = (T[F1index-1]+T[F1index])/2.0
                else:
                    TH_avg = T[F1index]

                for idx in range(len(y_predicted)):
                    if y_predicted[idx] < TH_avg:
                        y_predicted[idx] = 0
                    else:
                        y_predicted[idx] = 1
                f1_tune = f1_score (target_Y_tune , y_predicted) # (?) .detach() )
                print("max F1 from curve",max((2*(P*R)/(P+R))))
                print("TH_avg",TH_avg)
                print("f1_tune",f1_tune)

                # XGBoost: test
                y_predicted = xgBoost.predict_proba( X_test )
                y_predicted = y_predicted[:,1]
                y_reference = target_Y_test
                average_precision_TEST  = average_precision_score(y_reference, y_predicted)

                print("average_precision_TEST",average_precision_TEST)
                # score["average_precision_TEST"] = average_precision_TEST

                y_predicted = np.round(y_predicted,3)
                P, R, T = precision_recall_curve(y_reference, y_predicted)
                for i in range(len(P)):
                    if P[i] <= 0.00001 and R[i] <= 0.00001:
                        P[i] = 1.0
                #print(P.tolist())
                #print(R.tolist())

                for idx in range(len(y_predicted)):
                    if y_predicted[idx] < TH_avg:
                        y_predicted[idx] = 0
                    else:
                        y_predicted[idx] = 1
                f1_test = f1_score (target_Y_test , y_predicted) # (?) .detach() )
                print("f1_TEST",float(f1_test))

                J_EVAL_SCORES[PRIV_LABEL][XGB_MAX_DEPTH]["f1-score"]["DS-Emb"] = float(f1_test)

                #score["tes_f1"] = float(f1_test)

            elif len(PRIV_LABEL_DEF) == 4:
            
                # ---------- #
                # CONTINUOUS #
                # ---------- #

                J_EVAL_SCORES[PRIV_LABEL][XGB_MAX_DEPTH]["label-stdev"] = PRIV_LABEL_DEF[3]

                # TARGET LABEL 
                
                target_Y_train = Y_train.copy()
                target_Y_tune  = Y_tune .copy()
                target_Y_test  = Y_test .copy()
                
                mean_Y_train = np.nanmean(Y_train)
                mean_Y_tune  = np.nanmean(Y_tune )
                mean_Y_test  = np.nanmean(Y_test )
                
                target_Y_train = target_Y_train.apply(lambda x: mean_Y_train if math.isnan(x) else x)
                target_Y_tune  = target_Y_tune .apply(lambda x: mean_Y_tune  if math.isnan(x) else x)
                target_Y_test  = target_Y_test .apply(lambda x: mean_Y_test  if math.isnan(x) else x)
                #print("Train labels", target_Y_train.value_counts())
                #print("Tune labels", target_Y_tune.value_counts())
                #print("Test labels", target_Y_test.value_counts())
                print("Train labels", len(target_Y_train))
                print("Tune labels", len(target_Y_tune))
                print("Test labels", len(target_Y_test))

                #print(X_train.head())
                #print(target_Y_train.head())

                print(">>>")
                print(">>>",target_Y_train)
                print(">>>")
                # target_Y_stdev = stats.pstdev(target_Y_train)
                target_Y_stdev = np.nanstd(target_Y_train) 
                print("*** target_Y_stdev",target_Y_stdev)
               
                # XGBOOST: train
                xgBoost = xgb.XGBRegressor(max_depth=XGB_MAX_DEPTH, enable_categorical=True) # (max_iter = int(XGB_MAX_DEPTH)) # , verbose = 1)
                xgBoost.fit( X_train , target_Y_train )

                # XGBOOST: tune
                y_predicted = xgBoost.predict( X_tune )
                # y_predicted = y_predicted[:,1]
                # y_reference = target_Y_tune
                y_predicted_MRR = stats.mean ( 1 / ( 1 + abs(y_predicted - target_Y_tune) / (target_Y_stdev * 1.0) ) )
                print ("y_predicted_MRR (tune)",y_predicted_MRR)
                #average_precision_TUNE = average_precision_score(y_reference, y_predicted)
                #print("average_precision_TUNE",average_precision_TUNE)

                # XGBoost: test
                y_predicted = xgBoost.predict( X_test )
                #y_predicted = y_predicted[:,1]
                #y_reference = target_Y_test
                y_predicted_MRR = stats.mean ( 1 / ( 1 + abs(y_predicted - target_Y_test) / (target_Y_stdev * 1.0) ) )
                print ("y_predicted_MRR (test)",y_predicted_MRR)
                # print("average_precision_TEST",average_precision_TEST)
                # score["average_precision_TEST"] = average_precision_TEST

                J_EVAL_SCORES[PRIV_LABEL][XGB_MAX_DEPTH]["mrr-score"]["DS-Emb"] = float(y_predicted_MRR)

            # break # 1 feature only

        #
        # (D) SYNTHETIC + EMBEDDINGS
        #

        # XGB_MAX_DEPTH  = 12

        if False:
            v_Model_Name = "KRAL_"+P_DATA_SET+"_D"+str(P_KDIMS)
            # print("Opening SYNTHETIC dataset...",v_data_full_path)
            # X_train      , X_test      , Y_train      , Y_test       = ds.open_dataset(P_DATA_FULL_PATH = v_data_full_path, P_DROP_COLS = P_DROP_COLS, P_SPLIT_LABELS = [], P_SUBJECT_COLUMN = P_SUBJECT, P_CONTINUOUS = kg_continuous_features)
            X_train_synth, X_test_synth, Y_train_synth, Y_test_synth = ds.open_dataset(P_DATA_FULL_PATH = v_data_full_path, P_DROP_COLS = P_DROP_COLS, P_SPLIT_LABELS = [], P_SUBJECT_COLUMN = P_SUBJECT, P_CONTINUOUS = kg_continuous_features, synner_variation=P_VARIATION)
            X_train_embed, _ , _ , _                              = ds.open_embeddings(P_DATA_FULL_PATH = v_data_full_path, P_EMBD_FULL_PATH = v_embd_full_path, P_EMBD_MODEL_NAME = v_Model_Name, P_SPLIT_LABELS = P_LABELS, P_SUBJECT_COLUMN = P_SUBJECT, P_SUBJECT_TYPE = P_SUBJECT_TYPE)
            X_train_embed['head'] = X_train_embed['head'].apply(lambda x: x.replace(P_SUBJECT+":","") )
            
            # X_merge_train = pd.merge(X_train, X_train_synth, left_on=P_SUBJECT, right_on='LINK_COL', how='outer')
            
            # print("X_merge_train",X_merge_train)
            # print("X_train_embed",X_train_embed)
            # print("X_merge_train.columns",X_merge_train.columns)
            # print("X_train_embed.columns",X_train_embed.columns)
            # print("X_merge_train.shape",X_merge_train.shape)
            # print("X_train_embed.shape",X_train_embed.shape)

            X_merge_train = pd.merge(X_train_synth, X_train_embed, left_on="LINK_COL", right_on='head', how='outer')
            print("X_merge_train.shape",X_merge_train.shape)

            X_merge_train = X_merge_train.drop(columns=[P_SUBJECT,"LINK_COL","head"],axis=1)
            print(X_merge_train.columns)

            Y_train = X_merge_train[PRIV_LABEL].copy()
            X_train = X_merge_train.drop(columns=PRIV_LABEL,axis=1)
            # set categorical
            for cat_attrib in X_train.columns:
                if cat_attrib not in kg_continuous_features:
                    X_train[cat_attrib] = X_train[cat_attrib].astype('category')
                    #X_tune[cat_attrib] = X_tune[cat_attrib].astype('category')
                    #X_test[cat_attrib] = X_test[cat_attrib].astype('category')
            X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, test_size=0.20, random_state=72)
            X_train, X_tune, Y_train, Y_tune = train_test_split(X_train, Y_train, test_size=0.15, random_state=24)

            print("X_train.columns",X_train.columns)
            print("X_tune .columns",X_train.columns)
            print("X_test .columns",X_train.columns)
            print("Y_train.shape",Y_train.shape)

            #X_train = X_train.drop(columns="head",axis=1)
            #X_test  = X_test.drop(columns="head",axis=1)

            if len(PRIV_LABEL_DEF) == 2: 

                # ----------- #
                # CATEGORICAL #
                # ----------- #

                J_EVAL_SCORES[PRIV_LABEL][XGB_MAX_DEPTH]["label-value"] = PRIV_LABEL_DEF[0]
                J_EVAL_SCORES[PRIV_LABEL][XGB_MAX_DEPTH]["label-ratio"] = PRIV_LABEL_DEF[1]

                # TARGET LABEL 
                
                target_Y_train = Y_train.copy()
                target_Y_tune  = Y_tune .copy()
                target_Y_test  = Y_test .copy()
                target_Y_train = target_Y_train.apply(lambda x: 1 if x == PRIV_LABEL_DEF[0] else 0)
                target_Y_tune  = target_Y_tune .apply(lambda x: 1 if x == PRIV_LABEL_DEF[0] else 0)
                target_Y_test  = target_Y_test .apply(lambda x: 1 if x == PRIV_LABEL_DEF[0] else 0)
                print("Train labels", target_Y_train.value_counts())
                print("Tune labels", target_Y_tune.value_counts())
                print("Test labels", target_Y_test.value_counts())

                print(X_train.head())
                print(target_Y_train.head())
               
                # XGBOOST: train
                xgBoost = xgb.XGBClassifier(max_depth=XGB_MAX_DEPTH, enable_categorical=True) # (max_iter = int(XGB_MAX_DEPTH)) # , verbose = 1)
                xgBoost.fit( X_train , target_Y_train )

                # XGBOOST: tune
                y_predicted = xgBoost.predict_proba( X_tune )
                y_predicted = y_predicted[:,1]
                y_reference = target_Y_tune
                average_precision_TUNE = average_precision_score(y_reference, y_predicted)
                print("average_precision_TUNE",average_precision_TUNE)

                ttune_pred_auprc = np.round(y_predicted,3)
                P, R, T = precision_recall_curve(y_reference, ttune_pred_auprc)
                for i in range(len(P)):
                    if P[i] <= 0.00001 and R[i] <= 0.00001:
                        P[i] = 1.0
                F1index, = np.where( (2*(P*R)/(P+R)) == max((2*(P*R)/(P+R))))
                F1index = F1index[0]
                if F1index > 0:
                    TH_avg = (T[F1index-1]+T[F1index])/2.0
                else:
                    TH_avg = T[F1index]

                for idx in range(len(y_predicted)):
                    if y_predicted[idx] < TH_avg:
                        y_predicted[idx] = 0
                    else:
                        y_predicted[idx] = 1
                f1_tune = f1_score (target_Y_tune , y_predicted) # (?) .detach() )
                print("max F1 from curve",max((2*(P*R)/(P+R))))
                print("TH_avg",TH_avg)
                print("f1_tune",f1_tune)

                # XGBoost: test
                y_predicted = xgBoost.predict_proba( X_test )
                y_predicted = y_predicted[:,1]
                y_reference = target_Y_test
                average_precision_TEST  = average_precision_score(y_reference, y_predicted)

                print("average_precision_TEST",average_precision_TEST)
                # score["average_precision_TEST"] = average_precision_TEST

                y_predicted = np.round(y_predicted,3)
                P, R, T = precision_recall_curve(y_reference, y_predicted)
                for i in range(len(P)):
                    if P[i] <= 0.00001 and R[i] <= 0.00001:
                        P[i] = 1.0
                #print(P.tolist())
                #print(R.tolist())

                for idx in range(len(y_predicted)):
                    if y_predicted[idx] < TH_avg:
                        y_predicted[idx] = 0
                    else:
                        y_predicted[idx] = 1
                f1_test = f1_score (target_Y_test , y_predicted) # (?) .detach() )
                print("f1_TEST",float(f1_test))

                J_EVAL_SCORES[PRIV_LABEL][XGB_MAX_DEPTH]["f1-score"]["Syn-Emb"] = float(f1_test)

                #score["tes_f1"] = float(f1_test)

            elif len(PRIV_LABEL_DEF) == 4:
            
                # ---------- #
                # CONTINUOUS #
                # ---------- #

                J_EVAL_SCORES[PRIV_LABEL][XGB_MAX_DEPTH]["label-stdev"] = PRIV_LABEL_DEF[3]

                # TARGET LABEL 
                
                target_Y_train = Y_train.copy()
                target_Y_tune  = Y_tune .copy()
                target_Y_test  = Y_test .copy()
                
                mean_Y_train = np.nanmean(Y_train)
                mean_Y_tune  = np.nanmean(Y_tune )
                mean_Y_test  = np.nanmean(Y_test )
                
                target_Y_train = target_Y_train.apply(lambda x: mean_Y_train if math.isnan(x) else x)
                target_Y_tune  = target_Y_tune .apply(lambda x: mean_Y_tune  if math.isnan(x) else x)
                target_Y_test  = target_Y_test .apply(lambda x: mean_Y_test  if math.isnan(x) else x)
                #print("Train labels", target_Y_train.value_counts())
                #print("Tune labels", target_Y_tune.value_counts())
                #print("Test labels", target_Y_test.value_counts())
                print("Train labels", len(target_Y_train))
                print("Tune labels", len(target_Y_tune))
                print("Test labels", len(target_Y_test))

                #print(X_train.head())
                #print(target_Y_train.head())

                print(">>>")
                print(">>>",target_Y_train)
                print(">>>")
                # target_Y_stdev = stats.pstdev(target_Y_train)
                target_Y_stdev = np.nanstd(target_Y_train) 
                print("*** target_Y_stdev",target_Y_stdev)
               
                # XGBOOST: train
                xgBoost = xgb.XGBRegressor(max_depth=XGB_MAX_DEPTH, enable_categorical=True) # (max_iter = int(XGB_MAX_DEPTH)) # , verbose = 1)
                xgBoost.fit( X_train , target_Y_train )

                # XGBOOST: tune
                y_predicted = xgBoost.predict( X_tune )
                # y_predicted = y_predicted[:,1]
                # y_reference = target_Y_tune
                y_predicted_MRR = stats.mean ( 1 / ( 1 + abs(y_predicted - target_Y_tune) / (target_Y_stdev * 1.0) ) )
                print ("y_predicted_MRR (tune)",y_predicted_MRR)
                #average_precision_TUNE = average_precision_score(y_reference, y_predicted)
                #print("average_precision_TUNE",average_precision_TUNE)

                # XGBoost: test
                y_predicted = xgBoost.predict( X_test )
                #y_predicted = y_predicted[:,1]
                #y_reference = target_Y_test
                y_predicted_MRR = stats.mean ( 1 / ( 1 + abs(y_predicted - target_Y_test) / (target_Y_stdev * 1.0) ) )
                print ("y_predicted_MRR (test)",y_predicted_MRR)
                # print("average_precision_TEST",average_precision_TEST)
                # score["average_precision_TEST"] = average_precision_TEST

                J_EVAL_SCORES[PRIV_LABEL][XGB_MAX_DEPTH]["mrr-score"]["Syn-Emb"] = float(y_predicted_MRR)

            # break # 1 feature only

        #
        # (E) DATASET + SYNTHETIC + EMBEDDINGS
        #

        # XGB_MAX_DEPTH  = 16

        if True:
            v_Model_Name = "KRAL_"+P_DATA_SET+"_D"+str(P_KDIMS)
            # print("Opening SYNTHETIC dataset...",v_data_full_path)
            del X_train
            del X_test
            del Y_train
            del Y_test
            X_train      , X_test      , Y_train      , Y_test       = ds.open_dataset(P_DATA_FULL_PATH = v_data_full_path, P_DROP_COLS = P_DROP_COLS, P_SPLIT_LABELS = [], P_SUBJECT_COLUMN = P_SUBJECT, P_CONTINUOUS = kg_continuous_features)
            X_train_synth, X_test_synth, Y_train_synth, Y_test_synth = ds.open_dataset(P_DATA_FULL_PATH = v_data_full_path, P_DROP_COLS = P_DROP_COLS, P_SPLIT_LABELS = [], P_SUBJECT_COLUMN = P_SUBJECT, P_CONTINUOUS = kg_continuous_features, synner_variation=P_VARIATION)
            X_train_embed, _ , _ , _                              = ds.open_embeddings(P_DATA_FULL_PATH = v_data_full_path, P_EMBD_FULL_PATH = v_embd_full_path, P_EMBD_MODEL_NAME = v_Model_Name, P_SPLIT_LABELS = P_LABELS, P_SUBJECT_COLUMN = P_SUBJECT, P_SUBJECT_TYPE = P_SUBJECT_TYPE)
            X_train_embed['head'] = X_train_embed['head'].apply(lambda x: x.replace(P_SUBJECT+":","") )
            
            X_merge_train = pd.merge(X_train, X_train_synth, left_on=P_SUBJECT, right_on='LINK_COL', how='outer')
            
            # print("X_merge_train",X_merge_train)
            # print("X_train_embed",X_train_embed)
            print("X_merge_train.columns",X_merge_train.columns)
            print("X_train_embed.columns",X_train_embed.columns)
            print("X_merge_train.shape",X_merge_train.shape)
            print("X_train_embed.shape",X_train_embed.shape)

            X_merge_train = pd.merge(X_merge_train, X_train_embed, left_on=P_SUBJECT+'_x', right_on='head', how='outer')
            print("X_merge_train.shape",X_merge_train.shape)

            X_merge_train = X_merge_train.drop(columns=[P_SUBJECT+"_x",P_SUBJECT+"_y","LINK_COL","head"],axis=1)
            print(X_merge_train.columns)

            Y_train = X_merge_train[PRIV_LABEL+"_x"].copy()
            X_train = X_merge_train.drop(columns=PRIV_LABEL+"_x",axis=1)
            # set categorical
            for cat_attrib in X_train.columns:
                if cat_attrib not in kg_continuous_features:
                    X_train[cat_attrib] = X_train[cat_attrib].astype('category')
                    #X_tune[cat_attrib] = X_tune[cat_attrib].astype('category')
                    #X_test[cat_attrib] = X_test[cat_attrib].astype('category')
            X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, test_size=0.20, random_state=72)
            X_train, X_tune, Y_train, Y_tune = train_test_split(X_train, Y_train, test_size=0.15, random_state=24)

            print("X_train.columns",X_train.columns)
            print("X_tune .columns",X_train.columns)
            print("X_test .columns",X_train.columns)
            print("Y_train.shape",Y_train.shape)

            #X_train = X_train.drop(columns="head",axis=1)
            #X_test  = X_test.drop(columns="head",axis=1)

            if len(PRIV_LABEL_DEF) == 2: 

                # ----------- #
                # CATEGORICAL #
                # ----------- #

                J_EVAL_SCORES[PRIV_LABEL][XGB_MAX_DEPTH]["label-value"] = PRIV_LABEL_DEF[0]
                J_EVAL_SCORES[PRIV_LABEL][XGB_MAX_DEPTH]["label-ratio"] = PRIV_LABEL_DEF[1]

                # TARGET LABEL 
                
                target_Y_train = Y_train.copy()
                target_Y_tune  = Y_tune .copy()
                target_Y_test  = Y_test .copy()
                target_Y_train = target_Y_train.apply(lambda x: 1 if x == PRIV_LABEL_DEF[0] else 0)
                target_Y_tune  = target_Y_tune .apply(lambda x: 1 if x == PRIV_LABEL_DEF[0] else 0)
                target_Y_test  = target_Y_test .apply(lambda x: 1 if x == PRIV_LABEL_DEF[0] else 0)
                print("Train labels", target_Y_train.value_counts())
                print("Tune labels", target_Y_tune.value_counts())
                print("Test labels", target_Y_test.value_counts())

                print(X_train.head())
                print(target_Y_train.head())
               
                # XGBOOST: train
                xgBoost = xgb.XGBClassifier(max_depth=XGB_MAX_DEPTH, enable_categorical=True) # (max_iter = int(XGB_MAX_DEPTH)) # , verbose = 1)
                xgBoost.fit( X_train , target_Y_train )

                # XGBOOST: tune
                y_predicted = xgBoost.predict_proba( X_tune )
                y_predicted = y_predicted[:,1]
                y_reference = target_Y_tune
                average_precision_TUNE = average_precision_score(y_reference, y_predicted)
                print("average_precision_TUNE",average_precision_TUNE)

                ttune_pred_auprc = np.round(y_predicted,3)
                P, R, T = precision_recall_curve(y_reference, ttune_pred_auprc)
                for i in range(len(P)):
                    if P[i] <= 0.00001 and R[i] <= 0.00001:
                        P[i] = 1.0
                F1index, = np.where( (2*(P*R)/(P+R)) == max((2*(P*R)/(P+R))))
                F1index = F1index[0]
                if F1index > 0:
                    TH_avg = (T[F1index-1]+T[F1index])/2.0
                else:
                    TH_avg = T[F1index]

                for idx in range(len(y_predicted)):
                    if y_predicted[idx] < TH_avg:
                        y_predicted[idx] = 0
                    else:
                        y_predicted[idx] = 1
                f1_tune = f1_score (target_Y_tune , y_predicted) # (?) .detach() )
                print("max F1 from curve",max((2*(P*R)/(P+R))))
                print("TH_avg",TH_avg)
                print("f1_tune",f1_tune)

                # XGBoost: test
                y_predicted = xgBoost.predict_proba( X_test )
                y_predicted = y_predicted[:,1]
                y_reference = target_Y_test
                average_precision_TEST  = average_precision_score(y_reference, y_predicted)

                print("average_precision_TEST",average_precision_TEST)
                # score["average_precision_TEST"] = average_precision_TEST

                y_predicted = np.round(y_predicted,3)
                P, R, T = precision_recall_curve(y_reference, y_predicted)
                for i in range(len(P)):
                    if P[i] <= 0.00001 and R[i] <= 0.00001:
                        P[i] = 1.0
                #print(P.tolist())
                #print(R.tolist())

                for idx in range(len(y_predicted)):
                    if y_predicted[idx] < TH_avg:
                        y_predicted[idx] = 0
                    else:
                        y_predicted[idx] = 1
                f1_test = f1_score (target_Y_test , y_predicted) # (?) .detach() )
                print("f1_TEST",float(f1_test))

                J_EVAL_SCORES[PRIV_LABEL][XGB_MAX_DEPTH]["f1-score"]["DS-Syn-Emb"] = float(f1_test)

                #score["tes_f1"] = float(f1_test)

            elif len(PRIV_LABEL_DEF) == 4:
            
                # ---------- #
                # CONTINUOUS #
                # ---------- #

                J_EVAL_SCORES[PRIV_LABEL][XGB_MAX_DEPTH]["label-stdev"] = PRIV_LABEL_DEF[3]

                # TARGET LABEL 
                
                target_Y_train = Y_train.copy()
                target_Y_tune  = Y_tune .copy()
                target_Y_test  = Y_test .copy()
                
                mean_Y_train = np.nanmean(Y_train)
                mean_Y_tune  = np.nanmean(Y_tune )
                mean_Y_test  = np.nanmean(Y_test )
                
                target_Y_train = target_Y_train.apply(lambda x: mean_Y_train if math.isnan(x) else x)
                target_Y_tune  = target_Y_tune .apply(lambda x: mean_Y_tune  if math.isnan(x) else x)
                target_Y_test  = target_Y_test .apply(lambda x: mean_Y_test  if math.isnan(x) else x)
                #print("Train labels", target_Y_train.value_counts())
                #print("Tune labels", target_Y_tune.value_counts())
                #print("Test labels", target_Y_test.value_counts())
                print("Train labels", len(target_Y_train))
                print("Tune labels", len(target_Y_tune))
                print("Test labels", len(target_Y_test))

                #print(X_train.head())
                #print(target_Y_train.head())

                print(">>>")
                print(">>>",target_Y_train)
                print(">>>")
                # target_Y_stdev = stats.pstdev(target_Y_train)
                target_Y_stdev = np.nanstd(target_Y_train) 
                print("*** target_Y_stdev",target_Y_stdev)
               
                # XGBOOST: train
                xgBoost = xgb.XGBRegressor(max_depth=XGB_MAX_DEPTH, enable_categorical=True) # (max_iter = int(XGB_MAX_DEPTH)) # , verbose = 1)
                xgBoost.fit( X_train , target_Y_train )

                # XGBOOST: tune
                y_predicted = xgBoost.predict( X_tune )
                # y_predicted = y_predicted[:,1]
                # y_reference = target_Y_tune
                y_predicted_MRR = stats.mean ( 1 / ( 1 + abs(y_predicted - target_Y_tune) / (target_Y_stdev * 1.0) ) )
                print ("y_predicted_MRR (tune)",y_predicted_MRR)
                #average_precision_TUNE = average_precision_score(y_reference, y_predicted)
                #print("average_precision_TUNE",average_precision_TUNE)

                # XGBoost: test
                y_predicted = xgBoost.predict( X_test )
                #y_predicted = y_predicted[:,1]
                #y_reference = target_Y_test
                y_predicted_MRR = stats.mean ( 1 / ( 1 + abs(y_predicted - target_Y_test) / (target_Y_stdev * 1.0) ) )
                print ("y_predicted_MRR (test)",y_predicted_MRR)
                # print("average_precision_TEST",average_precision_TEST)
                # score["average_precision_TEST"] = average_precision_TEST

                J_EVAL_SCORES[PRIV_LABEL][XGB_MAX_DEPTH]["mrr-score"]["DS-Syn-Emb"] = float(y_predicted_MRR)

            # break # 1 feature only

        # exit(0) 
        
        ####
        # OUTPUT FINAL SCORES
        ####
        Output_Score_File_Name = "../privacy" + "/" + P_DATA_FOLDER + "-" + P_DATA_SET + "-" + "D" + str(P_KDIMS) + "-" + P_VARIATION + "."+"json"
        print("")
        print("Saving LABEL scores...",Output_Score_File_Name)
        print("J_EVAL_SCORES",J_EVAL_SCORES)
        with open(Output_Score_File_Name, 'w') as f:
            json.dump(J_EVAL_SCORES, f, ensure_ascii=False, indent=4)
        
print("")
print("The End!")
print("-----------------------------------------------------------------")
