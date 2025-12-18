import numpy as np
import argparse
import json
import pandas as pd
import pickle
#import os
#import math
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_curve
import _dataset_ as ds

print("")
print("=================================================================")

'''
Example:
python 500_synner_eval.py --config kaggle-fedesoriano-stroke-prediction-dataset.config.json     --variation S016-032 > log/500_synner_eval-kaggle-fedesoriano-stroke-prediction-dataset-S016-032.log
python 500_synner_eval.py --config kaggle-fedesoriano-stroke-prediction-dataset.config.json     --variation S032-064 > log/500_synner_eval-kaggle-fedesoriano-stroke-prediction-dataset-S032-064.log
python 500_synner_eval.py --config kaggle-fedesoriano-stroke-prediction-dataset.config.json     --variation S064-128 > log/500_synner_eval-kaggle-fedesoriano-stroke-prediction-dataset-S064-128.log
python 500_synner_eval.py --config kaggle-fedesoriano-stroke-prediction-dataset.config.json     --variation S128-256 > log/500_synner_eval-kaggle-fedesoriano-stroke-prediction-dataset-S128-256.log
'''

print("")
print("-------------------------")
print("Parameters...")

# Initialize parser
parser = argparse.ArgumentParser()
parser.add_argument('--config', help='Config file', required=True)
parser.add_argument('--variation', help='Variation name', required=True)
args = parser.parse_args()

P_CONFIG_FILE = open("../config/"+args.config)
P_CONFIG_DATA = json.load(P_CONFIG_FILE)
P_VARIATION   = args.variation
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

# ==========
# Input Data
# ==========

v_data_full_path = P_DATA_PATH + "/" + P_DATA_FOLDER + "/" + P_DATA_SET
v_embd_full_path = P_EMBD_PATH + "/" + P_DATA_FOLDER

# print("P_KDIMS",P_KDIMS)
# if P_KDIMS:
#     v_Model_Name = "KRAL_"+P_DATA_SET+"_D"+str(P_KDIMS)
#     print("v_Model_Name",v_Model_Name)
#     X_train, X_test, Y_train, Y_test = ds.open_embeddings(P_DATA_FULL_PATH = v_data_full_path, P_EMBD_FULL_PATH = v_embd_full_path, P_EMBD_MODEL_NAME = v_Model_Name, P_SPLIT_LABELS = P_LABELS, P_SUBJECT_COLUMN = P_SUBJECT, P_SUBJECT_TYPE = P_SUBJECT_TYPE)
#     X_train = X_train.drop(columns="head",axis=1)
#     X_test  = X_test.drop(columns="head",axis=1)
#     #print(X_train.head())
#     #print(Y_train.head())
# else:
X_train, X_test, Y_train, Y_test = ds.open_dataset(P_DATA_FULL_PATH = v_data_full_path, P_DROP_COLS = P_DROP_COLS, P_SPLIT_LABELS = P_LABELS, P_SUBJECT_COLUMN = P_SUBJECT, P_CONTINUOUS = kg_continuous_features, synner_variation=P_VARIATION)
X_train = X_train.drop(columns=P_SUBJECT,axis=1)
X_test  = X_test.drop(columns=P_SUBJECT,axis=1)

Y_train = Y_train.drop(columns=P_SUBJECT,axis=1) 
Y_test  = Y_test.drop(columns=P_SUBJECT,axis=1)
#print(X_train.head())
#print(Y_train.head())
# exit(0)
X_train, X_tune, Y_train, Y_tune = train_test_split(X_train, Y_train, test_size=0.2) # , random_state=1972)

print("Train")
print(X_train.shape, X_train.columns)
print(Y_train.shape, Y_train.columns)
      
print("Tune")
print(X_tune.shape, X_tune.columns)
print(Y_tune.shape, Y_tune.columns)
print(X_tune.head())
print(Y_tune.head())
      
print("Test")
print(X_test.shape, X_test.columns)
print(Y_test.shape, Y_test.columns)
print(X_test.head())
print(Y_test.head())

#if P_KDIMS:
#    None
#else:
for cat_attrib in X_train.columns:
    if cat_attrib not in kg_continuous_features:
        X_train[cat_attrib] = X_train[cat_attrib].astype('category')
        X_tune[cat_attrib] = X_tune[cat_attrib].astype('category')
        X_test[cat_attrib] = X_test[cat_attrib].astype('category')

for label_index in range(len(P_LABELS)):
    label = P_LABELS[label_index]
    target_label_values = P_TL_VALUES[label_index]
    if not (type(target_label_values) == list):
        target_label_values = [target_label_values]
    for target_label_value in target_label_values:
        print("")
        print("-----------------------------------")
        print("Label:",label)
        print("Target value:",target_label_value)
        print("Train labels", Y_train[label].value_counts())
        target_Y_train = Y_train[label].copy()
        target_Y_tune = Y_tune[label].copy()
        target_Y_test = Y_test[label].copy()
        target_Y_train = target_Y_train.apply(lambda x: 1 if x == target_label_value else 0)
        target_Y_tune = target_Y_tune.apply(lambda x: 1 if x == target_label_value else 0)
        target_Y_test = target_Y_test.apply(lambda x: 1 if x == target_label_value else 0)
        print("Train labels", target_Y_train.value_counts())
        print("Tune labels", target_Y_tune.value_counts())
        print("Test labels", target_Y_test.value_counts())

        s_target_label_value = target_label_value
        if not (type(s_target_label_value) == str):
            s_target_label_value = str(s_target_label_value)
        #if P_KDIMS:
        #    Output_Model_File_Name = "../xgboost" + "/zynner-" + P_DATA_FOLDER + "-" + P_DATA_SET + "-" + label + "-" + s_target_label_value + "-D" + str(P_KDIMS) + "."+"xgb"
        #    Output_Score_File_Name = "../xgboost" + "/zynner-" + P_DATA_FOLDER + "-" + P_DATA_SET + "-" + label + "-" + s_target_label_value + "-D" + str(P_KDIMS) + "."+"json"
        #else:
        Output_Model_File_Name = "../xgboost" + "/zynner-" + P_DATA_FOLDER + "-" + P_DATA_SET + "-" + P_VARIATION + "-" + label + "-" + s_target_label_value +"."+"xgb"
        Output_Score_File_Name = "../xgboost" + "/zynner-" + P_DATA_FOLDER + "-" + P_DATA_SET + "-" + P_VARIATION + "-" + label + "-" + s_target_label_value +"."+"json"
        max_average_precision_TRAIN = 0
        max_average_precision_TUNE = 0
        _features = list(X_train.columns)
        score = {"feature_coefs":{}}
        score["max_depth"] = []
        score["average_precision_TRAIN"] = []
        score["average_precision_TUNE"] = []
        score["tun_f1"] = []
        score["tun_th"] = []
        score["best_tun_f1"] = 0.0
        score["best_tun_th"] = 0.0
        score["best_max_depth"] = 0

        # for XGB_MAX_DEPTH in [4,6,8,10,12,16,18,20]: #[6]: #[4,6,8,10]:
        for XGB_MAX_DEPTH in [5,9,13,17,21]: #[3,5]: # [5,9,13,17,21]: #[3] [4,6,8,10]:
            print("")
            print("-----------------------------------")
            print(">>> MAX_DEPTH",XGB_MAX_DEPTH)
            xgBoost = xgb.XGBClassifier(max_depth=XGB_MAX_DEPTH, enable_categorical=True) # (max_iter = int(XGB_MAX_DEPTH)) # , verbose = 1)
            xgBoost.fit( X_train , target_Y_train )

            # print("xgBoost.classes_",xgBoost.classes_)
            # for label_class in xgBoost.classes_:
            # xgBoostClass1Index = np.where(xgBoost.classes_ == 1)[0]
            # print(label_class, xgBoostClass1Index)

            y_predicted = xgBoost.predict_proba( X_train )
            
            y_predicted = y_predicted[:,1]
            y_reference = target_Y_train
            average_precision_TRAIN  = average_precision_score(y_reference, y_predicted)

            print("average_precision_TRAIN",average_precision_TRAIN)
            
            y_predicted = xgBoost.predict_proba( X_tune )
            y_predicted = y_predicted[:,1]
            y_reference = target_Y_tune
            average_precision_TUNE = average_precision_score(y_reference, y_predicted)

            print("average_precision_TUNE",average_precision_TUNE)
            
            #if not math.isnan(average_precision_TRAIN) and not math.isnan(average_precision_TUNE):
            #
            #    print("",'Average precision-recall score (DEV,VAL): {0:0.4f}'.format(average_precision_TRAIN),'{0:0.4f}'.format(average_precision_TUNE))
            #    score["max_depth"].append(XGB_MAX_DEPTH) # .append(int(XGB_MAX_DEPTH))
            #    score["average_precision_TRAIN"].append(average_precision_TRAIN)
            #    score["average_precision_TUNE"].append(average_precision_TUNE)

            # P, R, T = precision_recall_curve(y_reference, y_predicted)
            # for i in range(len(P)):
            #     if P[i] <= 0.00001 and R[i] <= 0.00001:
            #         P[i] = 1.0

            ttune_pred_auprc = np.round(y_predicted,3)
            P, R, T = precision_recall_curve(y_reference, ttune_pred_auprc)
            for i in range(len(P)):
                if P[i] <= 0.00001 and R[i] <= 0.00001:
                    P[i] = 1.0
            #print(P.tolist())
            #print(R.tolist())

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
            
            score["average_precision_TRAIN"].append(float(average_precision_TRAIN))
            score["average_precision_TUNE"].append(float(average_precision_TUNE))
            score["tun_f1"].append(float(f1_tune))
            score["tun_th"].append(float(TH_avg))

            if float(f1_tune) >= score["best_tun_f1"]:
                score["best_tun_f1"] = float(f1_tune)
                score["best_tun_th"] = float(TH_avg)
                score["best_max_depth"] = XGB_MAX_DEPTH
                print("Saving best model...")
                xgBoost.best_tun_f1 = float(f1_tune)
                xgBoost.best_tun_th = float(TH_avg)
                xgBoost.best_max_depth = float(XGB_MAX_DEPTH)
                with open(Output_Model_File_Name, 'wb') as f:
                    pickle.dump(xgBoost,f)
            
        print("Evaluatin best model for test set...")
        with open(Output_Model_File_Name, 'rb') as f:
            xgBoost = pickle.load(f)
        
        y_predicted = xgBoost.predict_proba( X_test )
        y_predicted = y_predicted[:,1]
        y_reference = target_Y_test
        average_precision_TEST  = average_precision_score(y_reference, y_predicted)

        print("average_precision_TEST",average_precision_TEST)
        score["average_precision_TEST"] = average_precision_TEST

        y_predicted = np.round(y_predicted,3)
        P, R, T = precision_recall_curve(y_reference, y_predicted)
        for i in range(len(P)):
            if P[i] <= 0.00001 and R[i] <= 0.00001:
                P[i] = 1.0
        #print(P.tolist())
        #print(R.tolist())

        for idx in range(len(y_predicted)):
            if y_predicted[idx] < xgBoost.best_tun_th:
                y_predicted[idx] = 0
            else:
                y_predicted[idx] = 1
        f1_test = f1_score (target_Y_test , y_predicted) # (?) .detach() )
        score["tes_f1"] = float(f1_test)

        print("")
        print("Saving final scores...",Output_Score_File_Name)
        print(score)
        with open(Output_Score_File_Name, 'w') as f:
            json.dump(score, f, ensure_ascii=False, indent=4)
    
print("")
print("The End!")
print("-----------------------------------------------------------------")
