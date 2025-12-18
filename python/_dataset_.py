import pandas as pd

# ============
# open_dataset
# ============

def open_dataset(P_DATA_FULL_PATH, P_DROP_COLS = None, P_SPLIT_LABELS = None, P_SUBJECT_COLUMN = None, P_CONTINUOUS = None, synner_variation=None): 
    # v_data_full_path = full path - ('-train.csv','-test.csv')
    
    print ("Opening dataset:", P_DATA_FULL_PATH)
    if synner_variation:
        X_train_file_name = P_DATA_FULL_PATH+'-train-SYNNER-'+synner_variation+'.csv'
        X_train = pd.read_csv(X_train_file_name, keep_default_na=False)
    else:
        X_train_file_name = P_DATA_FULL_PATH+'-train.csv'
        X_train = pd.read_csv(X_train_file_name, keep_default_na=False)
    X_test_file_name = P_DATA_FULL_PATH+'-test.csv'
    X_test  = pd.read_csv(X_test_file_name, keep_default_na=False)
    Y_train = None
    Y_test  = None
    print("X_train_file_name",X_train_file_name)
    print("X_test_file_name",X_test_file_name)

    for col in P_CONTINUOUS:
        if col in X_train:
            X_train[col] = pd.to_numeric(X_train[col], errors='coerce') # X_train[col].astype(float)
        if col in X_test:
            X_test[col]  = pd.to_numeric(X_test[col], errors='coerce') # X_test[col].astype(float)

    if P_DROP_COLS:
        X_train = X_train.drop(columns=P_DROP_COLS,axis=1) 
        X_test  = X_test.drop(columns=P_DROP_COLS,axis=1)

    if P_SUBJECT_COLUMN:
        if P_SPLIT_LABELS: 
            v_split_columns = [P_SUBJECT_COLUMN]
            for lab in P_SPLIT_LABELS:
                v_split_columns.append(lab)
            Y_train = X_train[v_split_columns]
            Y_test  = X_test[v_split_columns]
            X_train = X_train.drop(columns=P_SPLIT_LABELS,axis=1) 
            X_test  = X_test.drop(columns=P_SPLIT_LABELS,axis=1)

    return X_train,X_test,Y_train,Y_test

def open_embeddings(P_DATA_FULL_PATH, P_EMBD_FULL_PATH, P_EMBD_MODEL_NAME, P_SPLIT_LABELS = None, P_SUBJECT_COLUMN = None, P_SUBJECT_TYPE = None): 
    # v_data_full_path = full path - ('-train.csv','-test.csv')

    if not P_SUBJECT_COLUMN:
        raise Exception('Invalid SUBJECT column')
    if not P_SUBJECT_TYPE:
        raise Exception('Invalid SUBJECT type')
    
    print ("Opening embeddings:", P_DATA_FULL_PATH)
    
    X_train = pd.read_csv(P_DATA_FULL_PATH+'-train.csv', keep_default_na=False)
    X_test  = pd.read_csv(P_DATA_FULL_PATH+'-test.csv', keep_default_na=False)
    E_train = pd.read_csv(P_EMBD_FULL_PATH+"/"+P_EMBD_MODEL_NAME+'-MODEL-head-train.csv', keep_default_na=False)
    E_test  = pd.read_csv(P_EMBD_FULL_PATH+"/"+P_EMBD_MODEL_NAME+'-MODEL-head-test.csv', keep_default_na=False)
    Y_train = None
    Y_test  = None

    print("Converting to str...")
    print(X_test[P_SUBJECT_COLUMN])
    X_train[P_SUBJECT_COLUMN] = X_train[P_SUBJECT_COLUMN].astype(str)
    X_test[P_SUBJECT_COLUMN] = X_test[P_SUBJECT_COLUMN].astype(str)
    print(X_train[P_SUBJECT_COLUMN].dtype,X_test[P_SUBJECT_COLUMN].dtype)

    for ix in X_train.index:
        X_train.loc[X_train.index == ix, P_SUBJECT_COLUMN] = P_SUBJECT_TYPE + ":" + X_train[P_SUBJECT_COLUMN][ix]
        # print(P_SUBJECT_TYPE,X_train[P_SUBJECT_COLUMN][ix])
    for ix in X_test.index:
        X_test.loc[X_test.index == ix, P_SUBJECT_COLUMN] = P_SUBJECT_TYPE + ":" + X_test[P_SUBJECT_COLUMN][ix]
        # print(P_SUBJECT_TYPE,X_test[P_SUBJECT_COLUMN][ix])

    if P_SPLIT_LABELS: 
        v_split_columns = [P_SUBJECT_COLUMN]
        for lab in P_SPLIT_LABELS:
            v_split_columns.append(lab)
        Y_train = X_train[v_split_columns]
        Y_test  = X_test[v_split_columns]

        #print ("####################################")
        #print (E_train.head())
        #print (Y_train.head())
        
        print("P_SUBJECT_COLUMN",P_SUBJECT_COLUMN)
        E_train = E_train.merge(Y_train, left_on="head", right_on=P_SUBJECT_COLUMN)
        E_test = E_test.merge(Y_test, left_on="head", right_on=P_SUBJECT_COLUMN)

        #print ("####################################")
        #print (E_train.head())
        #print (Y_train.head())
        #exit(0)

        Y_train = E_train[v_split_columns]
        Y_test  = E_test[v_split_columns]

        E_train = E_train.drop(columns=v_split_columns,axis=1) 
        E_test  = E_test.drop(columns=v_split_columns,axis=1)

    X_train = E_train
    X_test  = E_test

    for col in X_train.columns:
        if col != "head":
            X_train[col] = pd.to_numeric(X_train[col], errors='coerce') # X_train[col].astype(float)
            X_test[col]  = pd.to_numeric(X_test[col], errors='coerce') # X_test[col].astype(float)
                
    return X_train,X_test,Y_train,Y_test

