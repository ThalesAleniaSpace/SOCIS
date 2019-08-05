def data_pre( pathtocsv, pathtoMODEL):
    import pandas as pd 
    import numpy as np
    import pickle
    from sklearn import preprocessing 
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    import tensorflow as tf
    data_p = pd.read_csv(pathtocsv,dtype=object,error_bad_lines=False) 
    data_p.head()
    # data_p["id"] = data_p["id"]
    data_p.head()
    arr_p = data_p.values
    arr_p = arr_p[0:]
    typ ="on"
    ON_list =[]
    OFF_list = []
    for i in range(len(arr_p)):
        s = arr_p[i][1]
        s = str(s)
        
    #     print(type(st))
        if s.find("N") == -1:
            OFF_list.append(arr_p[i])
        
        else:
            ON_list.append(arr_p[i])
    # calculating for ON
   
   

    arr_on_p = np.array(ON_list)
    if len(ON_list) == 0:
        typ="off"
        arr_on_p = np.array(OFF_list)
    # arr_on_p = np.delete(arr_on_p, 1,  axis=1)
    arr_on_p_n = arr_on_p[:, 1::2]
    arr_on_p_f = arr_on_p_n
    data = arr_on_p_f

    df=pd.DataFrame(data=data[0:,0:],index=[i for i in range(data.shape[0])],
                    columns=['y'+str(i) for i in range(data.shape[1])])
    df.head()
    # print(df.shape)
    # for j in range(10000):
    #   var = "y"+str(j+1)

    #   df[var].fillna(df[var].mean(), inplace=True)
    df_no_miss = df.dropna()
    # print(df_no_miss.shape)
    # print(df.shape)
    df_no_miss.head()
    arr_p_no = df_no_miss.values
    # print(len(arr_p_no))
    df1= df_no_miss.rename(index=str, columns={"y0": "id"})
    df1.head()
    # print(df1.shape)

    combine = df1
    # print(df1.unique)
    # print(combine.shape)
    combine.head()
    input_1 = combine.iloc[:,0:10001]
    print(input_1)
# filling the missing values

# print(input_1.iloc[:,10008])
    k= input_1.values
    print(type(k),"k")
    print(k)
    fid = k[0][0]
    print(fid,"fid")
    miss = input_1.iloc[:,1:]
    miss.head()
    miss =miss.astype('float64')
    miss.head()
    import pandas as pd

    from sklearn.preprocessing import RobustScaler, MinMaxScaler
    from sklearn.feature_selection import SelectFromModel
    rand_na = miss
    # print(miss.shape)
    input_1_arr = rand_na.values
    input_1_arr[:,:]= input_1_arr[:,:].astype('float64')
    X=input_1_arr*1000
    # X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20)
    X_train= X
    # 
    print(X_train.shape,"PPPP")
    # print(y_train.shape)
    X_train_t = X_train.transpose()
    # y_train_t = y_train.transpose()
    scaler_rob_x = MinMaxScaler().fit((X_train_t.reshape(-1, 1)))
#   print(X_test_t.reshape(-1, 1))                            
    Xi = (scaler_rob_x.transform(X_train_t.reshape(-1, 1)))
#   print(X.shape,"X shape")\
    from sklearn.decomposition import FactorAnalysis
    # I = factor_fit.transform(Xi.transpose())
    if(typ=="on"):
        list_pickle_path = pathtoMODEL+'factor_fit_on.pkl'
        list_unpickle = open(list_pickle_path, 'rb')
        factor_fit_on = pickle.load(list_unpickle)
        # from perceptron_ON import dimen_red 
        I= factor_fit_on.transform(Xi.transpose())
    if(typ=="off"):
        list_pickle_path = pathtoMODEL+'factor_fit_on.pkl'
        list_unpickle = open(list_pickle_path, 'rb')
        factor_fit_off = pickle.load(list_unpickle)
        # from perceptro
        # n_OFF import factor_fit as factor_fit_off
        I = factor_fit_off.transform(Xi.transpose())

# Y_rob_train = scaler_rob_y.transform(y_train_t)

#     X_rob_train = X_rob_train.transpose()
# # Y_rob_train = Y_rob_train.transpose()

#     print(X_rob_train.shape)
#     from sklearn.decomposition import FactorAnalysis

#     transformer = FactorAnalysis(n_components=30, random_state=0)
#     factor_fit = transformer.fit(X_rob_train)
#     X_new = factor_fit.transform(X_rob_train)
#     X_new.shape
    return typ, I, scaler_rob_x, X, fid
# print(Y_rob_train.shape)


    
    # print(arr_on_p)
