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
    

    
    df_no_miss = df.dropna()
    
    df_no_miss.head()
    arr_p_no = df_no_miss.values
   
    df1= df_no_miss.rename(index=str, columns={"y0": "id"})
    df1.head()
    

    combine = df1
    
    combine.head()
    input_1 = combine.iloc[:,0:10001]
    print(input_1)

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
   
    input_1_arr = rand_na.values
    input_1_arr[:,:]= input_1_arr[:,:].astype('float64')
    X=input_1_arr*1000
    
    X_train= X
    # 
    
   
    X_train_t = X_train.transpose()
    
    scaler_rob_x = MinMaxScaler().fit((X_train_t.reshape(-1, 1)))
                   
    Xi = (scaler_rob_x.transform(X_train_t.reshape(-1, 1)))

    from sklearn.decomposition import FactorAnalysis
    
    if(typ=="on"):
        list_pickle_path = pathtoMODEL+'factor_fit_on.pkl'
        list_unpickle = open(list_pickle_path, 'rb')
        factor_fit_on = pickle.load(list_unpickle)
        # from perceptron_ON import dimen_red 
        I= factor_fit_on.transform(Xi.transpose())
    if(typ=="off"):
        list_pickle_path = pathtoMODEL+'factor_fit_off.pkl'
        list_unpickle = open(list_pickle_path, 'rb')
        factor_fit_off = pickle.load(list_unpickle)
        
        # n_OFF import factor_fit as factor_fit_off
        I = factor_fit_off.transform(Xi.transpose())


    return typ, I, scaler_rob_x, X, fid
