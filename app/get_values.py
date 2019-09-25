


from keras.models import load_model, model_from_json
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
def get_val(X_new,typ, scaler_rob_x,X,pathtoMODEL):
    from keras.models import load_model, model_from_json
    from keras.models import Sequential
    from keras.layers import Dense, Dropout, BatchNormalization
    import pickle
   
    if(typ=="on"):
        import pickle
        
        new_model = load_model(pathtoMODEL+'my_model_ON.h5')
        
        new_model_1 = load_model(pathtoMODEL+'my_model_1_ON.h5')
        print("--------------------------------------------------------------------------------")
        print (new_model_1.get_weights())
        print("--------------------------------------------------------------------------------")
        list_pickle_path = pathtoMODEL+'scaler_rob_x_1_ON.pkl'
        list_unpickle = open(list_pickle_path, 'rb')
        scaler_rob_x_1_ON = pickle.load(list_unpickle)
        list_pickle_path = pathtoMODEL+'scaler_rob_y_1_ON.pkl'
        list_unpickle = open(list_pickle_path, 'rb')
        scaler_rob_y_1_ON = pickle.load(list_unpickle)

        # from perceptron_ON import scaler_rob_x as scaler_rob_y_on
    else:
        # from perceptron_OFF import scaler_rob_x as scaler_rob_y_off
        list_pickle_path = pathtoMODEL+'scaler_rob_x_1_OFF.pkl'
        list_unpickle = open(list_pickle_path, 'rb')
        scaler_rob_x_1_OFF = pickle.load(list_unpickle)
        new_model = load_model(pathtoMODEL+'my_model_OFF.h5')
        new_model_1 = load_model(pathtoMODEL+'my_model_1_OFF.h5')
        list_pickle_path = pathtoMODEL+'scaler_rob_y_1_OFF.pkl'
        list_unpickle = open(list_pickle_path, 'rb')
        scaler_rob_y_1_OFF = pickle.load(list_unpickle)

        print("--------------------------------------------------------------------------------")
        print (new_model.get_weights())
        print("--------------------------------------------------------------------------------")
        print("--------------------------------------------------------------------------------")
        print (new_model_1.get_weights())
        print("--------------------------------------------------------------------------------")

    pred = new_model.predict(X_new)
    final_t = scaler_rob_x.inverse_transform(pred.reshape(-1, 1))
    final = final_t.transpose()
                                          
    # print(final[0],"final")
    import numpy as np
    import pickle
    X =  np.concatenate((X[0],final[0]), axis = None)
    # print(X,"XXX")

    if(typ=="on"):

        X_c = (scaler_rob_x_1_ON.transform(X.reshape(1, -1)))
        # print(X_c,"X_C")
        list_pickle_path = pathtoMODEL+ 'factor_fit_1_ON.pkl'
        list_unpickle = open(list_pickle_path, 'rb')
        factor_fit_ON_1 = pickle.load(list_unpickle)


        I = factor_fit_ON_1.transform(X_c[:,0:10000])
        I = np.concatenate((I,X_c[:,10000:10002]),axis=1)
    

        pred_c = new_model_1.predict(I)
       
    
    

    
        final1 = scaler_rob_y_1_ON.inverse_transform(pred_c.reshape(1, -1))
        

    else:
        X_c = (scaler_rob_x_1_OFF.transform(X.reshape(1, -1)))
        list_pickle_path = pathtoMODEL+'factor_fit_1_OFF.pkl'
        list_unpickle = open(list_pickle_path, 'rb')
        factor_fit_OFF_1 = pickle.load(list_unpickle)


        I = factor_fit_OFF_1.transform(X_c[:,0:10000])
        I = np.concatenate((I,X_c[:,10000:10002]),axis=1)
   

        pred_c = new_model_1.predict(I)
   
    
    

    
        final1 = scaler_rob_y_1_OFF.inverse_transform(pred_c.reshape(1, -1))
        
    import numpy as np                                     
    final1[0][0]= np.abs(np.round(final1[0][0]))
                                          
    
    
    FINAL =  np.concatenate((final1[0],final[0]), axis = None)
    
    return FINAL


    