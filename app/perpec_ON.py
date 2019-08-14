
# coding: utf-8

# In[1]:


# # Run this cell to mount your Google Drive.
# from google.colab import drive
# drive.mount('/content/drive')


# In[2]:



import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing 
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.wrappers.scikit_learn import KerasRegressor

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import tensorflow as tf

from sklearn.feature_selection import VarianceThreshold
# from keras.backend import manual_variable_initialization 
# # manual_variable_initialization(True)


# In[3]:


data_p = pd.read_csv("points.csv",dtype=object,error_bad_lines=False) 
data_p.head()
data_p["id"] = data_p["id"].map(str) +"_"+ data_p["dir"]
data_p.head()
# data_p.dtypes


# In[4]:



data_v = pd.read_csv("values.csv",dtype=object,error_bad_lines=False )
le = preprocessing.LabelEncoder()

data_v['power_state_spec'] = le.fit_transform(data_v['power_state_spec'].astype('str'))

data_v['power_state_value'] = le.fit_transform(data_v['power_state_value'].astype('str'))
data_v["id"] = data_v["id"].map(str) +"_"+data_v["dir"]
data_v.head()


# In[5]:


arr_v = data_v.values
arr_p = data_p.values


# In[6]:


arr_v = arr_v[0:]
# print(arr_v)
arr_p = arr_p[0:]
# print(arr_p)


# In[7]:


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
print(len(ON_list),"ON")
print(len(OFF_list),"OFF")
arr_on_p = np.array(ON_list)
# print(arr_on_p)


# In[8]:


arr_on_p = np.delete(arr_on_p, 3,  axis=1)
arr_on_p_n = arr_on_p[:, 1::2]
arr_on_p_f = np.delete(arr_on_p_n, 1,  axis=1)
# print(len(arr_on_p_f))
# print(len(arr_on_p_f[0]))
# print(arr_on_p_f[0])


# In[9]:


data = arr_on_p_f

df=pd.DataFrame(data=data[0:,0:],index=[i for i in range(data.shape[0])],
                columns=['y'+str(i) for i in range(data.shape[1])])
df.head()
# df.dtypes


# In[10]:


print(df.shape)
# for j in range(10000):
#   var = "y"+str(j+1)
#   df[var].fillna(df[var].mean(), inplace=True)
df_no_miss = df.dropna()
print(df_no_miss.shape)
print(df.shape)
df_no_miss.head()


# In[11]:


arr_p_no = df_no_miss.values
print(len(arr_p_no))


# In[12]:


df1= df_no_miss.rename(index=str, columns={"y0": "id"})
df1.head()


# In[13]:


print(df1.shape)
df2 = data_v
# print(df2.shape)
combine = (pd.merge(df1, df2, how='left', on='id'))
# print(df1.unique)
print(combine.shape)


# In[14]:


combine.head()


# In[15]:


combine.iloc[:,0:10010].head()
k = combine.drop(['s.no','dir','_file_'], axis = 1) 

k.head()


# In[16]:


input_1 = k.iloc[:,0:10009]
# filling the missing values

# print(input_1.iloc[:,10008])
miss = input_1.iloc[:,1:]
miss.head()


# In[17]:


input_1 = k.iloc[:,0:10009]
# filling the missing values

miss = input_1.iloc[:,1:]
miss.head()



  
for column in (miss.iloc[:,10000:]):
  su = 0
  div = 0
  for r in range(miss.shape[0]):
    if (pd.isna(miss[column][r]))== False:
      su = float(miss[column][r])+su

      div = div+1

  fin = float(su/div)

  miss[column].fillna(float(fin),inplace=True)
#########converting ever
# thing into float
miss =miss.astype('float64')
# print(miss.dtypes)
miss.head()


# In[18]:


import pandas as pd
from sklearn.ensemble.forest import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import robust_scale

scaler_min_x = MinMaxScaler()
scaler_min_y = MinMaxScaler()

scaler_norm_x = Normalizer()
scaler_norm_y = Normalizer()

scaler_stan_x = StandardScaler()
scaler_stan_y = StandardScaler()

scalar_qt_x =QuantileTransformer(output_distribution='uniform')
scalar_qt_y =QuantileTransformer(output_distribution='uniform')
       


# In[19]:


rand_na = miss
# print(miss.shape)
input_1_arr = rand_na.values
input_1_arr[:,:]= input_1_arr[:,:].astype('float64')

X = input_1_arr[:,0:10000]*1000
Y = input_1_arr[:,10002:10004]
# print(X.shape)
# print(Y.shape)
# print(Y)
y=np.reshape(Y, (-1,1))

# X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20)
X_train= X
y_train= Y


# ######minmax
scaler_min_x = MinMaxScaler().fit(X_train)
scaler_min_y = MinMaxScaler().fit(y_train)

X_minmax_train = scaler_min_x.transform(X_train)
Y_minmax_train = scaler_min_y.transform(y_train)


# print(X)
# print(Y)
#####standard

scaler_stan_x = StandardScaler().fit(X_train)
scaler_stan_y = StandardScaler().fit(y_train)


X_stan_train = scaler_stan_x.transform(X_train)
Y_stan_train = scaler_stan_y.transform(y_train)

#######normlised
scaler_norm_x = Normalizer().fit(X_train)
scaler_norm_y = Normalizer().fit(y_train)


X_norm_train = scaler_norm_x.transform(X_train)
Y_norm_train = scaler_norm_y.transform(y_train)


# ################qt

scaler_qt_x =  QuantileTransformer(output_distribution='normal').fit(X_train)
scaler_qt_y =  QuantileTransformer(output_distribution='normal').fit(y_train)


X_qt_train = scaler_qt_x.transform(X_train)
Y_qt_train = scaler_qt_y.transform(y_train)


##robust

##robust
print(np.amax(X_train[0,:]))
print(np.amax(y_train[0,:]))

X_train = np.concatenate((X_train, y_train), axis=1)
# print(np.amax(X_train[0,:]))
X_train_t = X_train.transpose()
# y_train_t = y_train.transpose()
# print(X_train,"after")
# print(y_train.shape,"after")

scaler_rob_x = MinMaxScaler().fit(X_train_t)
# scaler_rob_y = RobustScaler().fit(y_train_t)


X_rob_train = scaler_rob_x.transform(X_train_t)
# Y_rob_train = scaler_rob_x.transform(y_train_t)

X_rob_train = X_rob_train.transpose()
# Y_rob_train = Y_rob_train.transpose()

print(X_rob_train.shape)
# print(Y_rob_train.shape)
# print(Y_rob_train)

Y_rob_train = X_rob_train[:,10000:10002]
X_rob_train = X_rob_train[:,0:10000]
# print(Y_rob_train)
# print(X_rob_train)


# In[20]:


from sklearn.decomposition import FactorAnalysis

transformer = FactorAnalysis(n_components=30, random_state=0)
factor_fit = transformer.fit(X_rob_train)
X_new = factor_fit.transform(X_rob_train)
X_new.shape


# In[21]:



import pickle
pickle.dump(factor_fit, open( "./app/MODEL/factor_fit_off.pkl", "wb" ) )


# In[22]:


def baseline_model_30(optimizer='adam'):
    # create model
    model = Sequential()
    model.add(Dense(28, activation='relu', 
                    kernel_initializer = 'he_normal', 
                    input_shape=(30,)))
    model.add(BatchNormalization())
#     model.add(Dropout(0.5))
#     model.add(Dense(30, activation='relu',
#                     kernel_initializer = 'he_normal'))
#       model.add(BatchNormalization())
#     model.add(Dropout(0.5))
    model.add(Dense(12, activation='relu',
                    kernel_initializer = 'he_normal'))
    model.add(BatchNormalization())
#     model.add(Dropout(0.5))
    model.add(Dense(9, activation='relu',
                    kernel_initializer = 'he_normal'))
#     model.add(BatchNormalization())
#     model.add(Dropout(0.5))
    model.add(Dense(2, activation='linear', 
                    kernel_initializer='he_normal'))
    model.compile(loss = 'mse', optimizer=optimizer, metrics=['mae'])
#     model.summary()
    return model


# In[23]:


model = baseline_model_30()

print (model.get_weights())
# estimator = train_data_nn(X_new, Y_rob_train)

print(X_new.shape)
print(y_train.shape)
history = model.fit(X_new,  Y_rob_train, epochs=200, batch_size=5,  verbose=1, validation_split=0.0)


# In[24]:


def visualize_learning_curve(history):
    # list all data in history
    print(history.history.keys())
    # summarize history for accuracy
    plt.plot(history.history['loss'])
#     plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['mean_absolute_error'])
#     plt.plot(history.history['val_mean_absolute_error'])
    plt.title('model mean_absolute_error')
    plt.ylabel('mean_absolute_error')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()


# In[25]:



# print(X_test.shape)
# X_test_t = X_test.transpose()
# print(X_test_t.shape)
# scaler_rob_x = MinMaxScaler().fit(X_test_t)
# X_new_test_t = scaler_rob_x.transform(X_test_t)
# X_new_test = X_new_test_t.transpose()
# print(X_new_test.shape)

# y_test_t = y_test.transpose()
# # scaler_rob_y = RobustScaler().fit(y_test_t)
# Y_new_test_t = scaler_rob_x.transform(y_test_t)
# Y_new_test = Y_new_test_t.transpose()

# X_new_test = factor_fit.transform(X_new_test)
# print(X_new_test.shape)

# visualize_learning_curve(history)


# In[26]:


from sklearn.metrics import mean_squared_error,mean_absolute_error
import math
# from sklearn.metrics import max_error

pred = model.predict((X_new))
# print(pred)

mse = (mean_squared_error(Y_rob_train,pred))

print(mse)
visualize_learning_curve(history)


# In[27]:



X1 = input_1_arr[:,0:10000]*1000
Y1 = input_1_arr[:,10000:10004]

X=X1
Y=Y1[:,2:4]
# print(Y)
Y_new = np.zeros((Y.shape[0],2))
for i in range(len(Y)):
  
    print(Y[i],"y")
  

    X_t= X[i].transpose()

    scaler_rob_x = MinMaxScaler().fit((X_t.reshape(-1, 1)))
                        
    Xi = (scaler_rob_x.transform(X_t.reshape(-1, 1)))

    I = factor_fit.transform(Xi.transpose())

    pred = model.predict(I)

  
    Y_ti =Y[i].transpose()

#     scaler_rob_y = RobustScaler().fit(Y_ti.reshape(-1, 1))
    final_t = scaler_rob_x.inverse_transform(pred.reshape(-1, 1))
                                          

    final = final_t.transpose()
                                          
    print(final[0])

    h = abs(final-Y[i])
#   print(h,"h")
#     o=np.divide(h,Y[i])
#   print(o*100,"percentage") 
  
    Y_new[i]=final[0]


# In[28]:


X1 = X1
Y1 = Y1
# print(Y1)

# print(Y)
from sklearn.metrics import r2_score
print(Y1[0,2], Y_new[0,0])
# print(Y_new[:,0])
g = r2_score(Y1[:,2], Y_new[:,0])  
g1 = r2_score(Y1[:,3], Y_new[:,1]) 
print(g,g1)
Y1[:,2]= Y_new[:,0]
Y1[:,3]= Y_new[:,1]
print(Y1[0,2], Y_new[0,0])


# In[29]:



X1_new = np.concatenate((X1,Y1[:,2:4]),axis=1)
print(X1_new.shape)
Y1_new = Y1[:,0:2]
# print(Y1_new)
# X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X1_new, Y1_new, test_size=0.20)
X_train_c= X1_new
y_train_c = Y1_new


# In[30]:


#standardise


# ######minmax
scaler_min_x = MinMaxScaler().fit(X_train_c)
scaler_min_y = MinMaxScaler().fit(y_train_c)

X_minmax_train = scaler_min_x.transform(X_train_c)
Y_minmax_train = scaler_min_y.transform(y_train_c)


# print(X)
# print(Y)
#####standard

scaler_stan_x = StandardScaler().fit(X_train_c)
scaler_stan_y = StandardScaler().fit(y_train_c)


X_stan_train = scaler_stan_x.transform(X_train_c)
Y_stan_train = scaler_stan_y.transform(y_train_c)

# #######normlised
# scaler_norm_x = Normalizer().fit(X_train_c)
# scaler_norm_y = Normalizer().fit(y_train_c)


# X_norm_train = scaler_norm_x.transform(X_train_c)
# Y_norm_train = scaler_norm_y.transform(y_train_c)


# # ################qt

# scaler_qt_x =  QuantileTransformer(output_distribution='normal').fit(X_train_c)
# scaler_qt_y =  QuantileTransformer(output_distribution='normal').fit(y_train_c)


# X_qt_train = scaler_qt_x.transform(X_train_c)
# Y_qt_train = scaler_qt_y.transform(y_train_c)


##robust
# print(X_train.shape)
# print(y_train.shape)
# X_train_t = X_train.transpose()
# y_train_t = y_train.transpose()
# print(X_train.shape,"after")
# print(y_train.shape,"after")
scaler_rob_x = MinMaxScaler().fit(X_train_c)
scaler_rob_y = MinMaxScaler().fit(y_train_c)


# X_rob_train = scaler_rob_x.transform(X_train_c)
# Y_rob_train = scaler_rob_y.transform(y_train_c)


# In[31]:


import pickle
pickle.dump(scaler_rob_x, open( "./app/MODEL/scaler_rob_x_1_OFF.pkl", "wb" ) )
pickle.dump(scaler_rob_y, open( "./app/MODEL/scaler_rob_y_1_OFF.pkl", "wb" ) )
X_rob_train_c = scaler_rob_x.transform(X_train_c)
Y_rob_train_c = scaler_rob_y.transform(y_train_c)


# In[32]:


#apply PCA on X1_new
transformer = FactorAnalysis(n_components=30, random_state=0)
factor_fit = transformer.fit(X_rob_train_c[:,0:10000])
X_new1 = factor_fit.transform(X_rob_train_c[:,0:10000])
print(X_new1.shape)
X_new1 = np.concatenate((X_new1,X_rob_train_c[:,10000:10002]),axis=1)
print(X_new1.shape)
# print((X_rob_train[:,0:10000].shape))


# In[33]:


import pickle
pickle.dump(factor_fit, open( "./app/MODEL/factor_fit_1_OFF.pkl", "wb" ) )


# In[34]:


def baseline_model_31(optimizer='adam'):
    # create model
    model = Sequential()
    model.add(Dense(30, activation='relu', 
                    kernel_initializer = 'he_normal', 
                    input_shape=(32,)))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
#     model.add(Dense(30, activation='relu',
#                     kernel_initializer = 'he_normal'))
#       model.add(BatchNormalization())
#     model.add(Dropout(0.5))
    model.add(Dense(12, activation='relu',
                    kernel_initializer = 'he_normal'))
    model.add(BatchNormalization())
#     model.add(Dropout(0.5))
    model.add(Dense(9, activation='relu',
                    kernel_initializer = 'he_normal'))
    model.add(BatchNormalization())
#     model.add(Dropout(0.5))
    model.add(Dense(2, activation='linear', 
                    kernel_initializer='he_normal'))
    model.compile(loss = 'mse', optimizer=optimizer, metrics=['mae'])
#     model.summary()
    return model


# In[35]:


model1 = baseline_model_31()

# estimator1 = train_data_nn_1(X_new1, Y_rob_train)

# print(X_new.shape)
# print(y_train.shape)
history = model1.fit(X_new1,  Y_rob_train_c, epochs=400, batch_size=5,  verbose=1, validation_split=0.0)


# In[36]:


from sklearn.metrics import mean_squared_error,mean_absolute_error
import math
# from sklearn.metrics import max_error

pred_c = model1.predict((X_new1))
print(pred_c.shape)

mse = (mean_squared_error(Y_rob_train,pred_c))

print(mse)
visualize_learning_curve(history)


# In[37]:


for i in range(len(y_train_c)):
  
    print(y_train_c[i],"ytest[i]")
#     print(X_train_c[i])
    
    X_c = (scaler_rob_x.transform(X_train_c[i].reshape(1, -1)))

#     print(X_c)
    I = factor_fit.transform(X_c[:,0:10000])
    I = np.concatenate((I,X_c[:,10000:10002]),axis=1)
#   print(I.shape,"I shape")

    pred_c = model1.predict(I)
#     print(pred_c,"pred_c.shape")
  
 

  
    final = scaler_rob_y.inverse_transform(pred_c.reshape(1, -1))
#     print(final,"final")                               
    final[0][0]= np.abs(np.round(final[0][0]))
                                          
    print(final[0],"final")

    h = abs(final[0]-y_train_c[i])


# In[38]:



from keras.backend import manual_variable_initialization 
manual_variable_initialization(True)
model.save ("./app/MODEL/my_model_ON.h5")
model1.save ("./app/MODEL/my_model_1_ON.h5")





# print(model.get_weights())

# print (model1.get_weights())
# model1.save_weights("on_1.h5")


# In[39]:


# from keras.models import load_model
# new_model = load_model('my_model_ON.h5')
# new_model_1 = load_model('my_model_1_ON.h5')


# In[40]:


# pred = new_model_1.predict((X_new1))
# print(pred.shape)

# mse = (mean_squared_error(Y_rob_train,pred))

# print(mse)

