import pandas as pd
import xarray as xr
import numpy as np
import sys
from config import *
from sklearn.ensemble import RandomForestRegressor
from joblib import dump, load
from sklearn.model_selection import PredefinedSplit,RandomizedSearchCV


def inputs():

    msg = "You must specify whether to retrain the model (True) or just load from file (False)"
    try:
        retrain_model = sys.argv[1]
    except:
        sys.exit(msg)
    if retrain_model not in ["True", "False"]:
        sys.exit(msg)

    return retrain_model

def get_numpy_arrays(df):

    #Split the df into two numpy arrays, one for X features and one for Y outputs
    y = 'LST_Day_CMG' #the output column


    df_y = df[[y]]
    df_x = df.drop([y], axis=1)

    return df_x.to_numpy(), df_y.to_numpy().ravel()



def train(df):
    
    print ('Inside ML training function')
    
    #Setup model
    rf = RandomForestRegressor(n_estimators = 10, verbose=1)
    
    #Train the model on training data
    xtrain,ytrain = get_numpy_arrays(df)
    rf.fit(xtrain,ytrain)

    #save trained model to disk
    dump(rf,data_root+'trained_model.joblib')


    #Evaluate model training
    training_score = rf.score(xtrain, ytrain)

    return rf, training_score

def train_with_optimisation(df_train,df_validate):

    """Train and evaluate the model and save to disk"""


    # Bring together train and validate sets 
    X = pd.concat([df_train, df_validate])
    X_Train, Y_Train = get_numpy_arrays(X)

    # Create a list where train data indices are -1 and validation data indices are 0
    idx1 = [1] * len(df_train)
    idx2 = [0] * len(df_validate)
    
    split_index = idx1 + idx2
    pds = PredefinedSplit(test_fold = split_index)


    #Setup random search hyperparam opt
    random_search = {'n_estimators': list(np.linspace(10, 100, 10, dtype = int)),
                     'max_depth': list(np.linspace(10, 1200, 10, dtype = int)) + [None],
                     'max_features': ['auto', 'sqrt','log2', None],
                     'min_samples_leaf': [4, 6, 8, 12],
                     'min_samples_split': [5, 7, 10, 14],
                     }

  
    
    clf = RandomForestRegressor()
    model = RandomizedSearchCV(estimator = clf, 
                               param_distributions = random_search, 
                               n_iter = 2, 
                               cv = pds, 
                               verbose= 5, 
                               random_state= 101, 
                               n_jobs = 2) #njobs=-1 for all processors
    model.fit(X_Train,Y_Train)





    print('completed model train')







def predict(model,df):
    
    """Use the trained model to make predictions on the test df, then save to disk"""
    #Get the test data as arrays
    xtest,ytest = get_numpy_arrays(df)

    #Evaluate how good our model predictions are
    testing_score = model.score(xtest, ytest)

    #...also get the actual predicions themselves
    ypred = model.predict(xtest) #Make some predictions on the test data using the trained model

    #...and the error in these predictions
    relative_error = (ypred - ytest)/ytest


    #Create a df copy
    dfIO=df.copy()
    #Append error and predicitons to test df
    dfIO['predictions'] = ypred
    dfIO['relative_error'] = relative_error
    
    #Add the testing score as an attribute
    dfIO.attrs['testing_score'] = testing_score

    #IO
    dfIO.to_pickle(data_root+"predictions.pkl")


    print ('Max/min relative error:', max(abs(relative_error)), min(abs(relative_error)))

    return testing_score

print ('Starting ML')
retrain_model = inputs()

#Load the data
df = pd.read_pickle(clean_data)

print('loaded the data')

#Split into train/test
df_train = df.query('time <='+training_limit)
df_valid = df.query(training_limit+'< time <= '+validation_limit)
df_test =  df.query(validation_limit+'< time')



if retrain_model == "True":
    print ('Training')
    #model, training_score = train(df_train,df_valid)
    model, training_score = train(df_train)

else:
    print ('Loading trained model')
    s = data_root+'trained_model.joblib'
    model = load(s)
    print ('Loaded model OK')

#Test
print ('Testing')
testing_score = predict(model,df_test)










