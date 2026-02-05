from sklearn.model_selection import StratifiedShuffleSplit  ## for train test split we need to make a strata taki vo split karpaye  

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler,OneHotEncoder    ## standardscaler is used for make data in a speciffic ranges 
# one hot encoder is used for working on catogorical data 
from sklearn.impute import SimpleImputer
# simple imputer missinng value ko median , mean wagera se replace karega 
# strategy =" "
from sklearn.compose import ColumnTransformer  

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble  import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error


## load dataset 
housing = pd.read_csv("housing.csv")

#2 create a stratfied train test split 
# we will make strata 

# extra columns for strata is made out of exicting data 
# hence we used pd.cut and split them into bins 

housing["income_cat"] = pd.cut(
    housing["median_income"],
    bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
    labels=[1, 2, 3, 4, 5]  # labels are name 1,2,3,4,5

)
 
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index].drop("income_cat", axis=1)
    strat_test_set = housing.loc[test_index].drop("income_cat", axis=1)
 
# Work on a copy of training data 
housing = strat_train_set.copy()
 
# 3. Separate features and labels
# labels are something we want our model to predict 
# rest are features 
housing_labels = housing["median_house_value"].copy()
housing = housing.drop("median_house_value", axis=1)
 
# 4. Separate numerical and categorical columns
num_attribs = housing.drop("ocean_proximity", axis=1).columns.tolist()                      
# hosing is dataframe and we want its columnns hence .columns 
# converted  to list using .list 


cat_attribs = ["ocean_proximity"] # separated  catgorical data  

 
# 5. Pipelines for Numerical columnns
# pipelies takes a list 


# they made two pipelines seprate for number and category 

num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),  #simple imputer 
    ("scaler", StandardScaler()),
])
 
# Categorical pipeline
cat_pipeline = Pipeline([
    # ("ordinal", OrdinalEncoder())  # Use this if you prefer ordinal encoding
    ##     
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])
 
# Full pipeline
full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", cat_pipeline, cat_attribs),
])
 
# 6. Transform the data
housing_prepared = full_pipeline.fit_transform(housing)
 
# housing_prepared is now a NumPy array ready for training
#  print(housing_prepared)



#TRAIN THE MODEL 


# LINEAR REGRESSION 
lin_reg =LinearRegression()
lin_reg.fit(housing_prepared,housing_labels)
lin_predic=lin_reg.predict(housing_prepared)
#lin_rmse=root_mean_squared_error(housing_labels,lin_predic)

#print(f"the rmse of linear regressio is {lin_rmse} ")




#DECISION TREE
dec_reg =DecisionTreeRegressor()
dec_reg.fit(housing_prepared,housing_labels)
dec_predic=dec_reg.predict(housing_prepared)
#dec_rmse=root_mean_squared_error(housing_labels,dec_predic)

#print(f"the rmse of decision tree  is {dec_rmse} ")

      

# random forest regressor 
randforest_reg =RandomForestRegressor()
randforest_reg.fit(housing_prepared,housing_labels)
randforest_predic=randforest_reg.predict(housing_prepared)
#randforest_rmse=root_mean_squared_error(housing_labels,randforest_predic)

#print(f"the rmse of decision tree  is {randforest_rmse} ")