from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
import pandas as pd
import numpy as np


## load dataset 
housing = pd.read_csv("housing.csv")

#print(housing.columns.tolist()) 


#2 create a stratfied train test split 

# we will make strata 

housing["income_cat"] = pd.cut(
    housing["median_income"],
    bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
    labels=[1, 2, 3, 4, 5]
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
# hosing is dataframe and we want its columnns henncee .columns 
# converted  to list using .list 


cat_attribs = ["ocean_proximity"]
 
# 5. Pipelines for Numerical columnns
# pipelies takes a list 
num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])
 
# Categorical pipeline
cat_pipeline = Pipeline([
    # ("ordinal", OrdinalEncoder())  # Use this if you prefer ordinal encoding
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
print(housing_prepared)

