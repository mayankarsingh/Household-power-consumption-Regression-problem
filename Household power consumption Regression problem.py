#!/usr/bin/env python
# coding: utf-8

# ## Problem Statement:- Household power consumption Regression problem

# ### Regression problem
# 
# 1. Collect dataset from here
# 2. https://archive.ics.uci.edu/ml/datasets/individual+household+electric+power+consumption
# 3. Here the number of instances is very high, so take a random sample of 50k using the sample().
# 4. Add all the three columns named sub_metering_1, sub_metering_2 and sub_metering_3 to get the total energy consumed.
# 5. Create a Regression model on the basis of attributes.
# 6. Create Linear Regression, Ridge Regression, Lasso Regression,ElasticNet Regression, Support Vector Regression.
# 
# ### Steps to be followed
# 1. Data ingestion.
# 2. EDA (end to end).
# 3. Preprocessing of the data.
# 4. Use pickle to store the scaling of the data for later use.
# 5. Store the final processed data inside MongoDB.
# 6. Again load the data from MongoDB.
# 7. Model building.
# 8. Use GridSearchCV for hyper parameter tuning.
# 9. **Evaluation** :- R2 and adjusted R2 for regression model.

# ### Attribute Information:
#     
# 1. date: Date in format dd/mm/yyyy
# 2. time: time in format hh:mm:ss
# 3. global_active_power: household global minute-averaged active power (in kilowatt)
# 4. global_reactive_power: household global minute-averaged reactive power (in kilowatt)
# 5. voltage: minute-averaged voltage (in volt)
# 6. global_intensity: household global minute-averaged current intensity (in ampere)
# 7. sub_metering_1: energy sub-metering No. 1 (in watt-hour of active energy). It corresponds to the kitchen, containing mainly a dishwasher, an oven and a microwave (hot plates are not electric but gas powered).
# 8. sub_metering_2: energy sub-metering No. 2 (in watt-hour of active energy). It corresponds to the laundry room, containing a washing-machine, a tumble-drier, a refrigerator and a light.
# 9. sub_metering_3: energy sub-metering No. 3 (in watt-hour of active energy). It corresponds to an electric water-heater and an air-conditioner.

# ### 1. Data Ingestion:
# **Importing the required libraries**

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
sns.set()
get_ipython().run_line_magic('matplotlib', 'inline')
warnings.filterwarnings('ignore')


# In[2]:


data = pd.read_csv('C:\\Users\\mayankar\\Desktop\\mm\\household_power_consumption.csv')
data


# In[3]:


data.drop(['index'],axis=1,inplace=True)


# In[4]:


data.shape


# #### Note
# **1. Here the number of rows is very high (260640 rows), let's take a sample of 50000 observations**

# #### Creating dataframe with random 50000 observations

# In[5]:


data = data.sample(50000)
data


# In[6]:


data.info()


# ### Observations:
# 1. Now there are 50000 rows and 9 columns (features) in the dataset.
# 2. All the columns except Sub_metering_3 is of object type, even though they have float values

# ### 2. Data Cleaning:

# In[7]:


data.columns


# #### 2.1 Dropping unnecesarry columns

# In[8]:


data.drop(columns=['Date', 'Time'], axis=1, inplace=True)
data


# #### 2.2 Converting data types and replacing special characters

# In[9]:


for column in data.columns:
        print(f"The unique values in column {column}:")
        print(data[column].unique())
        print(f"\nThe number of unique values in {column} is:{len(data[column].unique())}")
print("----------------------------------\n")


# ### Observations:
# 1. We have special character **?** in columns **Sub_metering_1**, **Sub_metering_2**, **Global_intensity**.
# 2. Also the columns **Global_active_power** and **Voltage** have more than 1000 unique values. So we need to check for special characters in them as well.
# 3. We have **nan** in **Sub_metering_3** as well

# In[10]:


# To find special characters in these 2 columns
data.loc[data['Global_active_power'] == "?", :]


# In[11]:


data.loc[data['Voltage'] == "?", :]


# 1. **So yes there are 759 rows where the ? is present in the dataset**
# 2. **Also it looks like the sign appears in all the columns at the same time**
# 3. **As the percentage of these rows is 1% of the total dataset so we can drop them**

# ### Dropping the rows

# In[12]:


data.drop(data.loc[data['Voltage'] == "?", :].index, inplace=True)
data.shape


# In[13]:


data.loc[data['Voltage'] == "?", :]


# In[14]:


data.loc[data['Global_active_power'] == "?", :]


# In[15]:


# Now again checking for nan values
data.isnull().sum()


# In[16]:


# Converting the data types
data = data.astype({'Global_active_power':float,'Global_reactive_power':float, 'Voltage':float,'Global_intensity':float,'Sub_metering_1':float, 'Sub_metering_2':float})


# In[17]:


# checking the dataset
data.info()


# #### 2.3 Let's check for duplicate values

# In[18]:


data[data.duplicated()]


# In[19]:


# Dropping the duplicated values as well
data.drop_duplicates(inplace=True)


# In[20]:


# Creating a new column for 'total energy consumed'
# Then removing the columns 'Sub_metering_1', 'Sub_metering_2' and'Sub_metering_3'
data["Total_energy_consumed"] = data['Sub_metering_1'] + data['Sub_metering_2'] + data['Sub_metering_3']
data.drop(columns=['Sub_metering_1', 'Sub_metering_2','Sub_metering_3'], axis=1, inplace=True)
data.head()


# In[21]:


data.shape


# In[22]:


data[data.duplicated()]


# ### 3. Exploratory data analysis

# #### 3.1 Basic Profile of the data
# 

# In[23]:


# Checking the details of the dataframe
data.info()


# #### Differentiating numerical and categorical columns

# In[24]:


numerical_features = [feature for feature in data.columns if data[feature].dtypes != 'O']
categorical_features = [feature for feature in data.columns if data[feature].dtypes == 'O']
print(f"The number of Numerical features are:{len(numerical_features)}, and the column names are:\n{numerical_features}")
print(f"\nThe number of Categorical features are:{len(categorical_features)}, and the column names are:\n{categorical_features}")


# ### Observations:
#   1. **Now we have 49189 rows with no null and duplicate values and all the 5 columns have numerical (float) data type.**

# #### 3.2 Statistical Analysis of the data

# In[25]:


data.describe().T


# ### Observations:
# 1. There are possible Outliers in columns **Global_active_power,Global_intensity, Total_energy_consumed**

# #### 3.3 Graphical Analysis of the data

# #### 3.3.1 Univariate Analysis
# 1. For numerical features
# 2. Kernal Density plots

# In[ ]:


plt.figure(figsize=(15, 15))
plt.suptitle('Univariate Analysis of Numerical Features', fontsize=20,fontweight='bold', alpha=0.8, y=1.)
for i in range(0, len(numerical_features)):
    plt.subplot(3, 2, i+1)
    sns.kdeplot(x=data[numerical_features[i]], shade=True, color='r')
    plt.xlabel(numerical_features[i], fontsize=15)
    plt.tight_layout()


# In[27]:


# Histograms
plt.figure(figsize=(15, 15))
plt.suptitle('Univariate Analysis of Numerical Features', fontsize=20,fontweight='bold', alpha=0.8, y=1.)
for i in range(0, len(numerical_features)):
    plt.subplot(3, 2, i+1)
    sns.histplot(x=data[numerical_features[i]], kde=True, color='r')
    plt.xlabel(numerical_features[i], fontsize=15)
    plt.tight_layout()


# ### Observations:
# 1. Only Voltage has normal distribution.
# 2. All other columns are right skewed and they may have outliers.
# 3. Too many values near to 0 in Global_active_power, Global_reactive_power,Global_intensity and Total_energy_consumed columns.

# **3.3.2 Multivariate Analysis**
# #### Checking Multicollinearity in the numerical features

# In[28]:


data[list(data[numerical_features].columns)].corr()


# In[29]:


# Graphical representation
sns.pairplot(data[numerical_features])
plt.show()


# In[30]:


sns.set(rc={'figure.figsize':(15,10)})
sns.heatmap(data[numerical_features].corr(), cmap='CMRmap', annot=True)
plt.show()


# ### Observations:
# 1. Global_intensity and Global_active_power is completely correlated.
# 2. Total_energy_consumed is also highly correlated with Global_intensity and Global_active_power.

# ### 4. Data Pre-Processing
# #### 4.1 Number of unique values in each column

# In[31]:


data.nunique()


# #### 4.2 Outlier handling

# In[32]:


# Creating a function to detect outliers

def detect_outliers(col):
    percentile25 = data[col].quantile(0.25)
    percentile75 = data[col].quantile(0.75)
    print('\n ####', col , '####')
    print("25percentile: ",percentile25)
    print("75percentile: ",percentile75)
    iqr = percentile75 - percentile25
    upper_limit = percentile75 + 1.5 * iqr
    lower_limit = percentile25 - 1.5 * iqr
    print("Upper limit: ",upper_limit)
    print("Lower limit: ",lower_limit)
    data.loc[(data[col]>upper_limit), col]= upper_limit
    data.loc[(data[col]<lower_limit), col]= lower_limit
    return data


# In[33]:


# Now applying the function on all the columns as all are of continupus type
for col in numerical_features:
    detect_outliers(col)


# In[34]:


# Again checking
plt.figure(figsize=(15, 15))
plt.suptitle('Univariate Analysis of Numerical Features', fontsize=20,
fontweight='bold', alpha=0.8, y=1.)
for i in range(0, len(numerical_features)):
    plt.subplot(3, 2, i+1)
    sns.histplot(x=data[numerical_features[i]], kde=True, color='r')
    plt.xlabel(numerical_features[i], fontsize=15)
    plt.tight_layout()


# In[35]:


fig, ax = plt.subplots(figsize=(15,10))
plt.suptitle('Finding Outliers in Numerical Features', fontsize=20,fontweight='bold', alpha=0.8, y=1.)
sns.boxplot(data=data[numerical_features], width= 0.5, ax=ax,fliersize=3)
plt.show()


# ### Observations:
# 1. Now we can see that the outliers are gone.

# In[36]:


# Let's save the clean data to a folder and then to mongodb for later use

try:
    data.to_csv("C:\\Users\\mayankar\\Desktop\\mm\\household_power_cleaned.csv", index=None)
except Exception as err:
    print("Error is: ", err)
else:
    print("Clean csv file created successfully.")


# In[37]:


# converting to json file

df2 = pd.read_csv('C:\\Users\\mayankar\\Desktop\\mm\\household_power_cleaned.csv')
try:
    df2.to_json('C:\\Users\\mayankar\\Desktop\\mm\\household_power_cleaned.json')
except Exception as err:
    print("Error is: ", err)
else:
    print("Json file created successfully.")


# #### MongoDB part

# In[38]:


# Checking the file
df_json = pd.read_json('C:\\Users\\mayankar\\Desktop\\mm\\household_power_cleaned.json')
df_json.head()


# In[39]:


pip install pymongo


# In[40]:


import pymongo
from pymongo import MongoClient


# In[41]:


# connecting with the server

try:
    client = pymongo.MongoClient("mongodb+srv://ineuron:Project1@cluster0.rp4qzrr.mongodb.net/?retryWrites=true&w=majority")
except Exception as e:
    print(e)
else:
    print("Connection to MongoDB server is successful.")


# In[42]:


# creating database and collection
db = client["household_power_consumption"]
coll = db['power_consumption']
try:
    import json
except ImportError:
    import simplejson as json


# In[43]:


# Inserting the data into the collection

try:
    with open('C:\\Users\\mayankar\\Desktop\\mm\\household_power_cleaned.json') as file:
        file_data = json.load(file)
        coll.insert_many([file_data])
except Exception as e:
    print(e)
else:
    print("Data inserted successfully.")


# In[44]:


# Loading the data from MongoDB
# Now to read the data
# importing the library to take care of the objectid created by mongodb
import bson.json_util as json_util
results = coll.find()
try:
    for result in results:
        data = json_util.dumps(result)
        clean_df = pd.read_json(data, orient='index')
except Exception as e:
    print(e)
else:
    clean_df


# In[45]:


clean_df.T


# In[46]:


# Again transposing so we can get the '$oid' as a column
clean_df = clean_df.transpose()
clean_df.head()


# **Removing the column _id and row oid as we do not need them**

# In[47]:


# Removing the '_id' column
clean_df.drop(['_id'], axis=1, inplace=True)
clean_df


# In[48]:


# Again transposing so we can get the '$oid' as a column
clean_df = clean_df.transpose()
clean_df.head()


# In[49]:


# Removing the '$oid' column
clean_df.drop(['$oid'], axis=1, inplace=True)
clean_df


# In[50]:


# getting the actual dataframe
clean_df = clean_df.transpose()
clean_df.head()


# In[51]:


final_df = clean_df.copy()
final_df


# #### 4.3 Creating independent and dependent variables

# **Split X and y**
# 1. **Split Dataframe to X and y**
# 2. **Here we set a variable X i.e, independent columns, and a variable y i.e,dependent column as the Total_energy_consumed column.**

# In[52]:


X = final_df.drop("Total_energy_consumed", axis=1)
y = final_df["Total_energy_consumed"]


# In[53]:


# Checking the independent and dependent variables
X.head()


# In[54]:


y.head()


# In[55]:


# Doing Test Train split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.33, random_state=42)


# In[56]:


# Let's see the datasets
X_train.head()


# In[57]:


y_train.head()


# In[58]:


X_test.head()


# In[59]:


y_test.head()


# ### Let's check the shapes of each datasets

# In[60]:


X_train.shape


# In[61]:


X_test.shape


# ### Observations:
# ***1. So now we have 32493 rows for training and 16005 for test datasets***

# #### 4.4 Standardizing or feature scaling the dataset

# In[62]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler
StandardScaler()


# In[63]:


# calculate the mean and std dev
# Here we are fitting only the training data without transforming
scale = scaler.fit(X_train)
scale
StandardScaler()


# In[64]:


# Printing the mean
print(scale.mean_)


# **Saving the scale to use it later to transform the data and predict the values**

# In[65]:


# To save a Standard scaler object
import pickle
with open('scaled.pkl', 'wb') as f:
    pickle.dump(scale, f)


# In[66]:


# Loading the scaled object to transform the data
with open('scaled.pkl', 'rb') as f:
    scaled = pickle.load(f)


# In[67]:


# Now transforming the train and test dataset
X_train_tf = scaled.transform(X_train)
X_test_tf = scaled.transform(X_test)


# In[68]:


# Checking the transformed data
X_train_tf


# In[69]:


X_test_tf


# ### 5. Model Building

# #### 5.1 Import required packages for model training

# In[70]:


from sklearn.linear_model import LinearRegression, Ridge, Lasso,ElasticNet
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error,r2_score


# #### 5.2 Create a Function to evaluate all the models

# In[71]:


def evaluate_model(true, predicted, X_test_tf):
    mae = mean_absolute_error(true, predicted)
    mse = mean_squared_error(true, predicted)
    rmse = np.sqrt(mean_squared_error(true, predicted))
    r2_square = r2_score(true, predicted)
    adj_r2 = 1 - (1 - r2_square)*(len(true)-1)/(len(true) - X_test_tf.shape[1] - 1)
    return mae, rmse, r2_square, adj_r2
models = {
    "Linear Regression": LinearRegression(),
    "Lasso": Lasso(),
    "Ridge": Ridge(),
    "Elastic": ElasticNet(),
    "svr": SVR()
}
model_list = []
r2_list =[]
adj_r2_list = []
for i in range(len(list(models))):
    model = list(models.values())[i]
    # Train model
    model.fit(X_train_tf, y_train)
    # Make predictions
    y_train_pred = model.predict(X_train_tf)
    y_test_pred = model.predict(X_test_tf)
    # Evaluate Train and Test dataset
    model_train_mae , model_train_rmse, model_train_r2,model_train_adjusted_r2 = evaluate_model(y_train, y_train_pred,X_test_tf)
    model_test_mae , model_test_rmse, model_test_r2,model_test_adjusted_r2 = evaluate_model(y_test, y_test_pred,X_test_tf)
    print(list(models.keys())[i])
    model_list.append(list(models.keys())[i])
    print('Model performance for Training set')
    print("- Root Mean Squared Error:{:.4f}".format(model_train_rmse))
    print("- Mean Absolute Error: {:.4f}".format(model_train_mae))
    print("- R2 Score: {:.4f}".format(model_train_r2))
    print("- Adjusted R2 Score:{:.4f}".format(model_train_adjusted_r2))
    print('----------------------------------')
    print('Model performance for Test set')
    print("- Root Mean Squared Error: {:.4f}".format(model_test_rmse))
    print("- Mean Absolute Error: {:.4f}".format(model_test_mae))
    print("- R2 Score: {:.4f}".format(model_test_r2))
    print("- Adjusted R2 Score:{:.4f}".format(model_test_adjusted_r2))
    r2_list.append(model_test_r2)
    adj_r2_list.append(model_test_adjusted_r2)
    print('='*50)
    print('\n')


# #### 5.3 Results of all Models

# In[72]:


adj_r2_list


# In[73]:


pd.DataFrame(list(zip(model_list, r2_list, adj_r2_list)),columns=['Model Name', 'R2_Score', 'AdjustedR2_Score']).sort_values(by=["R2_Score"], ascending=False)


# ### Observations:
# 1. **We can see the best Adjusted R2 value is of the SVR model but the SVR, Ridge,Linear Regression models are also very close.**
# 2. **So now we can use SVR for Hyper Parameter Tuning to find it's best values**

# #### 5.4 Hyper Parameter Tuning (using GridSearchCV)

# In[74]:


# importing the library
from sklearn.model_selection import GridSearchCV


# In[75]:


# Creating the svr model
svr = SVR()
svr
SVR()


# In[76]:


# training the model
svr.fit(X_train_tf, y_train)
SVR()
params = {'kernel':('linear', 'rbf')}
grid = GridSearchCV(estimator=svr, param_grid=params, cv=3, verbose=2,n_jobs=-1)
grid.fit(X_train_tf, y_train)
print(grid.best_params_)


# #### 5.5 Training the model with best Parameters

# In[77]:


best_model = SVR(kernel='rbf')
best_model


# In[78]:


# training the model
best_model.fit(X_train_tf, y_train)


# #### 5.6 Saving the optimized model for later usage

# In[79]:


# saving the model
with open('model.pkl', 'wb') as f:
    pickle.dump(best_model, f)


# #### Testing the model with new data to get prediction
# 

# In[ ]:


# Inserting data from outside
Global_active_power = float(input("Enter The value: "))
Global_reactive_power = float(input("Enter The value: "))
Voltage = float(input("Enter The value: "))
Global_intensity = float(input("Enter The value: "))
test_set = {'Global_active_power': Global_active_power,'Global_reactive_power': Global_reactive_power,'Voltage':Voltage, 'Global_intensity':Global_intensity}
print("\nThe entered values are:\n")
print(test_set)


# In[ ]:


# creating a dataframe of the entered data
test_set = pd.DataFrame(test_set, index=[1])
test_set


# In[ ]:


test_set.shape


# In[ ]:


# Loading the scaled object to transform the entered data
with open('scaled.pkl', 'rb') as f:
    scaled = pickle.load(f)


# In[ ]:


# Now transforming the enetered data
test_set_tf = scaled.transform(test_set)
test_set_tf


# In[ ]:


# loading the model
with open('model.pkl', 'rb') as f:
    new_model = pickle.load(f)


# In[ ]:


# predicting the output
test_set_pred = new_model.predict(test_set_tf)
print("So total energy consumed by the entered data will be:{:.2f}".format(float(test_set_pred)))


# In[ ]:




