import pandas as pd

# loading data 
data=pd.read_csv('data/trainperf.csv')

#view first rows
data.head()
#view first columns
data.tail()
#check number of rows and columns
data.shape
#see column names
data.columns
#get full information
data.info()
#summary statistics
data.describe()
#check for missing values
data.isnull().sum()
# drop a column
data.drop()

data= data.drop("column_name", axis=1)
# data.drop("age", axis=1, inplace=True)





