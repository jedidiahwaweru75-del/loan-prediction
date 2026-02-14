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
#get full informationgit
data.info()
#summary statistics
data.describe()
#check for missing values
data.isnull().sum()
# drop a column
data.drop()

data= data.drop(columns="referredby")
data.drop("referredby", axis=1, inplace=True)





