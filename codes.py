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