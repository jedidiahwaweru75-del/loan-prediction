import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# loading data 
data_train=pd.read_csv('data/trainprevloans.csv')
data_test=pd.read_csv('data/testprevloans.csv')

#view first rows
data_train.head()
#view first columns
data_train.tail()
#check number of rows and columns
data_train.shape
#see column names
data_train.columns
#get full information
data_train.info()
#summary statistics
data_train.describe()
#check for missing values
data_train.isnull().mean()>0.5 
#drop reffered by 
data_train.drop(columns='referredby',inplace=True)
data_train.info()

# Check shape
print("trainprevloans shape:", data_train.shape)
print("testprevloans shape:", data_test.shape)

# Convert date columns for BOTH datasets
date_cols = [
    'approveddate',
    'creationdate',
    'closeddate',
    'firstduedate',
    'firstrepaiddate'
]

for col in date_cols:
    data_train[col] = pd.to_datetime(data_train[col])
    data_test[col] = pd.to_datetime(data_test[col])

# value counts for categorical values
categorical_cols = data_train.select_dtypes(include='object').columns

for col in categorical_cols:
    print(f"\nValue counts for {col}:")
    print(data_train[col].value_counts())

# STEP 4: DATA WRANGLING
# ==============================

# Select specific columns
selected_cols = ['loanamount', 'totaldue', 'termdays']
train_selected = data_train[selected_cols]

print("\nSelected Columns Preview:")
print(train_selected.head())


# Create repayment ratio for BOTH
data_train['repayment_ratio'] = data_train['totaldue'] / data_train['loanamount']
data_test['repayment_ratio'] = data_test['totaldue'] / data_test['loanamount']

# Create loan duration
data_train['loan_duration_days'] = (
    data_train['closeddate']-data_train['approveddate']
).dt.days

data_test['loan_duration_days'] = (
    data_test['closeddate'] - data_test['approveddate']
).dt.days

# Extract year & month
data_train['approved_year'] = data_train['approveddate'].dt.year
data_train['approved_month'] = data_train['approveddate'].dt.month

data_test['approved_year'] = data_test['approveddate'].dt.year
data_test['approved_month'] = data_test['approveddate'].dt.month

#grouping 
print(data_train.groupby('termdays')['loanamount'].mean())
print(data_train.groupby('approved_year')['repayment_ratio'].mean())

#DATA VISUALIATION
# 1️⃣ Histogram of Loan Amount
plt.figure()
plt.hist(data_train['loanamount'], bins=30)
plt.title("Distribution of Loan Amount")
plt.xlabel("Loan Amount")
plt.ylabel("Frequency")
plt.show()

# 2️⃣ Bar Chart of Term Days Count
term_counts = data_train['termdays'].value_counts()

plt.figure()
plt.bar(term_counts.index, term_counts.values)
plt.title("Loan Term Days Counts")
plt.xlabel("Term Days")
plt.ylabel("Count")
plt.show()

# 3️⃣ Scatter Plot: Loan Amount vs Total Due
plt.figure()
plt.scatter(data_train['loanamount'], data_train['totaldue'])
plt.title("Loan Amount vs Total Due")
plt.xlabel("Loan Amount")
plt.ylabel("Total Due")
plt.show()


print("\nTest Data Preview:")
print(data_test.head())

print("\nTest Summary Statistics:")
print(data_test.describe())




