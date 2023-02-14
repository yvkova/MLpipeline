# This script implements the preprocessing of the dataset used for the task of predicting EU sales based on global sales and characteristics of games
# Dataset: https://www.kaggle.com/gregorut/videogamesales

# Import libraries
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Read in the dataset
df = pd.read_csv('vgsales.csv')

# View the first 5 rows
print("View initial head(5) of dataset:\n")
print(df.head(5))

# View total rows and columns
print("\nView initial shape of dataset:\n")
print(df.shape)

# View statistics of numeric columns
print("\nView describe function on numeric columns:\n")
print(df.describe())

# Check for missing values
print("\nCheck for missing values:\n")
print(df.isnull().sum())

# Drop unwanted columns (rank, name)
df = df.drop(['Rank', 'Name'], axis=1)

# Re-view the first 5 rows
print("\nView head(5) after dropping 'Rank' and 'Name' columns:\n")
print(df.head(5))

# View total rows and columns before dropping data
print("\nView total rows and columns before dropping missing values:\n")
print(df.shape)

# Drop rows with missing values (Year, Publisher)
df = df.dropna(subset=['Year', 'Publisher'])

# View total rows and columns after dropping data
print("\nView total rows and columns after dropping missing values:\n")
print(df.shape)

# Re-check for missing values
print("\nRecheck for missing values:\n")
print(df.isnull().sum())

# Remove rows where global sales is less than 0.2 million
df = df[df['Global_Sales'] > 0.2]

# Re-view total rows and columns after dropping global sales less than 0.2
print("\nView total rows and columns after dropping global sales less than 0.2:\n")
print(df.shape)

# Visualise Platform, Genre and Publisher columns before LabelEncoder is applied
print("\nView head(5) of 'Platform', 'Genre' and 'Publisher' columns before applying LabelEncoder:\n")
print(df[['Platform', 'Genre', 'Publisher']].head(5))

# Create instance of LabelEncoder
le = LabelEncoder()

# Convert string values to numbers using LabelEncoder
df['Platform'] = le.fit_transform(df['Platform'])
df['Genre'] = le.fit_transform(df['Genre'])
df['Publisher'] = le.fit_transform(df['Publisher'])

# Visualise Platform, Genre and Publisher columns after LabelEncoder is applied
print("\nView head(5) of 'Platform', 'Genre' and 'Publisher' columns after applying LabelEncoder:\n")
print(df[['Platform', 'Genre', 'Publisher']].head(5))

# Split data into features (X) and target (y)
X = df[['Platform', 'Year', 'Genre', 'Publisher', 'Global_Sales']]
y = df['EU_Sales']

# Visualise the head of X features
print("\nView head(5) of X (features) after splitting data:\n")
print(X.head(5))

# Visualise the head of y target
print("\nView head(5) of y (target) after splitting data:\n")
print(y.head(5))
