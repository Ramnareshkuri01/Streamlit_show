import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Title
st.title("COVID-19 Data Analysis")

# Step 1: Load Dataset
url = "https://raw.githubusercontent.com/SR1608/Datasets/main/covid-data.csv"
df = pd.read_csv(url)
st.write("### Dataset Overview", df.head())

# Step 2: High-Level Data Understanding
st.write("### High-Level Data Understanding")
st.write("Number of rows and columns:", df.shape)
st.write("Data Types:", df.dtypes)
st.write("Dataset Description:", df.describe())

# Step 3: Low-Level Data Understanding
st.write("### Low-Level Data Understanding")
st.write("Unique locations count:", df['location'].nunique())
st.write("Continent with maximum frequency:", df['continent'].value_counts().idxmax())
st.write("Max and Mean of total_cases:", df['total_cases'].max(), df['total_cases'].mean())
st.write("25%, 50%, 75% Quartiles of total_deaths:", df['total_deaths'].quantile([0.25, 0.5, 0.75]))
st.write("Continent with highest human_development_index:", df.loc[df['human_development_index'].idxmax(), 'continent'])
st.write("Continent with lowest gdp_per_capita:", df.loc[df['gdp_per_capita'].idxmin(), 'continent'])

# Step 4: Data Cleaning
df = df[['continent', 'location', 'date', 'total_cases', 'total_deaths', 'gdp_per_capita', 'human_development_index']]
df.drop_duplicates(inplace=True)
df.dropna(subset=['continent'], inplace=True)
df.fillna(0, inplace=True)
st.write("### Cleaned Data", df.head())

# Step 5: Date Formatting
df['date'] = pd.to_datetime(df['date'])
df['month'] = df['date'].dt.month

# Step 6: Data Aggregation
df_groupby = df.groupby('continent').max().reset_index()
df_groupby['total_deaths_to_total_cases'] = df_groupby['total_deaths'] / df_groupby['total_cases']
st.write("### Aggregated Data", df_groupby)

# Step 7: Data Visualization
sns.set_style("whitegrid")

st.write("## Data Visualizations")

# GDP Per Capita Distribution
st.write("### GDP Per Capita Distribution")
fig, ax = plt.subplots()
sns.histplot(df['gdp_per_capita'], kde=True, bins=30, ax=ax)
ax.set_title("Distribution of GDP Per Capita")
st.pyplot(fig)

# Scatter Plot - Total Cases vs GDP Per Capita
st.write("### Total Cases vs GDP Per Capita")
fig, ax = plt.subplots()
sns.scatterplot(x=df['total_cases'], y=df['gdp_per_capita'], ax=ax)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_title("Total Cases vs GDP Per Capita")
st.pyplot(fig)

# Pairplot of Aggregated Data
st.write("### Pairplot of Aggregated Data")
st.pyplot(sns.pairplot(df_groupby))

# Bar Plot - Continent vs Total Cases
st.write("### Total Cases by Continent")
fig, ax = plt.subplots()
sns.barplot(x="continent", y="total_cases", data=df_groupby, ax=ax)
plt.xticks(rotation=45)
ax.set_title("Total Cases by Continent")
st.pyplot(fig)

# Step 8: Download Aggregated Data
csv = df_groupby.to_csv(index=False).encode('utf-8')
st.download_button("Download Aggregated Data", data=csv, file_name="df_groupby.csv", mime='text/csv')
