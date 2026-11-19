# Imports
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Sets LLM model and temperature
llm = ChatGoogleGenerativeAI(model = "gemini-3.5-flash-lite", temperature = 0.1)

# Title and introduction
st.title("Steam Games Data Analysis")
st.write("I'll first be asking an initial question: What attributes of a Steam game are associated with a positive user rating (>80% positive reviews), and can they be used to predict it?")

# Load and cache the raw data
@st.cache_data
def load_data():
    df = pd.read_csv("/Users/emmasladden/csc405-project/data/games.csv", index_col=False)
    return df

raw_data = load_data()

# An overview of the first five entries in the raw dataset
st.subheader("Dataset Overview")
st.dataframe(raw_data.head(), hide_index=True)

# A summary of the raw dataset
raw_summary = f"""
Dataset Shape:
{raw_data.shape}

Dataset Columns:
{raw_data.dtypes}

Missing Values:
{raw_data.isnull().sum()}

Basic Statistics:
{raw_data.describe(include='all').transpose()}
"""

# AI exploratory analysis using the dataset summary
if "analysis" not in st.session_state:
    prompt = ChatPromptTemplate.from_template(
        f"""
        You are an experienced data analyst. Perform an exploratory analysis of dataset summary, including:
        1. A dataset overview
        2. The data quality & anomalies
        3. Key hypotheses
        4. Recommended visualizations

        For recommended visualizations, include at least a correlation heatmap across all numeric attributes, plus histograms, box plots, scatter plots and a release-year trend.

        Dataset Summary:
        {raw_summary}
        """
    )

    chain = prompt | llm | StrOutputParser()

    response = chain.invoke({})
    st.session_state.analysis = response

st.subheader("AI-generated Exploratory Analysis")
st.write(st.session_state.analysis)

st.subheader("Evaluation of AI-generated Exploratory Analysis")
st.write("The AI accurately found columns with mostly missing values, but its thoughts on outliers may not be accurate because it cannot view the entire dataset. I must evaluate the data myself to observe skewness. The potential key hypotheses all deserve some exploration, as well as all the recommended figures for visualization.")

# Clean and cache data
@st.cache_data
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    # Drop columns with mostly or completely missing values
    cols_to_drop=['Movies', 'Reviews', 'Metacritic url', 'Notes', 'Website', 'Support url']
    df = df.drop(columns=cols_to_drop, errors='ignore')
    # Filter out non-games to help prevent playtime skewing due to idling software
    non_games = 'Utilities|Software|Animation & Modeling|Design & Illustration|Web Publishing|Video Production'
    if 'Genres' in df.columns:
        df = df[~df['Genres'].str.contains(non_games, case=False, na=False)]
    # Drop free-to-play games to correct the heavy right-skewedness of Price
    df = df[df['Price'] > 0].copy()
    # Calculate log-IQR bounds for Price and use those to drop outlier rows
    log_price = np.log(df['Price'])
    Q1 = log_price.quantile(0.25)
    Q3 = log_price.quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = np.exp(Q1 - 1.5 * IQR)
    upper_bound = np.exp(Q3 + 1.5 * IQR)

    df = df[(df['Price'] >= lower_bound) & (df['Price'] <= upper_bound)]
    # Filter out dead/unplayed games
    df = df[(df['Peak CCU'] > 0) | (df['Positive'] + df['Negative'] >= 10)]
    # Handle extremely right-skewed columns like Peak CCU, Positive, Recommendations, DLC count, Achievements, and Average playtime forever
    skewed_cols = ['Peak CCU', 'Positive', 'Recommendations', 'DLC count', 'Achievements', 'Average playtime forever']
    df[skewed_cols] = df[skewed_cols].fillna(0).clip(lower=0)
    for col in skewed_cols:
        df[f'{col}_log'] = np.log1p(df[col])
        upper_limit = df[col].quantile(0.99)
        df[f'{col}_capped'] = df[col].clip(upper=upper_limit)
    # Replace 0 Metacritic scores with NaN
    if 'Metacritic score' in df.columns:
        df['Metacritic score'] = df['Metacritic score'].replace(0, np.nan)
    # Drop rows with >50% missing values
    df = df.dropna(thresh=df.shape[1] * 0.5)
    # Handle duplicates by using AppID alongside Name to differentiate
    if "AppID" in df.columns:
        df = df.drop_duplicates(subset=["AppID"])
    # Drop exact duplicates if there are any
    df = df.drop_duplicates()

    return df

cleaned_data = clean_data(raw_data)
st.subheader("Cleaned Dataset Overview")
st.dataframe(cleaned_data.head())

st.subheader("Data Cleaning")
st.write("To clean the dataset, I first dropped columns with either mostly or completely missing values. Then I removed entries with genres that were non-game related, like Utilities, Software, Animation & Modeling, Design & Illustration, Web Publishing, and Video Production. The next step was handling Price outliers, starting with dropping all the free-to-play games. Then I used log-IQR to find the lower and upper bounds and remove remaining outliers. I used the absence of Peak CCU and lack of reviews to prune out dead or unplayed games. Following that, I dealt with all the values that were heavily right-skewed, mainly due to the blockbuster hits with extremely high positive reviews, playtime, recommendations, etc. I used log transformtion so the outliers didn't dominate my weights. I replaced Metacritic scores with a score of 0 with NaN because there are no games on Metacritic with a score of 0, which means they are most likely just unrated and not true zeros. Lastly, I dropped rows with more that 50% missing values and dropped duplicates using the Name and AppID together to get rid of repackaged games with the same name.")