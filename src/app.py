# Imports
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from scipy.stats import chi2_contingency
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)

# Sets LLM model and temperature
llm = ChatGoogleGenerativeAI(model = "gemini-3.5-flash-lite", temperature = 0.1)

# Title and introduction
st.title("Steam Games Data Analysis")
st.write("This dataset holds information on 125, 855 games published on Steam. I'll first be asking an initial question: What attributes of a Steam game are associated with a positive user rating (>80% positive reviews), and can they be used to predict it?")

# Load and cache the raw data
@st.cache_data
def load_data():
    df = pd.read_csv("/Users/emmasladden/csc405-project/data/games.csv", index_col=False)
    return df

raw_data = load_data()

# An overview of the first five entries in the raw dataset
st.header("Dataset Overview")
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

# AI exploratory analysis using the raw dataset summary
if "analysis" not in st.session_state:
    st.session_state.analysis = None

if st.session_state.analysis is None:
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

st.header("Exploratory Analysis")
st.subheader("AI-generated Exploratory Analysis")
st.write(st.session_state.analysis)

st.subheader("Evaluation of AI-generated Exploratory Analysis")
st.write(
    """
    The AI accurately observed columns with extremely high missing rates, but its thoughts on outliers and skewednessnmay not be accurate because it cannot view the entire dataset. 
    I must evaluate the data myself to observe what needs either transformation or removal.
    The potential key hypotheses all deserve some exploration, as well as all the recommended figures for visualization.
    """
)

# Clean and cache data
@st.cache_data
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    # Drop columns with mostly or completely missing values
    cols_to_drop=['Movies', 'Reviews', 'Metacritic url', 'Notes', 'Website', 'Support url', 'Score rank']
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
    df = df[(df['Peak CCU'] > 0) & (df['Positive'] + df['Negative'] >= 10)]
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
    # Parse string lists for tags and genres into dummy variables
    if "Genres" in df.columns:
        genres_dummies = df["Genres"].str.get_dummies(sep=",")
        genres_dummies = genres_dummies.add_prefix("genre_")
        df = pd.concat([df, genres_dummies], axis=1)
    if "Tags" in df.columns:
            tags_dummies = df["Tags"].str.get_dummies(sep=",")
            tags_dummies = tags_dummies.add_prefix("tags_")
            df = pd.concat([df, tags_dummies], axis=1)
    # Map estimated owner ranges to their midpoints, lower bounds, and upper bounds to make them usable numbers
    if "Estimated owners" in df.columns:
        bounds = df['Estimated owners'].str.replace(',', '').str.split(' - ', expand=True).astype(float)
        df['min_owners'] = bounds[0]
        df['max_owners'] = bounds[1]
        df['midpoint_owners'] = (bounds[0] + bounds[1]) / 2
    return df

cleaned_data = clean_data(raw_data)
st.subheader("Cleaned Dataset Overview")
st.dataframe(cleaned_data.head())

st.subheader("Data Cleaning")
st.write(
    """
    To clean the dataset, columns with mostly or completely missing values were dropped. 
    Entries with genres that are non-game related, like Utilities, Software, Animation & Modeling, Design & Illustration, Web Publishing, and Video Production, were removed. 
    Price outliers were handled using the log-IQR method after dropping all the free-to-play games.
    The absence of Peak CCU or a lack of reviews was utilized to prune out dead or unplayed games. 
    All the values that were heavily right-skewed, such as Peak CCU, Positive, Recommendations, DLC count, Achievements, Average playtime forever, were handled vis log transformtion so the outliers didn't dominate any weights. 
    Metacritic scores of 0 were replaced with NaN because they are most likely just unrated games and not true zeros. 
    Rows with more that 50% missing values and duplicates were dropped.
    The string lists for tags and genres were parsed into dummy variables.
    The estimated owners ranges were mapped to usable midpoint, max, and min numerical values.
    """
    )

# A summary of the cleaned dataset
clean_summary = f"""
Dataset Shape:
{cleaned_data.shape}

Dataset Columns:
{cleaned_data.dtypes}

Missing Values:
{cleaned_data.isnull().sum()}

Basic Statistics:
{cleaned_data.describe(include='all').transpose()}
"""

# AI overview using the cleaned dataset summary compared to the raw dataset summary
if "overview" not in st.session_state:
    st.session_state.overview = None

if st.session_state.overview is None:
    prompt = ChatPromptTemplate.from_template(
        f"""
        You are an experienced data analyst. Perform a dataset overview of my now cleaned dataset, including:
        1. A dataset overview
        2. The data quality & remaining anomalies (if any)
        3. Recommended visualizations: specify exactly what to plot (all exact column names), and why.

        For recommended visualizations, include at least a correlation heatmap across all numeric attributes, plus histograms, box plots, scatter plots and a release-year trend.

        Raw Dataset Summary:
        {raw_summary}

        Clean Dataset Summary:
        {clean_summary}
        """
    )

    chain = prompt | llm | StrOutputParser()

    response = chain.invoke({})
    st.session_state.overview = response

st.subheader("AI-generated Clean Dataset Overview")
st.write(st.session_state.overview)

st.subheader("Evaluation of AI-generated Clean Dataset Overview")
st.write(
    """
    Cleaning the data appears to have significatly cut down the number of entries left for analysis. 
    The AI has pointed out that there are still a few anomalies after cleaning, but they shouldn't affect my analysis moving forward. 
    The recommended plots are similar to the ones from the exploratory analysis, which will be used later in the report.
    """
    )

st.header("Statistical Analysis & Hypothesis Testing")

st.subheader("Hypothesis Testing: Genre vs. Game Success")

# Work with a copy of the dataset
h_data = cleaned_data.copy()

# State the null and alternative hypotheses
st.markdown("#### 1. Null and Alternative Hypotheses")
st.markdown(
    """
    - **Null Hypothesis ($H_0$):** Genre does not affect a game's overall chance at success (genre and success are independent).
    - **Alternative Hypothesis ($H_1$):** Genre affects a game's overall chance at success (genre and success are dependent).
    - **Significance Level ($α$):** 0.05
    """
)

# Explain why the test is appropriate
st.markdown("#### 2. Appropriateness of the Chi-Square Test")
st.markdown(
    """
    The **Chi-Square Test of Independence** is appropriate here because it is analyzing the relationship between two *categorical variables*:
    1. **Genre:** Categorical data (represented via tag columns or primary genres)
    2. **Success:** A binary categorical variable derived from the success criteria (>80% positive reviews and high estimated midpoint owners)

    The Chi-Square test evaluates whether the observed frequencies in each category differ significantly from the expected frequencies under the assumption of independence.
    """
)

# Data preparation & feature engineering
# Define success
if "Positive" in h_data.columns and "Negative" in h_data.columns:
    total_reviews = h_data['Positive'] + h_data['Negative']
    h_data['Positive_Ratio'] = np.where(
        total_reviews > 0, h_data['Positive'] / total_reviews, 0
    )
    h_data['Success'] =(h_data['Positive_Ratio'] > 0.8) & (
        h_data['midpoint_owners']
        > h_data['midpoint_owners'].median()
    )

h_data['Success_Label'] = h_data['Success'].map(
    {True: 'Successful', False: 'Not Successful'}
)
# Extract a primary genre column from avaliable tag/genre columns if 'Genre' isn't singular
genre_cols = [col for col in h_data.columns if col.startswith("genre_")]
if not genre_cols:
    genre_cols = [col for col in h_data.columns if col.startswith("tags_")]
# Select the top 5 most common genre/tag columns to create a categorical 'Primary Genre' feature
def get_primary_genre(row):
    for col in genre_cols[:10]:
        if row[col] == 1:
            return col.replace("genre_", "").replace("tags_", "")
    return "Other"

h_data["Primary_Genre"] = h_data.apply(get_primary_genre, axis=1)

# Create a contingency table
contingency_table = pd.crosstab(
    h_data["Primary_Genre"], h_data["Success_Label"]
)

# Perform the Chi-Square Test
chi2_stat, p_val, dof, ex = chi2_contingency(contingency_table)

# Report the test statistic and p-value
st.markdown("#### 3. Test Results")
st.markdown(
    f"""
    - **Chi-Square Statistic ($\chi^2$):** `{chi2_stat:.4f}`
    - **Degrees of Freedom (dof):** `{dof}`
    - **p-value:** `{p_val:.4e}`
    """
)

# Interpret the result in context
st.markdown("#### 4. Interpretation")
if p_val < 0.05:
    interpretation = (
        f"Since the p-value ({p_val:.4e}) is less than the significance level ($α = 0.05$), "
        "we reject the null hypothesis ($H_0$). There is statistically significant evidence to conclude "
        "that a game's genre affects its overall chance at success."
    )
else:
    interpretation = (
        f"Since the p-value ({p_val:.4e}) is greater than or equal to the significance level ($α = 0.05$), "
        "we fail to reject the null hypothesis ($H_0$). There is insufficient evidence to conclude "
        "that a game's genre affects its overall chance at success."
    )

st.markdown(interpretation)

st.header("Machine Learning")
st.subheader("Logistic Regression & Random Forest Classifier")

# Work with a copy of the hypotheses dataset
ml_data = h_data.copy()

st.markdown("#### 1. Data Preperation and Feature Selection")
st.write(
    """
    The feature matric `X` and target variable `y` are isolated.
    To prevent data leakage and dimensionality issues, non-numeric identifier columns, text names, date strings, direct target representations, and post-release metrics like `Positive_Ratio` or `Peak CCU` are dropped if they compromise pre-launch prediction goals.
    For this analysis, only numeric/boolean features are selected.
    """
)

# Drop known leakages, indentifiers, and non-numeric columns that aren't encoded
drop_cols = ['AppID', 'Name', 'Release date', 'Estimated owners', 'Success_Label', 'Success', 'Primary_Genre', 'Peak CCU', 'Positive_Ratio', 'Metacritic score']
feature_cols = [col for col in ml_data.columns if col not in drop_cols and ml_data[col].dtype in ['int64', 'float64', 'bool']]

X = ml_data[feature_cols].select_dtypes(include=[np.number, 'bool']).astype(float)
y = ml_data['Success'].astype(int)

st.markdown(f"**Features selected for modeling:** {X.shape[1]} columns")
st.markdown(f"**Target shape:** {y.shape}")

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

st.markdown("#### 2. Basic Baseline: Logistic Regression")
st.markdown("Logistic regression requires feature scaling for optimal gradient convergence and regularization performance.")

# Scale data for Logistic Regression
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train_scaled, y_train)

lr_preds = lr_model.predict(X_test_scaled)
lr_probs = lr_model.predict_proba(X_test_scaled)[:, 1]

lr_acc = accuracy_score(y_test, lr_preds)
lr_prec = precision_score(y_test, lr_preds, zero_division=0)
lr_rec = recall_score(y_test, lr_preds, zero_division=0)
lr_f1 = f1_score(y_test, lr_preds, zero_division=0)
lr_auc = roc_auc_score(y_test, lr_probs)
lr_cm = confusion_matrix(y_test, lr_preds)

st.markdown("##### Logistic Regression Results")
st.markdown(f"- **Accuracy:** {lr_acc:.4f}")
st.markdown(f"- **Precision:** {lr_prec:.4f}")
st.markdown(f"- **Recall:** {lr_rec:.4f}")
st.markdown(f"- **F1-Score:** {lr_f1:.4f}")
st.markdown(f"- **ROC-AUC:** {lr_auc:.4f}")
st.markdown(f"- **Confusion Matrix:**\n```\n{lr_cm}\n```")