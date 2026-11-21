# Imports
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
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
    I have chosen to do further research before committing to anything.
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
    I have chosen to use the AI recommendations for visualization to further my analysis.
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

st.header("Machine Learning: Predicting Game Success")
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

# Drop known leakages, identifiers, and non-numeric columns that aren't encoded
drop_cols = ['AppID', 'Name', 'Release date', 'Estimated owners', 'Success_Label', 'Success', 'Primary_Genre', 'Peak CCU', 'Positive_Ratio', 'Metacritic score', 'min_owners', 'max_owners', 'midpoint_owners', 'Positive_log', 'Negative', 'Positive_capped', 'Positive', 'Recommendations_capped', 'Recommendations', 'Recommendations_log', 'Peak CCU_log', 'Peak CCU_capped', 'Average playtime forever', 'Average playtime forever_capped', 'Average playtime forever_log', 'Average playtime two weeks', 'Median playtime forever', 'Median playtime forever_log', 'Median playtime forever_capped', 'Median playtime two weeks']
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

st.markdown("#### 3. Primary Model: Random Forest Classifier")

rf_model = RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42, class_weight='balanced', n_jobs=-1)
rf_model.fit(X_train, y_train)

rf_preds = rf_model.predict(X_test)
rf_probs = rf_model.predict_proba(X_test)[:, 1]

rf_acc = accuracy_score(y_test, rf_preds)
rf_prec = precision_score(y_test, rf_preds, zero_division=0)
rf_rec = recall_score(y_test, rf_preds, zero_division=0)
rf_f1 = f1_score(y_test, rf_preds, zero_division=0)
rf_auc = roc_auc_score(y_test, rf_probs)
rf_cm = confusion_matrix(y_test, rf_preds)

st.markdown("##### Random Forest Results")
st.markdown(f"- **Accuracy:** {rf_acc:.4f}")
st.markdown(f"- **Precision:** {rf_prec:.4f}")
st.markdown(f"- **Recall:** {rf_rec:.4f}")
st.markdown(f"- **F1-Score:** {rf_f1:.4f}")
st.markdown(f"- **ROC-AUC:** {rf_auc:.4f}")
st.markdown(f"- **Confusion Matrix:**\n```\n{rf_cm}\n```")

st.markdown("#### 4. Top Predictors of Game Success")

# Extract radnom Forest feature importances
rf_importances = pd.DataFrame({
    'Attribute': X.columns,
    'Importance': rf_model.feature_importances_
}).sort_values(by='Importance', ascending=False)

rfc_impotance_dict = dict(zip(rf_importances['Attribute'], rf_importances['Importance']))
rfc_top_15_dict = dict(zip(
    rf_importances['Attribute'].head(15),
    rf_importances['Importance'].head(15)
))

st.markdown("##### Random Forest: Top 15 Most Important Attributes")
st.bar_chart(rf_importances.head(15).set_index('Attribute'))
st.dataframe(rf_importances.head(15), use_container_width=True)

# AI interpretation of machine learning metrics
if "machine" not in st.session_state:
    st.session_state.machine = None

if st.session_state.machine is None:
    prompt = ChatPromptTemplate.from_template(
        f"""
        You are a senior data scientist with extensive machine learning knowledge. You used a random forest classifier to predict the success of video games using binary (1 for Positive, 0 for not). Logistic regression as the basic baseline.:

        Logistic Regression metrics:
        1. Accuracy: {lr_acc:.4f}
        2. Precision: {lr_prec:.4f}
        3. Recall: {lr_rec:.4f}
        4. F1-Score: {lr_f1:.4f}
        5. ROC-AUC: {lr_auc:.4f}
        6. Confusion Matrix: {lr_cm}
        Random Forest Classifier metrics:
        1. Accuracy: {rf_acc:.4f}
        2. Precision: {rf_prec:.4f}
        3. Recall: {rf_rec:.4f}
        4. F1-Score: {rf_f1:.4f}
        5. ROC-AUC: {rf_auc:.4f}
        6. Confusion Matrix: {rf_cm}
        RFC Top 15 Feature Importances:
        "tags_Great Soundtrack":0.03827142634259271
        "Achievements_capped":0.032546648563620086
        "Achievements_log":0.032385427488842135
        "Achievements":0.032092941817818195
        "Price":0.028980537255314895
        "Discount":0.027553649440065428
        "DLC count_capped":0.025345838420674064
        "DLC count":0.023591757719312403
        "DLC count_log":0.022964711388026027
        "tags_Classic":0.016346320302146188
        "Mac":0.014196442086004086
        "tags_Difficult":0.009567346396304594
        "genre_Casual":0.00905834543812068
        "tags_Singleplayer":0.008298006954282379
        "tags_Atmospheric":0.008168877969165834

        Write out an interpretation of your results.
        """
    )

    chain = prompt | llm | StrOutputParser()

    response = chain.invoke({})
    st.session_state.machine = response

st.subheader("AI-generated Machine Learning Results Interpretation")
st.write(st.session_state.machine)
st.subheader("Evaluation of AI-generated Machine Learning Results Interpretation")
st.write(
    """
    The AI interpreted that although the Random Forest accuracy isn't much better than the Logistic Regression *and* its it is not as precise, it has a far better recall and a slightly better ROC-AUC score, making it the more balanced model.
    It also found that games with more content depth, atmospheric appeal, good pricing, and tags that make a wide array of gamer more likely to play are the most important things when determining a games success before release.
    I agree with the AI that from what the RFC and LR have shown, these features appear to hold the most importance.
    I plan to use this information in my final conclusion.
    """
)

st.header("Visualizations")

# Set seaborn aesthetic
sns.set_theme(style="whitegrid")

# Ensure cleaned_data is available
if 'cleaned_data' in locals() or 'cleaned_data' in globals():
    df = cleaned_data.copy()
else:
    st.error("The `cleaned_data` DataFrame is not found in the environment.")
    st.stop()

# ==========================================
# A. Correlation Heatmap across all Numeric Attributes
# ==========================================
st.subheader("A. Correlation Heatmap")
st.markdown(
    """
    **Interpretation:** This heatmap shows the linear relationships between important numerical attributes.
    Noticeable strong correlations usually appear between review metrics (`Positive`, `Negative`, `Recommendations`) and ownership estimates (`min_owners`, `max_owners`, `midpoint_owners`), indicating that highly reviewed games tend to reach wider audiences.
    """
)
corr_columns = [
    'Price', 'Peak CCU', 'DLC count', 'Achievements', 
    'Positive', 'Negative', 'Recommendations', 
    'Average playtime forever', 'Median playtime forever', 
    'min_owners', 'max_owners', 'midpoint_owners'
]

# Check existing columns
available_corr_cols = [col for col in corr_columns if col in df.columns]

if len(available_corr_cols) > 1:
    fig, ax = plt.subplots(figsize=(10, 8))
    corr_matrix = df[available_corr_cols].corr()
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', ax=ax, cbar_kws={'label': 'Pearson Correlation'})
    plt.title("Correlation Matrix Heatmap", fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
else:
    st.warning("Insufficient columns available for the correlation heatmap.")


# ==========================================
# B. Histograms (Distribution Analysis)
# ==========================================
st.subheader("B. Histograms (Distribution Analysis with Logarithmic Scaling)")
st.markdown(
    """
    **Interpretation:** Game metrics on Steam tend to show severe right-skewness.
    By applying logarithmic scaling, the trends of the log-transformed values for metrics like `Price`, `Peak CCU`, `midpoint_owners`, and `Achievements` can be observed, revealing that the majority of games cluster near lower engagement, with a long tail due to massive successes.
    """
)
hist_cols = ['Price', 'Peak CCU', 'midpoint_owners', 'Achievements']
valid_hist_cols = [col for col in hist_cols if col in cleaned_data.columns]

log_x_cols = {'Peak CCU', 'midpoint_owners', 'Achievements'}

if valid_hist_cols:
    fig, axes = plt.subplots(len(valid_hist_cols), 1, figsize=(8, 4 * len(valid_hist_cols)), dpi=150)
    if len(valid_hist_cols) == 1:
        axes = [axes]

    for i, col in enumerate(valid_hist_cols):
        ax = axes[i]

        # Adding small constant for log scale if zeros exist
        has_zeros = (cleaned_data[col].min() == 0)
        data_to_plot = cleaned_data[col] + (1 if has_zeros else 0)
        use_log_x = col in log_x_cols

        sns.histplot(data_to_plot, bins=40, kde=False, log_scale=(use_log_x, False), ax=ax, color='teal', edgecolor='white', linewidth=0.5, alpha=0.85)

        ax.set_yscale('log')
        ax.set_ylim(bottom=0.8)
        ax.yaxis.set_major_formatter(ticker.EngFormatter())

        if col == 'Price':
            ax.xaxis.set_major_formatter(ticker.StrMethodFormatter('${x:,.0f}'))
        else:
            ax.xaxis.set_major_formatter(ticker.EngFormatter())

        sns.despine(ax=ax)
        ax.grid(True, which='major', linestyle='--', linewidth=0.6, alpha=0.5)
        ax.grid(False, which='minor')
        
        axes[i].set_title(f"Log-Scale Distribution of {col}")
        axes[i].set_xlabel(f"{col} (+1 shifted)" if (has_zeros and use_log_x) else col)
        axes[i].set_ylabel("Count (Log)")

    plt.tight_layout()
    st.pyplot(fig)
else:
    st.info("Required columns for histograms are missing.")


# ==========================================
# C. Box Plots (Outlier Detection & Spread)
# ==========================================
st.header("C. Box Plots (Outlier Detection & Spread)")
st.markdown(
    """
    **Interpretation:** Box plots reveal extreme outliers for `Price`, `Peak CCU`, and `DLC count`. 
    The interquartile ranges (IQR) for these variables surround the lower end, while extreme outliers tend to be lofty values, reflecting viral hits and major platform expansions.
    """
)
box_columns = ['Price', 'Peak CCU', 'DLC count']
available_box_cols = [col for col in box_columns if col in df.columns]

if available_box_cols:
    fig, axes = plt.subplots(1, len(available_box_cols), figsize=(5 * len(available_box_cols), 5))
    if len(available_box_cols) == 1:
        axes = [axes]

    for i, col in enumerate(available_box_cols):
        sns.boxplot(y=df[col], ax=axes[i], color='lightcoral')
        axes[i].set_title(f'Box Plot of {col}', fontsize=12)
        axes[i].set_ylabel(col)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
else:
    st.warning("None of the specified box plot columns are present in the dataset.")


# ==========================================
# D. Scatter Plots (Bivariate Relationships)
# ==========================================
st.header("D. Scatter Plots (Bivariate Relationships)")
st.markdown(
    """
    **Interpretation:** For `Price` vs. `midpoint_owners`, points cluster around lower prices, scattering as the cost increases.
    For `Positive` vs. `Negative` reviews, there appears to be a strong positive linear correlation between the two.
    Therefore, games with higher engagement get reviewed more, resulting in a kind of balance between positive and negative reviews.
    """
)
# 1. Price vs midpoint_owners
if 'Price' in df.columns and 'midpoint_owners' in df.columns:
    fig, ax = plt.subplots(figsize=(9, 6))

    # Add slight random vertical litter in log10 space
    rng = np.random.default_rng(42)
    log_y = np.log10(df['midpoint_owners'])
    jittered_y = 10 ** (log_y + rng.uniform(-0.08, 0.08, size=len(df)))

    ax.scatter(df['Price'], jittered_y, alpha=0.15, color='teal', s=12, edgecolors='none')

    ax.set_yscale('log') # Log scale recommended due to high skewness in owners
    ax.xaxis.set_major_formatter(ticker.StrMethodFormatter('${x:,.0f}'))
    ax.yaxis.set_major_formatter(ticker.EngFormatter())

    sns.despine(ax=ax)
    ax.grid(True, which='major', linestyle='--', linewidth=0.6, alpha=0.5)
    ax.grid(False, which='minor')

    ax.set_title('Price vs. Midpoint Owners')
    ax.set_xlabel('Price ($)')
    ax.set_ylabel('Midpoint Owners')

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
else:
    st.warning("Price or midpoint_owners column missing for scatter plot.")

# 2. Positive reviews vs Negative reviews
if 'Positive' in df.columns and 'Negative' in df.columns:
    fig, ax = plt.subplots(figsize=(8, 6))

    sns.scatterplot(data=df, x='Positive', y='Negative', alpha=0.5, ax=ax, color='purple')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.xaxis.set_major_formatter(ticker.EngFormatter())
    ax.yaxis.set_major_formatter(ticker.EngFormatter())
    
    ax.set_title('Positive Reviews vs. Negative Reviews', fontsize=14)
    ax.set_xlabel('Positive Reviews')
    ax.set_ylabel('Negative Reviews')

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
else:
    st.warning("Positive or Negative review columns missing for scatter plot.")


# ==========================================
# E. Release-Year Trend (Temporal Analysis)
# ==========================================
st.header("E. Temporal Analysis: Release-Year Trends")
st.markdown(
    """
    **Interpretation:** By deriving the `Release Year` from the `Release date` string values, the amount of games released annually can be tracked alongside the `midpoint_owners`.
    After doing some reasearch, it was discover that from 2012 to 1017, Steam published games through commmunity votes under Steam Greenlight.
    However, in 2017, Steam changed this to self-publishing with Steam Direct. That is why the number of games released increases more after 2017. More on this phenomenon can be found on the [Game Developer website](https://www.gamedeveloper.com/business/the-three-ages-of-steam-distribution-and-how-it-changes-indie-dev-).
    There are also major spikes in ownership games released before 2005, which is a year also mentioned in the Game Developer article.
    In 2005, Steam began exdanding to third party games, after having a majority of only Valve's game releases.
    Because their were only a handful of titles to choose from, Steam's users ownership was more concentrated. 
    Currently, the game market is heavily saturated, with a widespread of both big name and niche indie projects to play.
    """
)
st.markdown("Evaluates platform saturation (game releases per year) alongside average midpoint owners over time.")

if 'Release date' in cleaned_data.columns and 'AppID' in cleaned_data.columns and 'midpoint_owners' in cleaned_data.columns:
    # Derive Release Year variable from Release date string format ex. "Aug 20, 2023"
    temp_df = cleaned_data.copy()
    temp_df['Release Year'] = pd.to_datetime(temp_df['Release date'], errors='coerce').dt.year

    # Filter out invalid or extreme years
    temp_df = temp_df.dropna(subset=['Release Year'])
    temp_df['Release Year'] = temp_df['Release Year'].astype(int)
    temp_df = temp_df[(temp_df['Release Year'] >= 1995) & (temp_df['Release Year'] <= 2024)]

    # Aggregate by Year: Count of AppIDs and Mean midpoint_owners
    yearly_agg = temp_df.groupby('Release Year').agg(
        Game_Count=('AppID', 'count'),
        Mean_Owners=('midpoint_owners', 'mean')
    ).reset_index()

    fig_e, ax1 = plt.subplots(figsize=(12, 6))

    # Primary Axis: Game Releases Count
    color = 'tab:blue'
    ax1.set_xlabel('Release Year')
    ax1.set_ylabel('Number of Game Releases', color=color)
    line1 = ax1.plot(yearly_agg['Release Year'], yearly_agg['Game_Count'], color=color, marker='o', linewidth=2, label='Game Releases')
    ax1.tick_params(axis='y', labelcolor=color)

    # Secondary Axis: Average Midpoint Owners
    ax2 = ax1.twinx()
    color = 'tab:orange'
    ax2.set_ylabel('Average Midpoint Owners', color=color)
    line2 = ax2.plot(yearly_agg['Release Year'], yearly_agg['Mean_Owners'], color=color, marker='s', linestyle='--', linewidth=2, label='Avg Midpoint Owners')
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title("Steam Platform Saturation and Average Game Reach Over Time")
    fig_e.tight_layout()
    st.pyplot(fig_e)
else:
    st.warning("Required columns ('Release date', 'AppID', 'midpoint_owners') for temporal analysis are missing.")

st.header("Final conclusions")
st.write(
    """
    In conclusion, while the hypothesis testing failed to reject the null hypothesis that Genre does not affect game success, the machine learning models showed that tags for Great Soundtrack, Atmospheric, Classic, Difficult, Casual, and Singleplayer held some importance when the Random Forest model predicted games success.
    Looking at the different figures and visualization, specifically the Relase Year trends and the heatmap, it is clear that market oversaturation has made it more difficult to break out in the gaming industry.
    Games with more Positive and even more Negative reviews tend to get recommended more, but many of those highly reviewed games are outlier blockbusters which significantly skew the data. 
    The find that was most shocking was how important the tag of Great Soundtrack was for the machine learning model. Users seem to be looking for games they can fully immerse themselves in.
    Gaming is an experience and developers have to take every possible aspect into consideration when creating new games.
    """
)