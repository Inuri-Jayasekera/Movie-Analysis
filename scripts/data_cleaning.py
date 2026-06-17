
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder
from scipy.stats import skew, mode
import re
import numpy as np
import pandas as pd

df = pd.read_csv("data/movies.csv")
df.info()

#convert data type of year to string
df['YEAR'] = df['YEAR'].astype(str)

#fix Hidden Unicode or formatting issues
df['MOVIES'] = df['MOVIES'].str.normalize('NFKC')

#missing rates
missing_rate = df.isnull().mean() * 100
print(missing_rate)

#convert data type of year,genre,one-line,stars,gross to string
for col in ['MOVIES', 'YEAR', 'GENRE', 'ONE-LINE', 'STARS','Gross' ]:
    df[col] = (
        df[col]
        .astype(str)
        .str.strip()          # remove leading/trailing spaces
        .str.replace(r'\s+', ' ', regex=True)  # collapse extra spaces
        .str.replace('\n', '', regex=False)    # remove newlines
    )

#checking duplicates in the dataset
df.duplicated().sum()

#drop duplicates
df = df.reset_index(drop=True)




df = df.drop_duplicates(subset=['MOVIES', 'YEAR'], keep='first')

df.info()

#missing rates
missing_rate = df.isnull().mean() * 100
print(missing_rate)

df

#spliting year range to start year and end year



# remove parentheses and trim spaces
df['YEAR_clean'] = df['YEAR'].str.replace(r"[()]", "", regex=True).str.strip()

# extract all 4-digit numbers (list)
df['year_nums'] = df['YEAR_clean'].apply(lambda x: re.findall(r"\d{4}", str(x)))

#  extract first and second 4-digit numbers (if exist)
df['YEAR_start'] = df['year_nums'].apply(lambda x: int(x[0]) if len(x) >= 1 else np.nan)
df['YEAR_end']   = df['year_nums'].apply(lambda x: int(x[1]) if len(x) >= 2 else np.nan)

#  handle special case: starts with a dash (-2017, –2017, etc.)
def handle_dash_case(row):
    if re.match(r"^[–-]\s*\d{4}", str(row['YEAR_clean'])):
        return pd.Series([np.nan, int(re.search(r"\d{4}$", row['YEAR_clean']).group())])
    else:
        return pd.Series([row['YEAR_start'], row['YEAR_end']])

df[['YEAR_start', 'YEAR_end']] = df.apply(handle_dash_case, axis=1)

#  remove temporary columns
df = df.drop(columns=['YEAR_clean', 'year_nums'])

df.info()

df.head()

# create type variable with categories Movie and TV Series
for i in range(len(df)):
    year_start = df.iloc[i]["YEAR_start"]
    year_end = df.iloc[i]["YEAR_end"]

    if (pd.notna(year_start) and pd.notna(year_end)) or (pd.notna(year_start) and pd.isna(year_end)):
        df.loc[i, "type"] = "TV Series"
    elif pd.isna(year_start) and pd.notna(year_end):
        df.loc[i, "type"] = "Movie"
    elif pd.isna(year_start) and pd.isna(year_end):
        df.loc[i, "type"] = pd.NA

#create word_from_year variable
def extract_words(text):
    if pd.isna(text):
        return None
    words = re.findall(r"[A-Za-z]+", str(text))
    return " ".join(words) if words else None

df["word_from_year"] = df["YEAR"].apply(extract_words)

# other type categories

mapping = {
    "TV Special": "TV Special",
    "TV Movie": "TV Movie",
    "Video": "Video",
    "TV Short": "TV Short",
    "Video Game": "Video Game",
    "II TV Movie": "TV Movie"
}

df['type'] = df.apply(
    lambda row: (
        mapping[row['word_from_year']]
        if pd.notna(row['word_from_year']) and row['word_from_year'] in mapping
        else row['type']
    ),
    axis=1
)

#count of titles for each type category
df['type'].value_counts(dropna=False)

missing_rate_before_impute = df.isnull().mean() * 100
print(missing_rate_before_impute)

df["word_from_year"].unique()

df[df["type"]=='Video Game']

df_nan_type = df[df['type'].isna()]
df_nan_type

# Convert 'VOTES' to numeric, handling commas
df['VOTES'] = df['VOTES'].astype(str).str.replace(',', '', regex=False)
df['VOTES'] = pd.to_numeric(df['VOTES'], errors='coerce')


#summary statistics


# Example numeric columns
numeric_cols = ['RATING', 'VOTES', 'RunTime']

# Define a function to calculate summary stats for each group
def summary_stats(x):
    return pd.Series({
        'count': x.count(),
        'mean': x.mean(),
        'median': x.median(),
        'mode': x.mode().iloc[0] if not x.mode().empty else None,
        'min': x.min(),
        'max': x.max(),
        'skewness': skew(x.dropna())
    })

# Apply the function to each numeric column grouped by 'type'
summary_table = (
    df
    .groupby('type')[numeric_cols]
    .apply(lambda df: df.apply(summary_stats))
)

# To display neatly
pd.set_option('display.float_format', '{:.2f}'.format)
print(summary_table)

#gross drop
df = df.drop(columns='Gross')

#missing rates before imputations
missing_rate_before_impute = df.isnull().mean() * 100
print(missing_rate_before_impute)

#Genre imputing
df['GENRE'] = df['GENRE'].replace('nan', np.nan).fillna('UNKNOWN')

# Split the 'GENRE' column into up to genre1,genre2, genre3
genre_split = df['GENRE'].str.split(',', n=2, expand=True)

# Assign new column names
genre_split.columns = ['GENRE_1', 'GENRE_2', 'GENRE_3']

# Add these new columns to your main dataframe
df = pd.concat([df, genre_split], axis=1)

pd.crosstab(df['type'].isna(), df['RunTime'].isna())

#impute runtime





def get_mode(series):
    mode = series.mode()
    return mode.iloc[0] if not mode.empty else np.nan

# Only impute RunTime where type is NOT missing
df.loc[df['type'].notna(), 'RunTime'] = (
    df[df['type'].notna()]
    .groupby('type')['RunTime']
    .transform(lambda x: x.fillna(get_mode(x)))
)

pd.crosstab(df['type'].isna(), df['RunTime'].isna())

df[ df['RunTime'].isna() & df['VOTES'].isna() & df['RATING'].isna()]

#impute type by KNN imputation

# Make a copy to avoid changing the original data
df_knn = df.copy()

# Keep only rows where RunTime is NOT missing
df_knn = df_knn[df_knn['RunTime'].notna()].copy()

# Encode 'type' since KNNImputer works with numeric data
le = LabelEncoder()
mask_type_notna = df_knn['type'].notna()
df_knn.loc[mask_type_notna, 'type_encoded'] = le.fit_transform(df_knn.loc[mask_type_notna, 'type'])

# Prepare the dataset for KNN imputation
imputer_df = df_knn[['RunTime', 'type_encoded']].apply(pd.to_numeric, errors='coerce')

# Apply KNN imputation (using RunTime to estimate missing type)
imputer = KNNImputer(n_neighbors=5)
imputed_array = imputer.fit_transform(imputer_df)

# Replace imputed values
df_knn['type_encoded'] = imputed_array[:, 1].round().astype(int)

# Decode numeric labels back to original type names
df_knn['type_imputed'] = le.inverse_transform(df_knn['type_encoded'])

# Update only missing type rows in the original dataframe (where RunTime is not missing)
df.loc[df_knn.index[df_knn['type'].isna()], 'type'] = df_knn.loc[df_knn['type'].isna(), 'type_imputed']

pd.crosstab(df['type'].isna(), df['RunTime'].isna())

pd.crosstab(df['type'].isna(), df['RATING'].isna())

# Impute missing RATING values by the median within each 'type' group
df['RATING'] = df.groupby('type')['RATING'].transform(
    lambda x: x.fillna(x.median())
)

pd.crosstab(df['type'].isna(), df['RATING'].isna())

pd.crosstab(df['type'].isna(), df['VOTES'].isna())

# Impute missing VOTES values by the median within each 'type' group
df['VOTES'] = df.groupby('type')['VOTES'].transform(
    lambda x: x.fillna(x.median())
)

pd.crosstab(df['type'].isna(), df['VOTES'].isna())

#missing rates after imputations
missing_rate_after_impute = df.isnull().mean() * 100
print(missing_rate_after_impute)

#remove unwanted columns
df = df.drop(columns=['word_from_year'])

df

df.loc[df['YEAR_start'].isna(), 'YEAR_start'] = df.loc[df['YEAR_start'].isna(), 'YEAR_end']

df

df.to_csv("data/movies_cleaned.csv", index=False)

