# Movie Data Analysis and Power BI Dashboard

## Overview

This project focuses on cleaning, transforming, and analyzing a movie dataset using Python, followed by the development of an interactive Power BI dashboard to explore trends and insights related to movies and TV series.

## Objectives

* Clean and preprocess raw movie data.
* Handle missing values and inconsistencies.
* Transform and prepare data for analysis.
* Generate descriptive statistics and insights.
* Develop an interactive Power BI dashboard for visualization and decision-making.

## Dataset Features

The dataset contains information such as:

* Movie/TV Series Title
* Year
* Genre
* Runtime
* Rating
* Votes
* Gross Revenue


## Data Cleaning and Preprocessing

The following data preparation steps were performed using Python:

* Removed duplicate records.
* Extracted and standardized year information.
* Created and refined content type categories.
* Handled missing values using appropriate imputation techniques.
* Split multiple genres into separate genre fields.
* Converted variables into suitable data types.
* Split year range to start year and end year to identify movie type.
* Generated summary statistics for exploratory analysis.

## Tools and Technologies

* Python
  *pandas
  *numpy
  *scipy
  *scikit-learn
* Power BI
* Jupyter Notebook / VS Code

## Dashboard Features

The Power BI dashboard provides:

* Total titles by category
* Rating analysis by content type
* Runtime comparisons across content types
* Audience engagement analysis using votes
* Genre distribution and popularity
* Interactive filtering and drill-down capabilities

## Key Insights

* Comparison of ratings across different content types.
* Identification of the most popular genres.
* Analysis of audience engagement through vote counts.
* Runtime patterns among movies and TV series.

## Repository Structure

```
├── datasets/
│   └── movies.csv
│   └── movies_cleaned.csv
├── scripts/
│   └── data_cleaning.py
├── power_bi/
|   └── Movie_dashboard.pbix
└── README.md
```

## Getting Started

1. Clone the repository:

```bash
git clone <repository-url>
```

2. Install required packages:

```bash
pip install pandas numpy 
```

3. Run the data cleaning script.

4. Open the Power BI dashboard file (`.pbix`) to explore the visualizations.

## Author

R.I.V. Jayasekera


