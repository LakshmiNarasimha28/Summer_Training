# Walmart Sales Data Analysis

This project provides an end-to-end data preprocessing, feature engineering, and exploratory data analysis (EDA) solution on Walmart retail transaction data. It helps extract business insights such as revenue trends, popular product categories, customer buying behavior, and profit margin patterns.


## Dataset

- **Source**: [Walmart 10K Sales Dataset (Kaggle)](https://www.kaggle.com/najir0123/walmart-10k-sales-datasets)
- **Files**:
  - `Walmart_Sales_Data.csv` (Raw data)
  - `Walmart_cleaned_Sales_Data.csv` (Cleaned data)


## Tools & Technologies

- **Python** (Jupyter Notebook)
- **MySQL** (for structured data storage and querying)
- **Libraries**: `pandas`, `matplotlib`, `seaborn`, `mysql.connector`



## Project Workflow

### 1. Data Loading
- Loaded raw `.csv` file into a Pandas DataFrame.
- Created MySQL database and table schema (`st_project_1.walmart_dataset`).

### 2. Data Cleaning
- Removed currency symbols from `unit_price`.
- Parsed `date` and `time` fields to proper datetime formats.
- Dropped 31 rows with missing `unit_price` and `quantity`.
- Verified column types, null values, and duplicates.

### 3. Feature Engineering
- Created `total_amount` = `unit_price * quantity`.
- Saved final cleaned dataset to `Walmart_cleaned_Sales_Data.csv`.

### 4. EDA & Visualization
- Revenue and quantity trends across:
  - **Branches**
  - **Categories**
  - **Cities**
  - **Hours of day**
  - **Payment Methods**
- Analyzed **profit margins** by branch and category.
- Used clean bar plots and line charts (Seaborn + Matplotlib).


## Key Insights

- **Branch B** generated the highest revenue overall.
- **Electronic accessories** and **Food and beverages** were top-selling categories.
- Peak customer traffic observed between **13:00–15:00 hours**.
- **Credit card** was the most used payment method.
- **Branch A** had the highest average profit margin.

## How to Run

**Clone the repository**
- git clone https://github.com/yourusername/walmart-sales-analysis.git

**Install required libraries**
- pip install pandas matplotlib seaborn mysql-connector-python

**Launch Jupyter Notebook**
- jupyter notebook
**Open and run walmart_analysis.ipynb**
