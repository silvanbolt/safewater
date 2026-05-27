import pandas as pd
import numpy as np


# =========================
# Configuration
# =========================
# paths
CSV_PATH = "output/pre-processing/raw_data_plus_paths.csv"
FILTER_PATH = "input/pre-processing/column_filter.csv"
OUTPUT_PATH = "output/pre-processing/clean_dataset.csv"
LABEL_MAP_PATH = "input/pre-processing/label_map.csv"
# flags
drop_unknown = True


# =========================
# Load data
# =========================

# read inputs
df = pd.read_csv(CSV_PATH)
col_filt = pd.read_csv(FILTER_PATH)
label_map = pd.read_csv(LABEL_MAP_PATH)


# =========================
# Feature selection and engineering
# =========================

# select columns of interest
df = df[list(col_filt.keep)]

# delete # symbol in column names
df.columns = df.columns.str.replace('#', '', regex=False)

# convert install_year into full date (assume installment at the beginning of the year)
df['install_date'] = pd.to_datetime(df['install_year'], format='%Y', errors='coerce') 
# convert report date to datetime object
df['report_date'] = pd.to_datetime(df['report_date'])
# calculate time since installment
df['age_years'] = (df['report_date'] - df['install_date']).dt.days / 365.25
# set negative age to NA
df.loc[df['age_years'] < 0, 'age_years'] = np.nan
# drop install_year
df = df.drop(columns=['install_year'])

# convert rehab_year into datetime (assume rehabilitation at the beginnning of the year)
df['rehab_date'] = pd.to_datetime(
    df['rehab_year']
    .astype(str)
    .str.replace(',', '', regex=False),
    format='%Y',
    errors='coerce'
)
# calculate time since last rehabilitation
df['time_since_rehab'] = (df['rehab_date'] - df['install_date']).dt.days / 365.25
# set negative time to NA
df.loc[df['time_since_rehab'] < 0, 'time_since_rehab'] = np.nan
# drop columns
df = df.drop(columns=['rehab_year', 'rehab_date', 'install_date', 'report_date'])


# =========================
# Create target features
# =========================

# map water_source_clean column to improved/nonimproved 
df = df.merge(
    label_map[["water_source_clean", "improved"]],
    on="water_source_clean",
    how="left"
)
# map NAs in improved/nonimproved category to unknown 
df["improved"] = df["improved"].fillna("Unknown")

# drop unknown improved/nonimproved category
if drop_unknown :
    df = df[df.improved != 'Unknown']

# water source is the label
df.rename(columns={"water_source_clean": "label"}, inplace=True)
df['label'] = df.pop('label')



# =========================
# Write output
# =========================

# save result
df.to_csv(OUTPUT_PATH, index=False)


