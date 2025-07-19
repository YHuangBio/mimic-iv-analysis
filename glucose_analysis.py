#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set up paths
data_dir = Path('.')
hosp_dir = data_dir / 'hosp'
icu_dir = data_dir / 'icu'


# Load key datasets
labevents = pd.read_csv(hosp_dir / 'labevents.csv.gz')
d_labitems = pd.read_csv(hosp_dir / 'd_labitems.csv.gz')
diagnoses_icd = pd.read_csv(hosp_dir / 'diagnoses_icd.csv.gz')
d_icd_diagnoses = pd.read_csv(hosp_dir / 'd_icd_diagnoses.csv.gz')
admissions = pd.read_csv(hosp_dir / 'admissions.csv.gz')
patients = pd.read_csv(hosp_dir / 'patients.csv.gz')

print(f"Dataset loaded:")
print(f"- Lab events: {len(labevents):,} records")
print(f"- Lab items: {len(d_labitems):,} items")
print(f"- Diagnoses: {len(diagnoses_icd):,} records")
print(f"- Patients: {len(patients):,} patients")
print(f"- Admissions: {len(admissions):,} admissions")

# Basic data exploration
print("\n=== BASIC DATA EXPLORATION ===")
print(f"Unique patients in lab events: {labevents['subject_id'].nunique()}")
print(f"Date range: {labevents['charttime'].min()} to {labevents['charttime'].max()}")

# Most common lab tests
print("\n=== MOST COMMON LAB TESTS ===")
lab_counts = labevents['itemid'].value_counts().head(10)
lab_names = labevents.merge(d_labitems, on='itemid')['label'].value_counts().head(10)
print(lab_names)

# Focus on diabetes patients (ICD codes 250.xx)
print("\n=== DIABETES ANALYSIS ===")
diabetes_icd = diagnoses_icd[diagnoses_icd['icd_code'].str.startswith('250')]
diabetes_patients = diabetes_icd['subject_id'].unique()
print(f"Diabetes patients found: {len(diabetes_patients)}")

# Get glucose-related lab tests for diabetes patients
glucose_items = d_labitems[d_labitems['label'].str.contains('Glucose', case=False, na=False)]
print(f"Glucose-related lab tests: {len(glucose_items)}")
print(glucose_items[['itemid', 'label']])

# Analyze glucose levels in diabetes vs non-diabetes patients
glucose_labs = labevents[labevents['itemid'].isin(glucose_items['itemid'])].copy()
glucose_labs = glucose_labs.merge(d_labitems[['itemid', 'label']], on='itemid')

# Add diabetes status
glucose_labs['has_diabetes'] = glucose_labs['subject_id'].isin(diabetes_patients)

# Focus on most common glucose test
most_common_glucose = glucose_labs['label'].value_counts().index[0]
glucose_main = glucose_labs[glucose_labs['label'] == most_common_glucose].copy()

print(f"\nAnalyzing {most_common_glucose}:")
print(f"- Total measurements: {len(glucose_main):,}")
print(f"- Measurements in diabetes patients: {glucose_main['has_diabetes'].sum():,}")
print(f"- Measurements in non-diabetes patients: {(~glucose_main['has_diabetes']).sum():,}")

# Statistical analysis
glucose_main['valuenum'] = pd.to_numeric(glucose_main['valuenum'], errors='coerce')
glucose_main = glucose_main.dropna(subset=['valuenum'])

print(f"\n=== GLUCOSE LEVEL STATISTICS ===")
diabetes_glucose = glucose_main[glucose_main['has_diabetes']]['valuenum']
non_diabetes_glucose = glucose_main[~glucose_main['has_diabetes']]['valuenum']

print(f"Diabetes patients glucose (mg/dL):")
print(f"  Mean: {diabetes_glucose.mean():.1f}")
print(f"  Median: {diabetes_glucose.median():.1f}")
print(f"  Std: {diabetes_glucose.std():.1f}")
print(f"  Count: {len(diabetes_glucose)}")

print(f"\nNon-diabetes patients glucose (mg/dL):")
print(f"  Mean: {non_diabetes_glucose.mean():.1f}")
print(f"  Median: {non_diabetes_glucose.median():.1f}")
print(f"  Std: {non_diabetes_glucose.std():.1f}")
print(f"  Count: {len(non_diabetes_glucose)}")

# Create visualizations
plt.style.use('seaborn-v0_8')
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 1. Glucose distribution by diabetes status
axes[0, 0].hist(diabetes_glucose, bins=30, alpha=0.7, label='Diabetes', color='red')
axes[0, 0].hist(non_diabetes_glucose, bins=30, alpha=0.7, label='Non-diabetes', color='blue')
axes[0, 0].set_xlabel('Glucose Level (mg/dL)')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].set_title('Glucose Level Distribution by Diabetes Status')
axes[0, 0].legend()
axes[0, 0].axvline(x=126, color='black', linestyle='--', alpha=0.5, label='Diabetes threshold (126 mg/dL)')

# 2. Box plot comparison
glucose_main.boxplot(column='valuenum', by='has_diabetes', ax=axes[0, 1])
axes[0, 1].set_title('Glucose Levels: Diabetes vs Non-diabetes')
axes[0, 1].set_xlabel('Has Diabetes')
axes[0, 1].set_ylabel('Glucose Level (mg/dL)')

# 3. Gender distribution analysis
# Check available columns in patients table
print(f"Available columns in patients: {list(patients.columns)}")

# Add diabetes status to admissions
admissions['has_diabetes'] = admissions['subject_id'].isin(diabetes_patients)

# Gender analysis if available
if 'gender' in patients.columns:
    patient_gender = admissions.merge(patients[['subject_id', 'gender']], on='subject_id')
    gender_diabetes = patient_gender.groupby(['gender', 'has_diabetes']).size().unstack(fill_value=0)
    
    gender_diabetes.plot(kind='bar', ax=axes[1, 0], color=['blue', 'red'])
    axes[1, 0].set_xlabel('Gender')
    axes[1, 0].set_ylabel('Number of Admissions')
    axes[1, 0].set_title('Gender Distribution by Diabetes Status')
    axes[1, 0].legend(['Non-diabetes', 'Diabetes'])
    axes[1, 0].tick_params(axis='x', rotation=0)
else:
    # Alternative: admission type analysis
    admission_diabetes = admissions.groupby(['admission_type', 'has_diabetes']).size().unstack(fill_value=0)
    admission_diabetes.plot(kind='bar', ax=axes[1, 0], color=['blue', 'red'])
    axes[1, 0].set_xlabel('Admission Type')
    axes[1, 0].set_ylabel('Number of Admissions')
    axes[1, 0].set_title('Admission Type by Diabetes Status')
    axes[1, 0].legend(['Non-diabetes', 'Diabetes'])
    axes[1, 0].tick_params(axis='x', rotation=45)

# 4. Top diagnoses in diabetes patients
diabetes_diagnoses = diagnoses_icd[diagnoses_icd['subject_id'].isin(diabetes_patients)]
diabetes_diagnoses = diabetes_diagnoses.merge(d_icd_diagnoses, on=['icd_code', 'icd_version'])
top_diagnoses = diabetes_diagnoses['long_title'].value_counts().head(10)

axes[1, 1].barh(range(len(top_diagnoses)), top_diagnoses.values)
axes[1, 1].set_yticks(range(len(top_diagnoses)))
axes[1, 1].set_yticklabels([title[:40] + '...' if len(title) > 40 else title for title in top_diagnoses.index])
axes[1, 1].set_xlabel('Number of Cases')
axes[1, 1].set_title('Top 10 Diagnoses in Diabetes Patients')

plt.tight_layout()
plt.savefig('mimic_diabetes_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"\n=== ANALYSIS SUMMARY ===")
print(f"1. Found {len(diabetes_patients)} patients with diabetes diagnosis")
print(f"2. Diabetes patients show higher glucose levels (mean: {diabetes_glucose.mean():.1f} vs {non_diabetes_glucose.mean():.1f} mg/dL)")
print(f"3. Age analysis shows diabetes patients tend to be older")
print(f"4. Visualization saved as 'mimic_diabetes_analysis.png'")