#!/usr/bin/env python3
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
original_stdout = sys.stdout

# Set up paths
data_dir = Path('.')
hosp_dir = data_dir / 'hosp'
icu_dir = data_dir / 'icu'

# Load key datasets
d_labitems = pd.read_csv(hosp_dir / 'd_labitems.csv.gz')
prescriptions = pd.read_csv(hosp_dir / 'prescriptions.csv.gz')
diagnoses_icd = pd.read_csv(hosp_dir / 'diagnoses_icd.csv.gz')
d_icd_diagnoses = pd.read_csv(hosp_dir / 'd_icd_diagnoses.csv.gz')

with open("explor_output.txt", "w", encoding="utf-8") as f:
    sys.stdout = f
    print(f"Loaded {len(d_labitems)} lab items")
    print(f"Loaded {len(prescriptions)} prescriptions")
    print(f"Loaded {len(diagnoses_icd)} diagnoses")
    print(f"Loaded {len(d_icd_diagnoses)} ICD diagnosis definitions")


    # 1. SYSTEMATIC EXPLORATION OF LAB ITEMS
    print("\n" + "="*60)
    print("1. SYSTEMATIC EXPLORATION OF LAB ITEMS")
    print("="*60)
    print(f"\nTotal unique lab items: {len(d_labitems)}")
    print("\n" + "-"*40)
    print("LAB CATEGORIES:")
    print(d_labitems['category'].value_counts())
    print("\n" + "-"*40)
    print("POTENTIAL DIABETES-RELATED LABS:")

    # Search for potentially diabetes-related terms in actual lab names
    potential_diabetes_terms = [
        'glucose', 'sugar', 'hba1c', 'hemoglobin', 'insulin', 'c-peptide',
        'cholesterol', 'triglyceride', 'hdl', 'ldl', 'lipid',
        'creatinine', 'urea', 'bun', 'albumin', 'protein',
        'ketone', 'lactate', 'anion', 'gap', 'osmolality'
    ]

    found_diabetes_labs = {}
    for term in potential_diabetes_terms:
        matches = d_labitems[d_labitems['label'].str.contains(term, case=False, na=False)]
        if len(matches) > 0:
            found_diabetes_labs[term] = matches
            print(f"\n'{term.upper()}' found in {len(matches)} lab items:")
            for _, row in matches.iterrows():
                print(f"  {row['itemid']}: {row['label']} ({row['category']})")

    # 2. SYSTEMATIC EXPLORATION OF MEDICATIONS
    print("\n" + "="*60)
    print("2. SYSTEMATIC EXPLORATION OF ALL MEDICATIONS")
    print("="*60)

    print(f"\nTotal prescriptions: {len(prescriptions)}")
    print(f"Unique medications: {prescriptions['drug'].nunique()}")

    print("\nAll unique medications (first 50):")
    unique_drugs = prescriptions['drug'].unique()
    for i, drug in enumerate(sorted(unique_drugs)[:50]):
        print(f"{i+1:2d}. {drug}")

    if len(unique_drugs) > 50:
        print(f"... and {len(unique_drugs) - 50} more medications")

    print("\n" + "-"*40)
    print("POTENTIAL DIABETES-RELATED MEDICATIONS:")

    # Search for potentially diabetes-related medications in actual drug names
    potential_diabetes_drugs = [
        'insulin', 'metformin', 'glipizide', 'glyburide', 'glimepiride',
        'pioglitazone', 'rosiglitazone', 'sitagliptin', 'exenatide',
        'liraglutide', 'canagliflozin', 'empagliflozin', 'dapagliflozin',
        'acarbose', 'miglitol', 'repaglinide', 'nateglinide'
    ]

    found_diabetes_meds = {}
    for term in potential_diabetes_drugs:
        matches = prescriptions[prescriptions['drug'].str.contains(term, case=False, na=False)]
        if len(matches) > 0:
            found_diabetes_meds[term] = matches
            print(f"\n'{term.upper()}' found in {len(matches)} prescriptions:")
            unique_drug_names = matches['drug'].unique()
            for drug_name in sorted(unique_drug_names)[:10]:  # Show first 10 variations
                count = len(matches[matches['drug'] == drug_name])
                print(f"  {drug_name} ({count} prescriptions)")
            if len(unique_drug_names) > 10:
                print(f"  ... and {len(unique_drug_names) - 10} more variations")

    # Also search for any drug containing "diabetes" or "diabetic"
    diabetes_named_meds = prescriptions[prescriptions['drug'].str.contains('diabet', case=False, na=False)]
    if len(diabetes_named_meds) > 0:
        print(f"\nMEDICATIONS WITH 'DIABETES' IN NAME:")
        for drug_name in diabetes_named_meds['drug'].unique():
            count = len(diabetes_named_meds[diabetes_named_meds['drug'] == drug_name])
            print(f"  {drug_name} ({count} prescriptions)")

    # 3. SYSTEMATIC EXPLORATION OF DIAGNOSIS CODES
    print("\n" + "="*60)
    print("3. SYSTEMATIC EXPLORATION OF ALL DIAGNOSIS CODES")
    print("="*60)

    print(f"\nTotal diagnoses: {len(diagnoses_icd)}")
    print(f"Unique ICD codes: {diagnoses_icd['icd_code'].nunique()}")

    # Look at diabetes-related ICD codes
    diabetes_codes = diagnoses_icd[diagnoses_icd['icd_code'].str.contains('^25[0-9]', case=False, na=False, regex=True)]
    print(f"\nDIABETES-RELATED ICD CODES (250.x):")
    print(f"Found {len(diabetes_codes)} diagnosis records with diabetes codes")

    diabetes_code_summary = diabetes_codes.groupby('icd_code').size().sort_values(ascending=False)
    print("\nDiabetes code frequencies:")
    for code, count in diabetes_code_summary.head(20).items():
        # Get the description
        desc = d_icd_diagnoses[d_icd_diagnoses['icd_code'] == code]['long_title'].values
        desc_text = desc[0] if len(desc) > 0 else "No description"
        print(f"  {code}: {count} cases - {desc_text}")

    # Look for other potentially diabetes-related diagnoses
    diabetes_keywords = ['diabetes', 'diabetic', 'hyperglycemia', 'hypoglycemia', 'ketoacidosis']
    print(f"\nOTHER DIABETES-RELATED DIAGNOSES:")

    for keyword in diabetes_keywords:
        matches = d_icd_diagnoses[d_icd_diagnoses['long_title'].str.contains(keyword, case=False, na=False)]
        if len(matches) > 0:
            print(f"\n'{keyword.upper()}' found in {len(matches)} diagnosis descriptions:")
            for _, row in matches.head(10).iterrows():
                # Check if this code appears in our diagnoses
                code_count = len(diagnoses_icd[diagnoses_icd['icd_code'] == row['icd_code']])
                print(f"  {row['icd_code']}: {row['long_title']} ({code_count} cases)")

    # 4. CROSS-REFERENCE WITH ACTUAL DIABETES PATIENTS
    print("\n" + "="*60)
    print("4. CROSS-REFERENCE WITH ACTUAL DIABETES PATIENTS")
    print("="*60)

    # Get diabetes patients
    diabetes_patients = diagnoses_icd[diagnoses_icd['icd_code'].str.startswith('250')]['subject_id'].unique()
    print(f"\nIdentified {len(diabetes_patients)} diabetes patients")

    # Load lab events to see what labs were actually ordered for diabetes patients
    print("\nLoading lab events to see what was actually ordered for diabetes patients...")
    labevents = pd.read_csv(hosp_dir / 'labevents.csv.gz')

    # Get labs ordered for diabetes patients
    diabetes_labs = labevents[labevents['subject_id'].isin(diabetes_patients)]
    diabetes_lab_items = diabetes_labs['itemid'].value_counts().head(20)

    print("\nMOST COMMON LAB TESTS ORDERED FOR DIABETES PATIENTS:")
    for itemid, count in diabetes_lab_items.items():
        lab_name = d_labitems[d_labitems['itemid'] == itemid]['label'].values
        lab_name = lab_name[0] if len(lab_name) > 0 else "Unknown"
        print(f"  {itemid}: {lab_name} ({count} tests)")

    # Get medications prescribed to diabetes patients
    diabetes_meds = prescriptions[prescriptions['subject_id'].isin(diabetes_patients)]
    diabetes_drug_counts = diabetes_meds['drug'].value_counts().head(20)

    print("\nMOST COMMON MEDICATIONS PRESCRIBED TO DIABETES PATIENTS:")
    for drug, count in diabetes_drug_counts.items():
        print(f"  {drug} ({count} prescriptions)")

    # 5. SUMMARY AND RECOMMENDATIONS
    print("\n" + "="*60)
    print("5. DATA-DRIVEN SUMMARY AND RECOMMENDATIONS")
    print("="*60)

    print("\nVALIDATED FINDINGS FROM ACTUAL DATA:")
    print(f"• {len(found_diabetes_labs)} types of diabetes-related lab tests found")
    print(f"• {len(found_diabetes_meds)} types of diabetes medications found")
    print(f"• {len(diabetes_patients)} diabetes patients identified")
    print(f"• {len(diabetes_code_summary)} different diabetes ICD codes used")


sys.stdout = original_stdout