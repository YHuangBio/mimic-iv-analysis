#!/usr/bin/env python3
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
original_stdout = sys.stdout

# Set up paths
data_dir = Path('.')
hosp_dir = data_dir / 'hosp'

# Load datasets
labevents = pd.read_csv(hosp_dir / 'labevents.csv.gz')
d_labitems = pd.read_csv(hosp_dir / 'd_labitems.csv.gz')
diagnoses_icd = pd.read_csv(hosp_dir / 'diagnoses_icd.csv.gz')
d_icd_diagnoses = pd.read_csv(hosp_dir / 'd_icd_diagnoses.csv.gz')
prescriptions = pd.read_csv(hosp_dir / 'prescriptions.csv.gz')
admissions = pd.read_csv(hosp_dir / 'admissions.csv.gz')
patients = pd.read_csv(hosp_dir / 'patients.csv.gz')

# STEP 1: IDENTIFY DIABETES PATIENTS FROM ACTUAL ICD CODES
print("STEP 1: Identifying diabetes patients from actual ICD codes")
diabetes_diagnoses = diagnoses_icd[diagnoses_icd['icd_code'].str.startswith('250')]
diabetes_patients = diabetes_diagnoses['subject_id'].unique()

# Show actual diabetes ICD codes in the dataset
diabetes_codes = diabetes_diagnoses['icd_code'].value_counts()
for code, count in diabetes_codes.items():
    desc = d_icd_diagnoses[d_icd_diagnoses['icd_code'] == code]['long_title'].values
    desc_text = desc[0] if len(desc) > 0 else "No description"

# STEP 2: IDENTIFY DIABETES-RELATED LABS FROM ACTUAL DATA
print(f"\nSTEP 2: Identifying diabetes-related labs from actual data")

# Get labs actually ordered for diabetes patients
diabetes_lab_events = labevents[labevents['subject_id'].isin(diabetes_patients)]
diabetes_lab_counts = diabetes_lab_events['itemid'].value_counts()

# Get top 20 labs and their names
validated_diabetes_labs = {}
for itemid, count in diabetes_lab_counts.head(20).items():
    lab_info = d_labitems[d_labitems['itemid'] == itemid]
    if len(lab_info) > 0:
        lab_name = lab_info['label'].values[0]
        category = lab_info['category'].values[0]
        
        # Check if this lab is actually diabetes-related
        diabetes_related_keywords = [
            'glucose', 'hemoglobin', 'hba1c', 'insulin', 'cholesterol', 
            'triglyceride', 'creatinine', 'albumin', 'protein', 'lactate',
            'anion', 'sodium', 'potassium', 'chloride', 'urea'
        ]
        
        is_diabetes_related = any(keyword in lab_name.lower() for keyword in diabetes_related_keywords)
        if is_diabetes_related:
            validated_diabetes_labs[itemid] = {
                'name': lab_name,
                'category': category,
                'diabetes_tests': count
            }


# STEP 3: IDENTIFY DIABETES MEDICATIONS FROM ACTUAL DATA
print(f"\nSTEP 3: Identifying diabetes medications from actual data")

# Get medications actually prescribed to diabetes patients
diabetes_prescriptions = prescriptions[prescriptions['subject_id'].isin(diabetes_patients)]
diabetes_drug_counts = diabetes_prescriptions['drug'].value_counts()

# Look for actual diabetes medications
validated_diabetes_meds = {}
diabetes_drug_keywords = ['insulin', 'metformin', 'glipizide', 'glyburide', 'glucose']

for drug, count in diabetes_drug_counts.head(30).items():
    is_diabetes_med = any(keyword in drug.lower() for keyword in diabetes_drug_keywords)
    if is_diabetes_med:
        validated_diabetes_meds[drug] = count


# STEP 4: ANALYZE VALIDATED DIABETES-RELATED DATA
print(f"\nSTEP 4: Analyzing validated diabetes-related data")

# Analyze each validated lab test
analysis_results = {}

for itemid, lab_info in validated_diabetes_labs.items():
    lab_data = labevents[labevents['itemid'] == itemid].copy()
    
    if len(lab_data) > 20:  # Only analyze if sufficient data
        # Add diabetes status
        lab_data['has_diabetes'] = lab_data['subject_id'].isin(diabetes_patients)
        
        # Convert values to numeric
        lab_data['valuenum'] = pd.to_numeric(lab_data['valuenum'], errors='coerce')
        lab_data = lab_data.dropna(subset=['valuenum'])
        
        if len(lab_data) > 10:
            diabetes_values = lab_data[lab_data['has_diabetes']]['valuenum']
            non_diabetes_values = lab_data[~lab_data['has_diabetes']]['valuenum']
            
            if len(diabetes_values) > 0 and len(non_diabetes_values) > 0:
                analysis_results[lab_info['name']] = {
                    'itemid': itemid,
                    'diabetes_mean': diabetes_values.mean(),
                    'diabetes_std': diabetes_values.std(),
                    'diabetes_count': len(diabetes_values),
                    'non_diabetes_mean': non_diabetes_values.mean(),
                    'non_diabetes_std': non_diabetes_values.std(),
                    'non_diabetes_count': len(non_diabetes_values),
                    'diabetes_values': diabetes_values,
                    'non_diabetes_values': non_diabetes_values
                }


# STEP 5: CREATE COMPREHENSIVE EVIDENCE-BASED VISUALIZATIONS
print(f"\nSTEP 5: Creating comprehensive evidence-based visualizations")

if len(analysis_results) > 0:
    # Set up style
    plt.style.use('seaborn-v0_8')
    
    # Calculate layout
    n_tests = len(analysis_results)
    n_cols = 3
    n_rows = (n_tests + n_cols - 1) // n_cols
    
    # 1. BOX PLOTS - Enhanced version
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 6*n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1) if n_cols > 1 else [axes]
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    plot_idx = 0
    for test_name, results in analysis_results.items():
        row = plot_idx // n_cols
        col = plot_idx % n_cols
        ax = axes[row, col] if n_rows > 1 else axes[col]
        
        # Create box plot
        box_data = [results['diabetes_values'], results['non_diabetes_values']]
        bp = ax.boxplot(box_data, labels=['Diabetes', 'Non-Diabetes'], patch_artist=True)
        
        # Color the boxes
        bp['boxes'][0].set_facecolor('lightcoral')
        bp['boxes'][1].set_facecolor('lightblue')
        
        ax.set_title(f'{test_name}\n(ID: {results["itemid"]})', fontsize=12, fontweight='bold')
        ax.set_ylabel('Value', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Add statistical info
        mean_diff = results['diabetes_mean'] - results['non_diabetes_mean']
        info_text = f'Difference: {mean_diff:.2f}\n'
        info_text += f'DM: {results["diabetes_mean"]:.1f}±{results["diabetes_std"]:.1f} (n={results["diabetes_count"]})\n'
        info_text += f'Non-DM: {results["non_diabetes_mean"]:.1f}±{results["non_diabetes_std"]:.1f} (n={results["non_diabetes_count"]})'
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes, 
                verticalalignment='top', fontsize=8, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        plot_idx += 1
    
    # Hide unused subplots
    for i in range(plot_idx, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        ax = axes[row, col] if n_rows > 1 else axes[col]
        ax.set_visible(False)
    
    plt.tight_layout()
    plt.savefig('diabetes_boxplots.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 2. HISTOGRAM OVERLAYS
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 6*n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1) if n_cols > 1 else [axes]
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    plot_idx = 0
    for test_name, results in analysis_results.items():
        row = plot_idx // n_cols
        col = plot_idx % n_cols
        ax = axes[row, col] if n_rows > 1 else axes[col]
        
        # Create histogram overlay
        ax.hist(results['diabetes_values'], bins=30, alpha=0.7, label='Diabetes', 
                color='red', density=True, edgecolor='darkred', linewidth=0.5)
        ax.hist(results['non_diabetes_values'], bins=30, alpha=0.7, label='Non-Diabetes', 
                color='blue', density=True, edgecolor='darkblue', linewidth=0.5)
        
        ax.set_xlabel('Value', fontsize=10)
        ax.set_ylabel('Density', fontsize=10)
        ax.set_title(f'{test_name} Distribution\n(ID: {results["itemid"]})', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add mean lines
        ax.axvline(results['diabetes_mean'], color='red', linestyle='--', alpha=0.8, linewidth=2)
        ax.axvline(results['non_diabetes_mean'], color='blue', linestyle='--', alpha=0.8, linewidth=2)
        
        plot_idx += 1
    
    # Hide unused subplots
    for i in range(plot_idx, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        ax = axes[row, col] if n_rows > 1 else axes[col]
        ax.set_visible(False)
    
    plt.tight_layout()
    plt.savefig('diabetes_histograms.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 3. SCATTER PLOTS SHOWING INDIVIDUAL VALUES
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 6*n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1) if n_cols > 1 else [axes]
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    plot_idx = 0
    for test_name, results in analysis_results.items():
        row = plot_idx // n_cols
        col = plot_idx % n_cols
        ax = axes[row, col] if n_rows > 1 else axes[col]
        
        # Create scatter plot with jitter
        np.random.seed(42)  # For reproducible jitter
        diabetes_y = np.random.normal(1, 0.05, len(results['diabetes_values']))
        non_diabetes_y = np.random.normal(2, 0.05, len(results['non_diabetes_values']))
        
        ax.scatter(results['diabetes_values'], diabetes_y, alpha=0.6, color='red', 
                   label='Diabetes', s=15, edgecolors='darkred', linewidth=0.3)
        ax.scatter(results['non_diabetes_values'], non_diabetes_y, alpha=0.6, color='blue', 
                   label='Non-Diabetes', s=15, edgecolors='darkblue', linewidth=0.3)
        
        ax.set_xlabel('Value', fontsize=10)
        ax.set_ylabel('Group', fontsize=10)
        ax.set_title(f'{test_name} Individual Values\n(ID: {results["itemid"]})', fontsize=12, fontweight='bold')
        ax.set_yticks([1, 2])
        ax.set_yticklabels(['Diabetes', 'Non-Diabetes'])
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add mean lines
        ax.axvline(results['diabetes_mean'], color='red', linestyle='--', alpha=0.8, linewidth=2)
        ax.axvline(results['non_diabetes_mean'], color='blue', linestyle='--', alpha=0.8, linewidth=2)
        
        plot_idx += 1
    
    # Hide unused subplots
    for i in range(plot_idx, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        ax = axes[row, col] if n_rows > 1 else axes[col]
        ax.set_visible(False)
    
    plt.tight_layout()
    plt.savefig('diabetes_scatter.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 4. EFFECT SIZE COMPARISON
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Calculate effect sizes
    test_names = []
    effect_sizes = []
    colors = []
    
    for test_name, results in analysis_results.items():
        diabetes_mean = results['diabetes_mean']
        non_diabetes_mean = results['non_diabetes_mean']
        diabetes_std = results['diabetes_std']
        non_diabetes_std = results['non_diabetes_std']
        
        # Cohen's d effect size
        pooled_std = np.sqrt(((results['diabetes_count']-1)*diabetes_std**2 + 
                             (results['non_diabetes_count']-1)*non_diabetes_std**2) / 
                             (results['diabetes_count']+results['non_diabetes_count']-2))
        cohens_d = (diabetes_mean - non_diabetes_mean) / pooled_std if pooled_std > 0 else 0
        
        test_names.append(test_name)
        effect_sizes.append(cohens_d)
        
        # Color by effect size magnitude
        if abs(cohens_d) > 0.8:
            colors.append('red')
        elif abs(cohens_d) > 0.5:
            colors.append('orange')
        elif abs(cohens_d) > 0.2:
            colors.append('yellow')
        else:
            colors.append('lightblue')
    
    # Create horizontal bar chart
    y_pos = np.arange(len(test_names))
    bars = ax.barh(y_pos, effect_sizes, color=colors, alpha=0.7, edgecolor='black')
    
    # Add value labels on bars
    for i, (bar, effect_size) in enumerate(zip(bars, effect_sizes)):
        width = bar.get_width()
        ax.text(width + (0.05 if width >= 0 else -0.05), bar.get_y() + bar.get_height()/2, 
                f'{effect_size:.3f}', ha='left' if width >= 0 else 'right', va='center', fontweight='bold')
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(test_names)
    ax.set_xlabel('Effect Size (Cohen\'s d)', fontsize=12)
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add reference lines
    ax.axvline(0, color='black', linewidth=1)
    ax.axvline(0.2, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(0.5, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(0.8, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(-0.2, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(-0.5, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(-0.8, color='gray', linestyle='--', alpha=0.5)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='red', alpha=0.7, label='Large Effect (|d| > 0.8)'),
        Patch(facecolor='orange', alpha=0.7, label='Medium Effect (0.5 < |d| ≤ 0.8)'),
        Patch(facecolor='yellow', alpha=0.7, label='Small Effect (0.2 < |d| ≤ 0.5)'),
        Patch(facecolor='lightblue', alpha=0.7, label='Minimal Effect (|d| ≤ 0.2)')
    ]
    ax.legend(handles=legend_elements, loc='lower right')
    
    plt.tight_layout()
    plt.savefig('diabetes_effect_sizes.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 5. MEDICATION ANALYSIS
    if validated_diabetes_meds:
        fig, ax = plt.subplots(figsize=(12, 6))
        
        drugs = list(validated_diabetes_meds.keys())
        counts = list(validated_diabetes_meds.values())
        
        bars = ax.bar(drugs, counts, color='skyblue', alpha=0.7, edgecolor='darkblue')
        
        # Add value labels on bars
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + max(counts)*0.01,
                    f'{count}', ha='center', va='bottom', fontweight='bold')
        
        ax.set_xlabel('Diabetes Medications', fontsize=12)
        ax.set_ylabel('Number of Prescriptions', fontsize=12)
        ax.set_title('Validated Diabetes Medications in Dataset', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig('diabetes_medications.png', dpi=300, bbox_inches='tight')
        plt.show()

# STEP 6: SUMMARY OF VALIDATED FINDINGS
with open("analysis_output.txt", "w", encoding="utf-8") as f:
    sys.stdout = f
    print(f"\n" + "="*60)
    print("SUMMARY OF VALIDATED FINDINGS")
    print("="*60)

    print(f"\nDATASET VALIDATION RESULTS:")
    print(f"• Diabetes patients identified: {len(diabetes_patients)} ({len(diabetes_patients)/len(patients)*100:.1f}% of total)")
    print(f"• Validated diabetes-related labs: {len(validated_diabetes_labs)}")
    print(f"• Validated diabetes medications: {len(validated_diabetes_meds)}")
    print(f"• Successful lab analyses: {len(analysis_results)}")

    print(f"\nVALIDATED DIABETES ICD CODES:")
    for code, count in diabetes_codes.items():
        print(f"  {code}: {count} cases")

    print(f"\nVALIDATED DIABETES LABS:")
    for itemid, info in validated_diabetes_labs.items():
        print(f"  {itemid}: {info['name']} ({info['diabetes_tests']} tests)")

    print(f"\nVALIDATED DIABETES MEDICATIONS:")
    for drug, count in validated_diabetes_meds.items():
        print(f"  {drug}: {count} prescriptions")

    print(f"\nSTATISTICAL RESULTS:")
    for test_name, results in analysis_results.items():
        mean_diff = results['diabetes_mean'] - results['non_diabetes_mean']
        print(f"\n{test_name}:")
        print(f"  Mean difference: {mean_diff:.2f}")
        print(f"  Diabetes: {results['diabetes_mean']:.2f} ± {results['diabetes_std']:.2f}")
        print(f"  Non-diabetes: {results['non_diabetes_mean']:.2f} ± {results['non_diabetes_std']:.2f}")

sys.stdout = original_stdout