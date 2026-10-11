"""
Exploratory Data Analysis for Cyber Breach Dataset
PRC (Privacy Rights Clearinghouse) Data Breach Analysis

This script performs:
1. Statistical summaries of the breach data
2. Visualizations of key variables
3. Correlation analysis
4. Identification of key influencing factors
5. Generation of a structured insights report

Author: Generated for Cyber Risk Quantification Project
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
import os

# Set up plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_and_inspect_data(filepath):
    """
    Load the breach data and perform initial inspection
    """
    print("Loading data...")
    df = pd.read_csv(filepath)
    
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Basic info
    print("\n=== DATASET INFO ===")
    print(df.info())
    
    # Check for missing values
    print("\n=== MISSING VALUES ===")
    missing = df.isnull().sum()
    missing_percent = (missing / len(df)) * 100
    missing_df = pd.DataFrame({
        'Missing Count': missing,
        'Missing Percentage': missing_percent
    })
    print(missing_df[missing_df['Missing Count'] > 0])
    
    return df

def clean_and_preprocess(df):
    """
    Clean and preprocess the data for analysis
    """
    print("\n=== DATA CLEANING ===")
    
    # Create a copy to avoid modifying original
    df_clean = df.copy()
    
    # Convert date columns to datetime
    date_columns = ['reported_date', 'breach_date', 'end_breach_date']
    for col in date_columns:
        if col in df_clean.columns:
            df_clean[col] = pd.to_datetime(df_clean[col], errors='coerce')
    
    # Extract year from reported_date for temporal analysis
    if 'reported_date' in df_clean.columns:
        df_clean['reported_year'] = df_clean['reported_date'].dt.year
        df_clean['reported_month'] = df_clean['reported_date'].dt.month
    
    # Handle the 'information_affected' column which contains JSON-like strings
    # We'll extract key information from it
    if 'information_affected' in df_clean.columns:
        # Parse the JSON-like string to extract affected information types
        def extract_affected_info(json_str):
            try:
                if pd.isna(json_str) or json_str == '':
                    return {}
                # Replace single quotes with double quotes for valid JSON
                json_str = json_str.replace("'", '"')
                # Handle the case where it might be malformed
                data = json.loads(json_str)
                return data
            except:
                return {}
        
        # Apply extraction
        info_affected_parsed = df_clean['information_affected'].apply(extract_affected_info)
        
        # Create binary columns for each information type
        info_types = [
            'Personal Identifiers Affected',
            'Protected Classifications Affected', 
            'Financial Account Information Affected',
            'Biometric Information Affected',
            'Internet Network Activity Affected',
            'Geolocation Data Affected',
            'Sensory Data Affected',
            'Professional Employment Affected',
            'Education Information Affected',
            'Inferences Affected',
            'Health Information Affected',
            'Customer Records Affected',
            'Other Information Affected'
        ]
        
        for info_type in info_types:
            # Create a column indicating if this information type was affected (Yes/No)
            df_clean[f'{info_type}_binary'] = info_affected_parsed.apply(
                lambda x: 1 if isinstance(x, dict) and x.get(info_type) == 'Yes' else 0
            )
    
    # Clean numerical columns
    if 'total_affected' in df_clean.columns:
        # Convert to numeric, handling any non-numeric values
        df_clean['total_affected'] = pd.to_numeric(df_clean['total_affected'], errors='coerce')
        # Log transform for skewed distribution
        df_clean['log_total_affected'] = np.log1p(df_clean['total_affected'])
    
    print(f"Cleaned dataset shape: {df_clean.shape}")
    return df_clean

def statistical_summary(df):
    """
    Generate statistical summaries of the data
    """
    print("\n=== STATISTICAL SUMMARY ===")
    
    # Numerical columns summary
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if numerical_cols:
        print("\nNumerical Variables Summary:")
        print(df[numerical_cols].describe())
    
    # Categorical columns summary
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    if categorical_cols:
        print("\nCategorical Variables Summary (top 5 values):")
        for col in categorical_cols[:10]:  # Limit to first 10 to avoid too much output
            print(f"\n{col}:")
            print(df[col].value_counts().head())
    
    # Date-related summaries
    date_cols = [col for col in df.columns if 'date' in col.lower() and df[col].dtype == 'datetime64[ns]']
    if date_cols:
        print("\nDate Variables Summary:")
        for col in date_cols:
            print(f"\n{col}:")
            print(f"  Range: {df[col].min()} to {df[col].max()}")
            print(f"  Missing: {df[col].isnull().sum()}")

def create_visualizations(df, output_dir):
    """
    Create visualizations for the data exploration
    """
    print("\n=== CREATING VISUALIZATIONS ===")
    
    # Create output directory for plots if it doesn't exist
    plots_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # 1. Distribution of breach sizes (total_affected)
    if 'total_affected' in df.columns:
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.hist(df['total_affected'].dropna(), bins=50, edgecolor='black', alpha=0.7)
        plt.title('Distribution of Total Affected Individuals')
        plt.xlabel('Number of Individuals Affected')
        plt.ylabel('Frequency')
        plt.yscale('log')  # Log scale due to skewed distribution
        
        plt.subplot(1, 2, 2)
        plt.hist(df['log_total_affected'].dropna(), bins=50, edgecolor='black', alpha=0.7)
        plt.title('Distribution of Log(Total Affected + 1)')
        plt.xlabel('Log(Total Affected + 1)')
        plt.ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'breach_size_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # 2. Breaches over time (if reported_date available)
    if 'reported_year' in df.columns:
        plt.figure(figsize=(14, 6))
        
        # Yearly counts
        yearly_counts = df['reported_year'].value_counts().sort_index()
        plt.subplot(1, 2, 1)
        plt.bar(yearly_counts.index, yearly_counts.values, edgecolor='black', alpha=0.7)
        plt.title('Number of Breaches Reported by Year')
        plt.xlabel('Year')
        plt.ylabel('Number of Breaches')
        plt.xticks(rotation=45)
        
        # Monthly trends (for most recent year or all data)
        if 'reported_month' in df.columns:
            monthly_counts = df['reported_month'].value_counts().sort_index()
            plt.subplot(1, 2, 2)
            plt.bar(monthly_counts.index, monthly_counts.values, edgecolor='black', alpha=0.7)
            plt.title('Number of Breaches by Month (All Years)')
            plt.xlabel('Month')
            plt.ylabel('Number of Breaches')
            plt.xticks(range(1, 13), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'temporal_trends.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # 3. Breach type analysis
    if 'breach_type' in df.columns:
        plt.figure(figsize=(12, 8))
        breach_type_counts = df['breach_type'].value_counts().head(10)  # Top 10
        plt.barh(range(len(breach_type_counts)), breach_type_counts.values, edgecolor='black', alpha=0.7)
        plt.yticks(range(len(breach_type_counts)), breach_type_counts.index)
        plt.title('Top 10 Breach Types')
        plt.xlabel('Number of Breaches')
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'breach_types.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # 4. Organization type analysis
    if 'organization_type' in df.columns:
        plt.figure(figsize=(12, 8))
        org_type_counts = df['organization_type'].value_counts().head(10)
        plt.barh(range(len(org_type_counts)), org_type_counts.values, edgecolor='black', alpha=0.7)
        plt.yticks(range(len(org_type_counts)), org_type_counts.index)
        plt.title('Top 10 Organization Types Affected')
        plt.xlabel('Number of Breaches')
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'organization_types.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # 5. Correlation matrix (for numerical variables)
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numerical_cols) > 1:
        # Select key numerical variables for correlation
        key_numerical = [col for col in numerical_cols if any(keyword in col.lower() 
                        for keyword in ['affected', 'total', 'count', 'number'])]
        if not key_numerical:
            key_numerical = numerical_cols[:5]  # First 5 if no specific matches
        
        if len(key_numerical) > 1:
            corr_matrix = df[key_numerical].corr()
            plt.figure(figsize=(10, 8))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                        square=True, linewidths=0.5)
            plt.title('Correlation Matrix of Key Numerical Variables')
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, 'correlation_matrix.png'), dpi=300, bbox_inches='tight')
            plt.close()
    
    # 6. Information affected analysis (if we created binary columns)
    info_binary_cols = [col for col in df.columns if col.endswith('_binary') and 
                       any(info in col for info in ['Personal', 'Financial', 'Health', 'Customer'])]
    if info_binary_cols:
        # Calculate percentage of breaches affecting each information type
        info_percentages = df[info_binary_cols].mean() * 100
        info_percentages = info_percentages.sort_values(ascending=False)
        
        plt.figure(figsize=(12, 8))
        plt.barh(range(len(info_percentages)), info_percentages.values, edgecolor='black', alpha=0.7)
        plt.yticks(range(len(info_percentages)), [col.replace('_binary', '') for col in info_percentages.index])
        plt.title('Percentage of Breaches Affecting Each Information Type')
        plt.xlabel('Percentage of Breaches')
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'information_affected.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"Visualizations saved to {plots_dir}")

def correlation_analysis(df):
    """
    Perform correlation analysis to identify key relationships
    """
    print("\n=== CORRELATION ANALYSIS ===")
    
    # Get numerical columns
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numerical_cols) < 2:
        print("Not enough numerical columns for correlation analysis")
        return {}
    
    # Calculate correlation matrix
    corr_matrix = df[numerical_cols].corr()
    
    # Find strong correlations (absolute value > 0.5)
    strong_correlations = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            col1 = corr_matrix.columns[i]
            col2 = corr_matrix.columns[j]
            corr_value = corr_matrix.iloc[i, j]
            if abs(corr_value) > 0.5:
                strong_correlations.append({
                    'variable_1': col1,
                    'variable_2': col2,
                    'correlation': corr_value,
                    'strength': 'strong' if abs(corr_value) > 0.7 else 'moderate'
                })
    
    # Sort by absolute correlation value
    strong_correlations.sort(key=lambda x: abs(x['correlation']), reverse=True)
    
    print(f"Found {len(strong_correlations)} strong correlations (|r| > 0.5):")
    for corr in strong_correlations[:10]:  # Show top 10
        print(f"  {corr['variable_1']} <-> {corr['variable_2']}: {corr['correlation']:.3f} ({corr['strength']})")
    
    return {
        'correlation_matrix': corr_matrix.to_dict(),
        'strong_correlations': strong_correlations
    }

def identify_key_factors(df):
    """
    Identify key factors that influence breach impact
    """
    print("\n=== KEY FACTOR IDENTIFICATION ===")
    
    insights = {}
    
    # Factor 1: Breach size statistics
    if 'total_affected' in df.columns:
        total_affected = df['total_affected'].dropna()
        insights['breach_size'] = {
            'mean_affected': float(total_affected.mean()),
            'median_affected': float(total_affected.median()),
            'std_affected': float(total_affected.std()),
            'max_affected': float(total_affected.max()),
            '90th_percentile': float(total_affected.quantile(0.9)),
            '95th_percentile': float(total_affected.quantile(0.95)),
            'breaches_affecting_over_1M': int((total_affected >= 1_000_000).sum()),
            'breaches_affecting_over_100K': int((total_affected >= 100_000).sum())
        }
    
    # Factor 2: Temporal trends
    if 'reported_year' in df.columns:
        yearly_counts = df['reported_year'].value_counts().sort_index()
        insights['temporal_trends'] = {
            'years_covered': f"{yearly_counts.index.min()}-{yearly_counts.index.max()}",
            'yearly_breach_counts': yearly_counts.to_dict(),
            'year_with_most_breaches': int(yearly_counts.idxmax()),
            'most_breaches_count': int(yearly_counts.max())
        }
    
    # Factor 3: Organization type risk
    if 'organization_type' in df.columns and 'total_affected' in df.columns:
        # Average breach size by organization type
        org_impact = df.groupby('organization_type')['total_affected'].agg(['mean', 'median', 'count'])
        org_impact = org_impact[org_impact['count'] >= 5]  # Only org types with at least 5 breaches
        org_impact = org_impact.sort_values('mean', ascending=False)
        
        insights['organization_risk'] = {
            'highest_avg_impact_org_types': org_impact.head(5)[['mean', 'count']].to_dict('index'),
            'lowest_avg_impact_org_types': org_impact.tail(5)[['mean', 'count']].to_dict('index')
        }
    
    # Factor 4: Breach type impact
    if 'breach_type' in df.columns and 'total_affected' in df.columns:
        breach_impact = df.groupby('breach_type')['total_affected'].agg(['mean', 'median', 'count'])
        breach_impact = breach_impact[breach_impact['count'] >= 5]  # Only breach types with at least 5 incidents
        breach_impact = breach_impact.sort_values('mean', ascending=False)
        
        insights['breach_type_risk'] = {
            'highest_avg_impact_breach_types': breach_impact.head(5)[['mean', 'count']].to_dict('index'),
            'lowest_avg_impact_breach_types': breach_impact.tail(5)[['mean', 'count']].to_dict('index')
        }
    
    # Factor 5: Information affected patterns
    info_binary_cols = [col for col in df.columns if col.endswith('_binary') and 
                       any(info in col for info in ['Personal', 'Financial', 'Health', 'Customer'])]
    if info_binary_cols and 'total_affected' in df.columns:
        info_impact = {}
        for col in info_binary_cols:
            info_type = col.replace('_binary', '')
            # Compare mean breach size when this info type is affected vs not
            affected_mean = df[df[col] == 1]['total_affected'].mean()
            not_affected_mean = df[df[col] == 0]['total_affected'].mean()
            info_impact[info_type] = {
                'mean_when_affected': float(affected_mean) if not pd.isna(affected_mean) else 0,
                'mean_when_not_affected': float(not_affected_mean) if not pd.isna(not_affected_mean) else 0,
                'percentage_breaches': float(df[col].mean() * 100)
            }
        
        insights['information_impact'] = info_impact
    
    return insights

def generate_report(df, stats, correlations, insights, output_dir):
    """
    Generate a structured report of the analysis
    """
    print("\n=== GENERATING REPORT ===")
    
    report = {
        'analysis_metadata': {
            'timestamp': datetime.now().isoformat(),
            'dataset_size': {
                'rows': int(len(df)),
                'columns': int(len(df.columns))
            },
            'analysis_type': 'Exploratory Data Analysis - Cyber Breach Dataset'
        },
        'statistical_summary': stats,
        'correlation_findings': correlations,
        'key_insights': insights,
        'recommendations': [
            "Focus security efforts on organization types with highest average breach impact",
            "Monitor temporal patterns for potential seasonal variations in breach occurrences",
            "Prioritize protection of information types most commonly associated with large breaches",
            "Consider implementing breach size prediction models based on organizational and temporal factors",
            "Regularly update analysis as new breach data becomes available"
        ]
    }
    
    # Save report as JSON
    report_path = os.path.join(output_dir, 'eda_report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    # Also create a markdown summary for easy reading
    markdown_path = os.path.join(output_dir, 'eda_summary.md')
    with open(markdown_path, 'w') as f:
        f.write("# Exploratory Data Analysis: Cyber Breach Dataset\n\n")
        f.write(f"**Analysis Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**Dataset Size:** {len(df)} records, {len(df.columns)} variables\n\n")
        
        f.write("## Key Findings\n\n")
        
        # Breach size insights
        if 'breach_size' in insights:
            bs = insights['breach_size']
            f.write("### Breach Size Distribution\n")
            f.write(f"- Average individuals affected per breach: {bs['mean_affected']:,.0f}\n")
            f.write(f"- Median individuals affected per breach: {bs['median_affected']:,.0f}\n")
            f.write(f"- Breaches affecting over 1 million people: {bs['breaches_affecting_over_1M']}\n")
            f.write(f"- Breaches affecting over 100,000 people: {bs['breaches_affecting_over_100K']}\n\n")
        
        # Temporal trends
        if 'temporal_trends' in insights:
            tt = insights['temporal_trends']
            f.write("### Temporal Trends\n")
            f.write(f"- Data covers years: {tt['years_covered']}\n")
            f.write(f"- Year with most breaches: {tt['year_with_most_breaches']} ({tt['most_breaches_count']} breaches)\n\n")
        
        # Organization risk
        if 'organization_risk' in insights:
            org = insights['organization_risk']
            f.write("### Organization Type Risk\n")
            f.write("**Highest Average Impact Organization Types:**\n")
            for org_type, stats in org['highest_avg_impact_org_types'].items():
                f.write(f"- {org_type}: {stats['mean']:,.0f} average affected ({stats['count']} breaches)\n")
            f.write("\n")
        
        # Breach type risk
        if 'breach_type_risk' in insights:
            breach = insights['breach_type_risk']
            f.write("### Breach Type Risk\n")
            f.write("**Highest Average Impact Breach Types:**\n")
            for breach_type, stats in breach['highest_avg_impact_breach_types'].items():
                f.write(f"- {breach_type}: {stats['mean']:,.0f} average affected ({stats['count']} breaches)\n")
            f.write("\n")
        
        # Correlation highlights
        if 'strong_correlations' in correlations and correlations['strong_correlations']:
            f.write("### Significant Correlations Found\n")
            for corr in correlations['strong_correlations'][:5]:  # Top 5
                f.write(f"- {corr['variable_1']} ↔ {corr['variable_2']}: {corr['correlation']:.3f} ({corr['strength']})\n")
            f.write("\n")
        
        f.write("## Recommendations\n\n")
        for i, rec in enumerate(report['recommendations'], 1):
            f.write(f"{i}. {rec}\n")
    
    print(f"Report saved to {report_path}")
    print(f"Markdown summary saved to {markdown_path}")
    
    return report

def main():
    """
    Main function to run the EDA pipeline
    """
    print("Starting Exploratory Data Analysis for Cyber Breach Data")
    print("=" * 60)
    
    # Define paths
    data_path = "/c/New D/Quntification/New folder/data/PRC_sample.csv"
    output_dir = "/c/New D/Quntification/New folder/eda_analysis"
    
    # Check if data file exists
    if not os.path.exists(data_path):
        print(f"ERROR: Data file not found at {data_path}")
        return
    
    # Run analysis pipeline
    try:
        # 1. Load and inspect data
        df_raw = load_and_inspect_data(data_path)
        
        # 2. Clean and preprocess
        df_clean = clean_and_preprocess(df_raw)
        
        # 3. Statistical summary
        stats = statistical_summary(df_clean)
        
        # 4. Create visualizations
        create_visualizations(df_clean, output_dir)
        
        # 5. Correlation analysis
        correlations = correlation_analysis(df_clean)
        
        # 6. Identify key factors
        insights = identify_key_factors(df_clean)
        
        # 7. Generate report
        report = generate_report(df_clean, stats, correlations, insights, output_dir)
        
        print("\n" + "=" * 60)
        print("EDA COMPLETED SUCCESSFULLY!")
        print(f"Results saved to: {output_dir}")
        print("=" * 60)
        
    except Exception as e:
        print(f"ERROR during analysis: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()