# Exploratory Data Analysis for Cyber Breach Dataset

This directory contains an exploratory data analysis (EDA) script for the Privacy Rights Clearinghouse (PRC) data breach dataset included in the cyber risk quantification project.

## Files

- `eda_breach_data.py`: Main Python script performing EDA
- `README.md`: This file

## Requirements

The script requires the following Python packages:
- pandas
- numpy
- matplotlib
- seaborn

These should already be available in the project's virtual environment.

## Usage

To run the analysis, activate the virtual environment and execute:

```bash
cd /c/folder/
.\venv\Scripts\activate
python eda_analysis/eda_breach_data.py
```

## Outputs

The analysis will generate:
1. **Visualizations**: Saved in `eda_analysis/plots/` directory
   - Breach size distribution
   - Temporal trends (yearly and monthly)
   - Top breach types and organization types
   - Correlation matrix
   - Information affected analysis

2. **Reports**: Saved in `eda_analysis/` directory
   - `eda_report.json`: Detailed analysis results in JSON format
   - `eda_summary.md`: Human-readable summary in Markdown format

## Analysis Performed

The script conducts:
1. Data loading and initial inspection
2. Data cleaning and preprocessing (date conversion, JSON parsing)
3. Statistical summaries of numerical and categorical variables
4. Visualizations of key distributions and trends
5. Correlation analysis to identify key relationships
6. Identification of factors influencing breach impact
7. Generation of structured reports with insights and recommendations

## Notes

- The script handles the complex JSON-like structure in the `information_affected` column
- Date columns are properly converted for temporal analysis
- Breach size distributions are visualized with both linear and log scales due to high skewness
- Only organization types and breach types with sufficient sample size (≥5 incidents) are considered in risk analysis
