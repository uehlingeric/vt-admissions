# Virginia Tech Admissions Trends Analysis (2018-2025)

A comprehensive data analysis project examining Virginia Tech admissions patterns, demographic disparities, COVID-19 impact, and future enrollment projections.

## Project Overview

This project analyzes Virginia Tech admissions data spanning seven academic years (2018-19 through 2024-25), providing insights into application volumes, acceptance rates, yield rates, and enrollment metrics across various demographic dimensions. The analysis reveals significant shifts in admissions patterns, with particular attention to demographic equity considerations and the transformative impact of the COVID-19 pandemic.

## Key Features

- **Trend Analysis**: Tracks applications, offers, and enrollment volumes over time
- **Demographic Comparisons**: Analyzes disparities across racial/ethnic groups, residency categories, generation status, and gender
- **COVID-19 Impact Assessment**: Examines how the pandemic affected admissions metrics
- **Yield Optimization Analysis**: Identifies opportunities to improve enrollment yield for underrepresented groups
- **Geographic Diversity Analysis**: Evaluates trends in domestic vs. international enrollment
- **Predictive Modeling**: Projects future applications, enrollment, and yield rates
- **Interactive Visualizations**: Includes comprehensive dashboards and charts

## Data Description

The dataset includes admissions metrics for various demographic groups:

- **Race/Ethnicity**: Asian, Black, Hispanic, White
- **Residency**: In-state, Out-of-state, International, National
- **Generation Status**: First-generation, Non-first-generation
- **Gender**: Male, Female

Five key metrics are analyzed for each group:
- Applied: Number of students who submitted applications
- Offered: Number of students who received admission offers
- Enrolled: Number of students who accepted offers and enrolled
- Offered Rate: Percentage of applicants who received offers
- Yield: Percentage of admitted students who enrolled

## Project Structure

```
.
├── admissions.ipynb          # Main Jupyter notebook with analysis
├── utils.py                  # Utility functions for data processing and visualization
├── report.pdf                # Comprehensive analysis report
├── requirements.txt          # Required Python packages
├── LICENSE.txt               # License information
├── data/                     # CSV data files for each demographic group
│   ├── all.csv               # Overall admissions data
│   ├── asian.csv             # Admissions data for Asian students
│   ├── black.csv             # Admissions data for Black students
│   └── ...                   # Additional demographic data files
├── images/                   # Visualizations generated during analysis
└── docs/                     # Source files for the report
```

## Installation

1. Clone this repository:
```bash
git clone https://github.com/uehlingeric/vt-admissions.git
cd vt-admissions
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

## Key Analysis Functions

The `utils.py` module provides numerous analysis functions:

- `load_all_data()`: Load and organize CSV files by demographic categories
- `calculate_year_over_year_change()`: Analyze annual percentage changes
- `analyze_covid_impact()`: Assess pandemic effects across demographic groups
- `calculate_demographic_ratios()`: Analyze demographic composition changes
- `compare_acceptance_rates()`: Identify disparities in acceptance rates
- `compare_yield_rates()`: Examine differences in yield across groups
- `find_significant_trends()`: Identify statistically significant patterns
- `predict_future_trends()`: Project future admissions metrics
- Visualization functions: Create various charts and dashboards

## Dependencies

- pandas
- numpy
- matplotlib
- seaborn
- scipy
- jupyter

## Key Findings

1. **Application Growth**: 65% increase in applications over the seven-year period
2. **Yield Rate Decline**: Yield rates declined from 34% to 25.2%, requiring more offers to meet enrollment goals
3. **Demographic Disparities**: Significant gaps in acceptance rates across racial/ethnic groups
4. **COVID-19 Impact**: The pandemic accelerated application growth but disrupted yield patterns
5. **Geographic Patterns**: Different acceptance and yield rates for in-state, out-of-state, and international students
6. **First-Generation Students**: Only 13% of enrolled students are first-generation, indicating opportunity for improvement

## License

This project is licensed under the terms included in LICENSE.txt.

## Author

Eric Uehling
