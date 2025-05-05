"""
Admissions Data Analysis Utilities

This module provides functions for loading, analyzing, and visualizing college admissions data
across various demographic categories.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple
from matplotlib.gridspec import GridSpec
import matplotlib.ticker as mtick
from scipy import stats

plt.style.use('ggplot')
sns.set_palette("viridis")

YEARS = ['2018-19', '2019-20', '2020-21', '2021-22', '2022-23', '2023-24', '2024-25']
COVID_PERIODS = {
    'Pre-COVID': ['2018-19', '2019-20'],
    'During-COVID': ['2020-21', '2021-22'],
    'Post-COVID': ['2022-23', '2023-24', '2024-25']
}
METRICS = ['Applied', 'Offered', 'Enrolled', 'Offered Rate', 'Yield']
DEMOGRAPHIC_GROUPS = {
    'race_ethnicity': ['asian', 'black', 'white', 'hispanic'],
    'residence': ['in_state', 'out_of_state', 'international', 'national'],
    'generation': ['first_gen', 'non_first_gen'],
    'gender': ['male', 'female'],
    'overall': ['all']
}

def load_csv(filepath: str) -> pd.DataFrame:
    """
    Load a single CSV file and convert it to a formatted DataFrame.
    
    Args:
        filepath: Path to the CSV file
        
    Returns:
        Formatted pandas DataFrame
    """
    df = pd.read_csv(filepath, header=0)
    
    df.columns = [col.strip('"') for col in df.columns]
    
    for col in df.columns[1:]:
        if pd.api.types.is_string_dtype(df[col]):
            df[col] = df[col].str.strip('"')
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df.set_index('Admissions', inplace=True)
    
    return df

def get_category_from_filename(filename: str) -> Tuple[str, str]:
    """
    Extract demographic category and specific group from a filename.
    
    Args:
        filename: Name of the CSV file
        
    Returns:
        Tuple of (demographic_category, group_name)
    """
    basename = os.path.basename(filename).replace('.csv', '')
    
    for category, groups in DEMOGRAPHIC_GROUPS.items():
        if basename in groups:
            return category, basename
    
    return 'unknown', basename

def load_all_data(data_dir: str = 'data/admission') -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Load all CSV files from the data directory and organize by demographic categories.
    
    Args:
        data_dir: Directory containing the CSV files
        
    Returns:
        Nested dictionary with structure {category: {group: dataframe}}
    """
    data = {}
    
    for category in DEMOGRAPHIC_GROUPS.keys():
        data[category] = {}
    
    for file in os.listdir(data_dir):
        if file.endswith('.csv'):
            filepath = os.path.join(data_dir, file)
            category, group = get_category_from_filename(file)
            
            if category != 'unknown':
                data[category][group] = load_csv(filepath)
    
    return data

def calculate_year_over_year_change(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate year-over-year changes for each metric.
    """
    yoy_df = df.copy().astype(float)
    
    for i in range(1, len(YEARS)):
        prev_year = YEARS[i-1]
        curr_year = YEARS[i]
        
        for metric in df.index:
            if pd.notna(df.loc[metric, prev_year]) and pd.notna(df.loc[metric, curr_year]) and df.loc[metric, prev_year] != 0:
                pct_change = ((df.loc[metric, curr_year] - df.loc[metric, prev_year]) / df.loc[metric, prev_year]) * 100
                yoy_df.loc[metric, curr_year] = pct_change
    
    yoy_df[YEARS[0]] = np.nan
    
    return yoy_df

def analyze_covid_impact(data: Dict[str, Dict[str, pd.DataFrame]]) -> Dict[str, pd.DataFrame]:
    """
    Analyze the impact of COVID-19 on admissions metrics across demographic groups.
    
    Args:
        data: Nested dictionary of admissions data by demographic category and group
        
    Returns:
        Dictionary of DataFrames with COVID period analysis for each metric
    """
    results = {}
    
    for metric in METRICS:
        covid_df = pd.DataFrame(index=DEMOGRAPHIC_GROUPS.keys(), columns=['Pre-COVID', 'During-COVID', 'Post-COVID'])
        
        for category, groups in DEMOGRAPHIC_GROUPS.items():
            if category not in data or not data[category]:
                continue
            
            for period, years in COVID_PERIODS.items():
                period_values = []
                
                for group in groups:
                    if group in data[category]:
                        group_df = data[category][group]
                        if metric in group_df.index:
                            period_avg = group_df.loc[metric, years].mean()
                            period_values.append(period_avg)
                
                if period_values:
                    covid_df.loc[category, period] = np.mean(period_values)
        
        covid_df['During vs Pre (%)'] = ((covid_df['During-COVID'] - covid_df['Pre-COVID']) / covid_df['Pre-COVID']) * 100
        covid_df['Post vs During (%)'] = ((covid_df['Post-COVID'] - covid_df['During-COVID']) / covid_df['During-COVID']) * 100
        covid_df['Post vs Pre (%)'] = ((covid_df['Post-COVID'] - covid_df['Pre-COVID']) / covid_df['Pre-COVID']) * 100
        
        results[metric] = covid_df
    
    return results

def calculate_demographic_ratios(data: Dict[str, Dict[str, pd.DataFrame]]) -> Dict[str, pd.DataFrame]:
    """
    Calculate the ratio of each demographic group within its category for each year.
    
    Args:
        data: Nested dictionary of admissions data by demographic category and group
        
    Returns:
        Dictionary of DataFrames with ratios for each year and category
    """
    results = {}
    
    for metric in ['Applied', 'Offered', 'Enrolled']:
        metric_results = {}
        
        for category, groups in DEMOGRAPHIC_GROUPS.items():
            if category in ['overall']:
                continue
                
            ratio_df = pd.DataFrame(index=groups, columns=YEARS)
            
            totals = {}
            for year in YEARS:
                year_total = 0
                for group in groups:
                    if group in data[category] and metric in data[category][group].index:
                        value = data[category][group].loc[metric, year]
                        if pd.notna(value):
                            year_total += value
                totals[year] = year_total
            
            for group in groups:
                if group in data[category] and metric in data[category][group].index:
                    for year in YEARS:
                        if totals[year] > 0:
                            ratio = data[category][group].loc[metric, year] / totals[year] * 100
                            ratio_df.loc[group, year] = ratio
            
            metric_results[category] = ratio_df
        
        results[metric] = metric_results
    
    return results

def compare_acceptance_rates(data: Dict[str, Dict[str, pd.DataFrame]]) -> Dict[str, pd.DataFrame]:
    """
    Compare acceptance rates across demographic groups.
    
    Args:
        data: Nested dictionary of admissions data by demographic category and group
        
    Returns:
        Dictionary of DataFrames with acceptance rate comparisons
    """
    results = {}
    
    for category, groups in DEMOGRAPHIC_GROUPS.items():
        if category in ['overall']:
            continue
            
        rate_df = pd.DataFrame(index=groups, columns=YEARS)
        
        for group in groups:
            if group in data[category] and 'Offered Rate' in data[category][group].index:
                for year in YEARS:
                    rate = data[category][group].loc['Offered Rate', year]
                    rate_df.loc[group, year] = rate
        
        results[category] = rate_df
    
    return results

def compare_yield_rates(data: Dict[str, Dict[str, pd.DataFrame]]) -> Dict[str, pd.DataFrame]:
    """
    Compare yield rates across demographic groups.
    
    Args:
        data: Nested dictionary of admissions data by demographic category and group
        
    Returns:
        Dictionary of DataFrames with yield rate comparisons
    """
    results = {}
    
    for category, groups in DEMOGRAPHIC_GROUPS.items():
        if category in ['overall']:
            continue
            
        rate_df = pd.DataFrame(index=groups, columns=YEARS)
        
        for group in groups:
            if group in data[category] and 'Yield' in data[category][group].index:
                for year in YEARS:
                    rate = data[category][group].loc['Yield', year]
                    rate_df.loc[group, year] = rate
        
        results[category] = rate_df
    
    return results

def find_significant_trends(data: Dict[str, Dict[str, pd.DataFrame]]) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Identify significant trends in the data using linear regression.
    
    Args:
        data: Nested dictionary of admissions data by demographic category and group
        
    Returns:
        Nested dictionary with trend analysis results
    """
    results = {}
    
    years_numeric = list(range(len(YEARS)))
    
    for category, groups_data in data.items():
        category_results = {}
        
        for group, df in groups_data.items():
            group_results = {}
            
            for metric in METRICS:
                if metric in df.index:
                    values = df.loc[metric, :].values
                    
                    slope, intercept, r_value, p_value, std_err = stats.linregress(years_numeric, values)
                    
                    group_results[metric] = {
                        'slope': slope,
                        'r_squared': r_value**2,
                        'p_value': p_value,
                        'is_significant': p_value < 0.05
                    }
            
            category_results[group] = group_results
        
        results[category] = category_results
    
    return results

def plot_time_series(df: pd.DataFrame, metric: str, title: str = None, ax=None, color=None):
    """
    Plot a time series for a specific metric.
    
    Args:
        df: DataFrame containing admissions data
        metric: Metric to plot ('Applied', 'Offered', 'Enrolled', etc.)
        title: Title for the plot
        ax: Matplotlib axis to plot on
        color: Color for the line
        
    Returns:
        Matplotlib axis object
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    if metric in df.index:
        values = df.loc[metric, YEARS]
        if color:
            ax.plot(YEARS, values, marker='o', linestyle='-', linewidth=2, label=metric, color=color)
        else:
            ax.plot(YEARS, values, marker='o', linestyle='-', linewidth=2, label=metric)
        
        if metric in ['Offered Rate', 'Yield']:
            ax.yaxis.set_major_formatter(mtick.PercentFormatter())
        else:
            ax.get_yaxis().set_major_formatter(mtick.FuncFormatter(lambda x, p: format(int(x), ',')))
        
        if title:
            ax.set_title(title, fontsize=14)
        
        ax.set_xlabel('Academic Year', fontsize=12)
        ax.set_ylabel(metric, fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', rotation=45)
    
    return ax

def plot_demographic_comparison(data: Dict[str, Dict[str, pd.DataFrame]], category: str, metric: str, year: str = None):
    """
    Plot a comparison of a metric across demographic groups within a category.
    
    Args:
        data: Nested dictionary of admissions data by demographic category and group
        category: Demographic category to plot
        metric: Metric to plot ('Applied', 'Offered', 'Enrolled', etc.)
        year: Specific year to plot (if None, will plot the most recent year)
        
    Returns:
        Matplotlib figure
    """
    if category not in DEMOGRAPHIC_GROUPS:
        raise ValueError(f"Category '{category}' not found in data")
    
    if year is None:
        year = YEARS[-1]
    
    if year not in YEARS:
        raise ValueError(f"Year '{year}' not found in data")
    
    groups = DEMOGRAPHIC_GROUPS[category]
    values = []
    labels = []
    
    for group in groups:
        if group in data[category] and metric in data[category][group].index:
            value = data[category][group].loc[metric, year]
            if pd.notna(value):
                values.append(value)
                labels.append(group.replace('_', ' ').title())
    
    if values:
        sorted_indices = np.argsort(values)
        values = [values[i] for i in sorted_indices]
        labels = [labels[i] for i in sorted_indices]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = sns.color_palette("viridis", len(values))
    bars = ax.bar(labels, values, color=colors)
    
    for bar in bars:
        height = bar.get_height()
        if metric in ['Offered Rate', 'Yield']:
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                   f'{height:.1f}%', ha='center', va='bottom', fontsize=10)
        else:
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                   f'{int(height):,}', ha='center', va='bottom', fontsize=10)
    
    if metric in ['Offered Rate', 'Yield']:
        ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    else:
        ax.get_yaxis().set_major_formatter(mtick.FuncFormatter(lambda x, p: format(int(x), ',')))
    
    title = f"{metric} by {category.replace('_', ' ').title()} ({year})"
    ax.set_title(title, fontsize=14)
    ax.set_xlabel(category.replace('_', ' ').title(), fontsize=12)
    ax.set_ylabel(metric, fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    return fig

def plot_time_series_by_group(data: Dict[str, Dict[str, pd.DataFrame]], category: str, metric: str):
    """
    Plot time series for a specific metric across all groups in a category.
    """
    if category not in DEMOGRAPHIC_GROUPS:
        raise ValueError(f"Category '{category}' not found in data")
    
    groups = DEMOGRAPHIC_GROUPS[category]
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    colors = sns.color_palette("viridis", len(groups))
    for i, group in enumerate(groups):
        if group in data[category] and metric in data[category][group].index:
            values = data[category][group].loc[metric, YEARS]
            ax.plot(YEARS, values, marker='o', linestyle='-', linewidth=2, 
                    label=group.replace('_', ' ').title(), color=colors[i])
    
    # Add COVID-19 period indicators
    ax.axvline(x='2019-20', color='red', linestyle='--', alpha=0.7, label='COVID-19 Start')
    ax.axvline(x='2021-22', color='orange', linestyle='--', alpha=0.7, label='COVID-19 End')
    
    # Add shaded area for COVID period
    covid_start_idx = YEARS.index('2019-20')
    covid_end_idx = YEARS.index('2021-22')
    ax.axvspan(YEARS[covid_start_idx], YEARS[covid_end_idx], alpha=0.2, color='red')
    
    if metric in ['Offered Rate', 'Yield']:
        ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    else:
        ax.get_yaxis().set_major_formatter(mtick.FuncFormatter(lambda x, p: format(int(x), ',')))
    
    title = f"{metric} Trends by {category.replace('_', ' ').title()} (2018-2025)"
    ax.set_title(title, fontsize=14)
    ax.set_xlabel('Academic Year', fontsize=12)
    ax.set_ylabel(metric, fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='x', rotation=45)
    ax.legend(loc='best')
    
    plt.tight_layout()
    return fig

def plot_heatmap(df: pd.DataFrame, title: str = None):
    """
    Create a heatmap for a DataFrame.
    
    Args:
        df: DataFrame to visualize
        title: Title for the heatmap
        
    Returns:
        Matplotlib figure
    """
    cmap = sns.diverging_palette(240, 10, as_cmap=True)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    sns.heatmap(df, annot=True, cmap=cmap, center=0, linewidths=0.5, 
                fmt='.1f', ax=ax, cbar_kws={'label': 'Value'})
    
    if title:
        ax.set_title(title, fontsize=14, pad=20)
    
    plt.tight_layout()
    return fig

def plot_admissions_funnel(df: pd.DataFrame, year: str = None):
    """
    Create a funnel chart for admissions process (Applied -> Offered -> Enrolled).
    
    Args:
        df: DataFrame containing admissions data
        year: Specific year to plot (if None, will plot the most recent year)
        
    Returns:
        Matplotlib figure
    """
    if year is None:
        year = YEARS[-1]
    
    if year not in YEARS:
        raise ValueError(f"Year '{year}' not found in data")
    
    applied = df.loc['Applied', year] if 'Applied' in df.index else 0
    offered = df.loc['Offered', year] if 'Offered' in df.index else 0
    enrolled = df.loc['Enrolled', year] if 'Enrolled' in df.index else 0
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    stages = ['Applied', 'Offered', 'Enrolled']
    values = [applied, offered, enrolled]
    
    colors = sns.color_palette("viridis", 3)
    bars = ax.barh(stages, values, color=colors, height=0.6)
    
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax.text(width + (width * 0.01), bar.get_y() + bar.get_height()/2,
               f'{int(width):,}', ha='left', va='center', fontsize=12)
    
    ax.text(offered + (offered * 0.01), bars[1].get_y() + bars[1].get_height()/2 + 0.3,
           f'({offered/applied*100:.1f}% of Applied)', ha='left', va='center', fontsize=10, color='#555555')
    
    ax.text(enrolled + (enrolled * 0.01), bars[2].get_y() + bars[2].get_height()/2 + 0.3,
           f'({enrolled/offered*100:.1f}% of Offered)', ha='left', va='center', fontsize=10, color='#555555')
    
    ax.set_title(f'Admissions Funnel ({year})', fontsize=14)
    ax.set_xlabel('Number of Students', fontsize=12)
    ax.get_xaxis().set_major_formatter(mtick.FuncFormatter(lambda x, p: format(int(x), ',')))
    ax.grid(True, alpha=0.3, axis='x')
    ax.invert_yaxis()
    
    plt.tight_layout()
    return fig

def plot_covid_impact(covid_data: Dict[str, pd.DataFrame], metric: str):
    """
    Plot the impact of COVID-19 on a specific metric across demographic categories.
    """
    if metric not in covid_data:
        raise ValueError(f"Metric '{metric}' not found in COVID data")
    
    df = covid_data[metric]
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    cols_to_plot = ['During vs Pre (%)', 'Post vs During (%)', 'Post vs Pre (%)']
    
    valid_rows = df[cols_to_plot].dropna().index
    
    plot_data = df.loc[valid_rows, cols_to_plot].copy()
    for col in plot_data.columns:
        plot_data[col] = pd.to_numeric(plot_data[col], errors='coerce')
    
    cmap = sns.diverging_palette(240, 10, as_cmap=True)
    
    sns.heatmap(plot_data, annot=True, cmap=cmap, center=0, 
                linewidths=0.5, fmt='.1f', ax=ax)
    
    title = f"COVID-19 Impact on {metric} (% Change)"
    ax.set_title(title, fontsize=14)
    
    plt.tight_layout()
    return fig

def plot_stacked_area(ratio_data: Dict[str, pd.DataFrame], category: str, metric: str):
    """
    Plot stacked area chart showing the composition of demographic groups over time.
    """
    if category not in ratio_data[metric]:
        raise ValueError(f"Category '{category}' not found in ratio data")
    
    df = ratio_data[metric][category]
    
    data_for_plot = df.copy()
    data_for_plot = data_for_plot.fillna(0)

    for col in data_for_plot.columns:
        data_for_plot[col] = pd.to_numeric(data_for_plot[col], errors='coerce')
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    ax.stackplot(YEARS, data_for_plot.values, labels=df.index, alpha=0.8)
    
    title = f"{metric} Demographic Composition: {category.replace('_', ' ').title()} (2018-2025)"
    ax.set_title(title, fontsize=14)
    ax.set_xlabel('Academic Year', fontsize=12)
    ax.set_ylabel('Percentage (%)', fontsize=12)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='x', rotation=45)
    
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    
    plt.tight_layout()
    return fig

def plot_dashboard(data: Dict[str, Dict[str, pd.DataFrame]], year: str = None):
    """
    Create a comprehensive dashboard for the admissions data.
    """
    if year is None:
        year = YEARS[-1]
    
    if year not in YEARS:
        raise ValueError(f"Year '{year}' not found in data")
    
    fig = plt.figure(figsize=(20, 16))
    gs = GridSpec(3, 3, figure=fig)
    
    # 1. Overall admissions funnel - Fix the empty graph issue
    if 'all' in data['overall']:
        ax1 = fig.add_subplot(gs[0, 0])
        
        # Extract the data
        df = data['overall']['all']
        applied = df.loc['Applied', year] if 'Applied' in df.index else 0
        offered = df.loc['Offered', year] if 'Offered' in df.index else 0
        enrolled = df.loc['Enrolled', year] if 'Enrolled' in df.index else 0
        
        # Create bars directly in the current axis
        stages = ['Applied', 'Offered', 'Enrolled']
        values = [applied, offered, enrolled]
        
        colors = sns.color_palette("viridis", 3)
        bars = ax1.barh(stages, values, color=colors, height=0.6)
        
        # Add value labels
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax1.text(width + (width * 0.01), bar.get_y() + bar.get_height()/2,
                   f'{int(width):,}', ha='left', va='center', fontsize=11)
        
        # Add percentage labels
        ax1.text(offered + (offered * 0.01), bars[1].get_y() + bars[1].get_height()/2 + 0.3,
               f'({offered/applied*100:.1f}% of Applied)', ha='left', va='center', fontsize=9, color='#555555')
        
        ax1.text(enrolled + (enrolled * 0.01), bars[2].get_y() + bars[2].get_height()/2 + 0.3,
               f'({enrolled/offered*100:.1f}% of Offered)', ha='left', va='center', fontsize=9, color='#555555')
        
        # Format the plot
        ax1.set_title(f'Admissions Funnel ({year})', fontsize=14)
        ax1.set_xlabel('Number of Students', fontsize=12)
        ax1.get_xaxis().set_major_formatter(mtick.FuncFormatter(lambda x, p: format(int(x), ',')))
        ax1.grid(True, alpha=0.3, axis='x')
        ax1.invert_yaxis()  # Invert y-axis to have funnel shape
    
    # 2. Acceptance rate by race/ethnicity
    if 'race_ethnicity' in data:
        ax2 = fig.add_subplot(gs[0, 1:])
        values = []
        labels = []
        
        for group in DEMOGRAPHIC_GROUPS['race_ethnicity']:
            if group in data['race_ethnicity'] and 'Offered Rate' in data['race_ethnicity'][group].index:
                value = data['race_ethnicity'][group].loc['Offered Rate', year]
                if pd.notna(value):
                    values.append(value)
                    labels.append(group.replace('_', ' ').title())
        
        colors = sns.color_palette("viridis", len(values))
        bars = ax2.bar(labels, values, color=colors)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                   f'{height:.1f}%', ha='center', va='bottom', fontsize=10)
        
        ax2.set_title(f"Acceptance Rate by Race/Ethnicity ({year})", fontsize=14)
        ax2.set_ylabel('Acceptance Rate (%)', fontsize=12)
        ax2.yaxis.set_major_formatter(mtick.PercentFormatter())
        ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. Application trends over time
    ax3 = fig.add_subplot(gs[1, 0:2])
    
    if 'all' in data['overall']:
        df = data['overall']['all']
        values = df.loc['Applied', YEARS]
        ax3.plot(YEARS, values, marker='o', linestyle='-', linewidth=2, color='#4c78a8')
        
        # Add COVID-19 period indicators
        ax3.axvline(x='2019-20', color='red', linestyle='--', alpha=0.7, label='COVID-19 Start')
        ax3.axvline(x='2021-22', color='orange', linestyle='--', alpha=0.7, label='COVID-19 End')
        
        # Add shaded area for COVID period
        covid_start_idx = YEARS.index('2019-20')
        covid_end_idx = YEARS.index('2021-22')
        ax3.axvspan(YEARS[covid_start_idx], YEARS[covid_end_idx], alpha=0.2, color='red')
        
        ax3.set_title(f"Application Trends (2018-2025)", fontsize=14)
        ax3.set_xlabel('Academic Year', fontsize=12)
        ax3.set_ylabel('Applied', fontsize=12)
        ax3.get_yaxis().set_major_formatter(mtick.FuncFormatter(lambda x, p: format(int(x), ',')))
        ax3.grid(True, alpha=0.3)
        ax3.tick_params(axis='x', rotation=45)
        ax3.legend(loc='upper left')
    
    # 4. Gender comparison
    if 'gender' in data:
        ax4 = fig.add_subplot(gs[1, 2])
        males = data['gender']['male'].loc['Enrolled', year] if 'male' in data['gender'] else 0
        females = data['gender']['female'].loc['Enrolled', year] if 'female' in data['gender'] else 0
        
        ax4.pie([males, females], labels=['Male', 'Female'], autopct='%1.1f%%', 
               startangle=90, colors=sns.color_palette("viridis", 2))
        ax4.set_title(f"Enrolled Students by Gender ({year})", fontsize=14)
    
    # 5. Yield rate comparison
    ax5 = fig.add_subplot(gs[2, 0])
    
    if 'residence' in data:
        values = []
        labels = []
        
        for group in DEMOGRAPHIC_GROUPS['residence']:
            if group in data['residence'] and 'Yield' in data['residence'][group].index:
                value = data['residence'][group].loc['Yield', year]
                if pd.notna(value):
                    values.append(value)
                    labels.append(group.replace('_', ' ').title())
        
        colors = sns.color_palette("viridis", len(values))
        bars = ax5.barh(labels, values, color=colors)
        
        # Add value labels
        for bar in bars:
            width = bar.get_width()
            ax5.text(width + 0.5, bar.get_y() + bar.get_height()/2,
                   f'{width:.1f}%', ha='left', va='center', fontsize=10)
        
        ax5.set_title(f"Yield Rate by Residence Status ({year})", fontsize=14)
        ax5.set_xlabel('Yield Rate (%)', fontsize=12)
        ax5.xaxis.set_major_formatter(mtick.PercentFormatter())
        ax5.grid(True, alpha=0.3, axis='x')
    
    # 6. First-gen vs non-first-gen trend
    ax6 = fig.add_subplot(gs[2, 1:])
    
    if 'generation' in data:
        for group, color in zip(['first_gen', 'non_first_gen'], 
                              sns.color_palette("viridis", 2)):
            if group in data['generation'] and 'Enrolled' in data['generation'][group].index:
                values = data['generation'][group].loc['Enrolled', :]
                ax6.plot(YEARS, values, marker='o', linestyle='-', linewidth=2, 
                      label=group.replace('_', ' ').title(), color=color)
        
        # Add COVID-19 period indicators
        ax6.axvline(x='2019-20', color='red', linestyle='--', alpha=0.7, label='COVID-19 Start')
        ax6.axvline(x='2021-22', color='orange', linestyle='--', alpha=0.7, label='COVID-19 End')
        
        # Add shaded area for COVID period
        covid_start_idx = YEARS.index('2019-20')
        covid_end_idx = YEARS.index('2021-22')
        ax6.axvspan(YEARS[covid_start_idx], YEARS[covid_end_idx], alpha=0.2, color='red')
        
        ax6.set_title(f"Enrollment Trends: First Generation vs Non-First Generation", fontsize=14)
        ax6.set_xlabel('Academic Year', fontsize=12)
        ax6.set_ylabel('Number of Students', fontsize=12)
        ax6.get_yaxis().set_major_formatter(mtick.FuncFormatter(lambda x, p: format(int(x), ',')))
        ax6.grid(True, alpha=0.3)
        ax6.tick_params(axis='x', rotation=45)
        ax6.legend(loc='best')
    
    # Set overall title
    fig.suptitle(f"Admissions Dashboard: {year}", fontsize=18, y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return fig

def export_to_csv(df: pd.DataFrame, filename: str):
    """
    Export a DataFrame to a CSV file.
    
    Args:
        df: DataFrame to export
        filename: Output filename
    """
    df.to_csv(filename)
    print(f"Data exported to {filename}")

def export_to_excel(data: Dict[str, Dict[str, pd.DataFrame]], filename: str):
    """
    Export all data to an Excel file with multiple sheets.
    
    Args:
        data: Nested dictionary of admissions data by demographic category and group
        filename: Output filename
    """
    with pd.ExcelWriter(filename) as writer:
        for category, groups_data in data.items():
            for group, df in groups_data.items():
                sheet_name = f"{category}_{group}"
                df.to_excel(writer, sheet_name=sheet_name)
    
    print(f"Data exported to {filename}")

def predict_future_trends(data: Dict[str, Dict[str, pd.DataFrame]], years_ahead: int = 3) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Predict future admission trends using simple linear regression.
    
    Args:
        data: Nested dictionary of admissions data by demographic category and group
        years_ahead: Number of years to predict ahead
        
    Returns:
        Dictionary with predictions for each group and metric
    """
    predictions = {}
    
    x = np.arange(len(YEARS))
    
    future_years = []
    last_year = YEARS[-1]
    start_year, end_year = map(int, last_year.split('-'))
    
    for i in range(1, years_ahead + 1):
        future_start = start_year + i
        future_end = end_year + i
        future_years.append(f"{future_start}-{future_end}")
    
    all_years = YEARS + future_years
    
    for category, groups_data in data.items():
        category_predictions = {}
        
        for group, df in groups_data.items():
            pred_df = pd.DataFrame(index=df.index, columns=all_years)
            
            for year in YEARS:
                pred_df[year] = df[year]
            
            for metric in METRICS:
                if metric in df.index:
                    y = df.loc[metric, :].values
                    
                    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
                    
                    for i, year in enumerate(future_years):
                        future_x = len(YEARS) + i
                        prediction = slope * future_x + intercept
                        
                        if metric in ['Offered Rate', 'Yield']:
                            prediction = max(0, min(100, prediction))
                        else:
                            prediction = max(0, prediction)
                            if metric != 'Offered Rate' and metric != 'Yield':
                                prediction = int(prediction)
                        
                        pred_df.loc[metric, year] = prediction
            
            category_predictions[group] = pred_df
        
        predictions[category] = category_predictions
    
    return predictions

def plot_predictions(predictions: Dict[str, Dict[str, pd.DataFrame]], original_data: Dict[str, Dict[str, pd.DataFrame]], 
                    category: str, group: str, metric: str):
    """
    Plot original data with future predictions for a specific metric.
    
    Args:
        predictions: Dictionary with predictions
        original_data: Original data dictionary
        category: Demographic category to plot
        group: Specific group within the category
        metric: Metric to plot ('Applied', 'Offered', 'Enrolled', etc.)
        
    Returns:
        Matplotlib figure
    """
    if (category not in predictions or group not in predictions[category] or 
        metric not in predictions[category][group].index):
        raise ValueError(f"Data not found for {category}/{group}/{metric}")
    
    pred_df = predictions[category][group]
    orig_df = original_data[category][group]
    
    hist_years = orig_df.columns
    all_years = pred_df.columns
    future_years = [y for y in all_years if y not in hist_years]
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    ax.plot(hist_years, orig_df.loc[metric, :], marker='o', linestyle='-', 
           linewidth=2, color='blue', label='Historical Data')
    
    future_values = pred_df.loc[metric, future_years]
    ax.plot(future_years, future_values, marker='x', linestyle='--', 
           linewidth=2, color='red', label='Predictions')
    
    if metric in ['Offered Rate', 'Yield']:
        ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    else:
        ax.get_yaxis().set_major_formatter(mtick.FuncFormatter(lambda x, p: format(int(x), ',')))
    
    title = f"{metric} Predictions: {group.replace('_', ' ').title()} ({category.replace('_', ' ').title()})"
    ax.set_title(title, fontsize=14)
    ax.set_xlabel('Academic Year', fontsize=12)
    ax.set_ylabel(metric, fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='x', rotation=45)
    ax.legend(loc='best')
    
    ax.axvline(x=hist_years[-1], color='gray', linestyle=':', alpha=0.7)
    ax.text(hist_years[-1], ax.get_ylim()[0], 'Now', ha='center', va='bottom', 
           color='gray', fontsize=10)
    
    plt.tight_layout()
    return fig

def analyze_yield_optimization(data: Dict[str, Dict[str, pd.DataFrame]]) -> pd.DataFrame:
    """
    Analyze yield optimization potential for different demographic groups.
    
    Args:
        data: Nested dictionary of admissions data by demographic category and group
        
    Returns:
        DataFrame with yield optimization analysis
    """
    columns = ['Category', 'Group', 'Avg Yield Rate', 'Trend', 
              'Yield Gap', 'Potential Additional Enrollments']
    results = pd.DataFrame(columns=columns)
    
    row_idx = 0
    for category, groups_data in data.items():
        if category == 'overall':
            continue
            
        overall_yield = None
        if 'all' in data['overall']:
            overall_yield = data['overall']['all'].loc['Yield', YEARS[-3:]].mean()
        
        for group, df in groups_data.items():
            if 'Yield' in df.index and 'Offered' in df.index:
                avg_yield = df.loc['Yield', YEARS[-3:]].mean()
                
                x = np.arange(len(YEARS))
                y = df.loc['Yield', :].values
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
                
                if p_value < 0.05:
                    if slope > 0:
                        trend = 'Increasing'
                    else:
                        trend = 'Decreasing'
                else:
                    trend = 'Stable'
                
                yield_gap = 0
                potential_enrollments = 0
                
                if overall_yield is not None:
                    yield_gap = overall_yield - avg_yield
                    
                    recent_offers = df.loc['Offered', YEARS[-1]]
                    potential_enrollments = int(recent_offers * yield_gap / 100)
                
                results.loc[row_idx] = [
                    category.replace('_', ' ').title(),
                    group.replace('_', ' ').title(),
                    avg_yield,
                    trend,
                    yield_gap,
                    potential_enrollments
                ]
                
                row_idx += 1
    
    results = results.sort_values('Potential Additional Enrollments', ascending=False)
    
    return results

def plot_geographic_trends(data: Dict[str, Dict[str, pd.DataFrame]]):
    """
    Plot geographic trends in admissions (in-state vs. out-of-state vs. international).
    
    Args:
        data: Nested dictionary of admissions data by demographic category and group
        
    Returns:
        Matplotlib figure
    """
    if 'residence' not in data:
        raise ValueError("Residence data not found")
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 15))
    
    ax1 = axes[0]
    for group in ['in_state', 'out_of_state', 'international']:
        if group in data['residence'] and 'Applied' in data['residence'][group].index:
            values = data['residence'][group].loc['Applied', :]
            ax1.plot(YEARS, values, marker='o', linestyle='-', linewidth=2, 
                    label=group.replace('_', ' ').title())
    
    ax1.set_title('Applications by Geographic Origin', fontsize=14)
    ax1.set_ylabel('Number of Applications', fontsize=12)
    ax1.get_yaxis().set_major_formatter(mtick.FuncFormatter(lambda x, p: format(int(x), ',')))
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='best')
    
    ax2 = axes[1]
    for group in ['in_state', 'out_of_state', 'international']:
        if group in data['residence'] and 'Offered Rate' in data['residence'][group].index:
            values = data['residence'][group].loc['Offered Rate', :]
            ax2.plot(YEARS, values, marker='o', linestyle='-', linewidth=2, 
                    label=group.replace('_', ' ').title())
    
    ax2.set_title('Acceptance Rates by Geographic Origin', fontsize=14)
    ax2.set_ylabel('Acceptance Rate (%)', fontsize=12)
    ax2.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='best')
    
    ax3 = axes[2]
    for group in ['in_state', 'out_of_state', 'international']:
        if group in data['residence'] and 'Yield' in data['residence'][group].index:
            values = data['residence'][group].loc['Yield', :]
            ax3.plot(YEARS, values, marker='o', linestyle='-', linewidth=2, 
                    label=group.replace('_', ' ').title())
    
    ax3.set_title('Yield Rates by Geographic Origin', fontsize=14)
    ax3.set_xlabel('Academic Year', fontsize=12)
    ax3.set_ylabel('Yield Rate (%)', fontsize=12)
    ax3.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc='best')
    
    for ax in axes:
        ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    return fig

def analyze_covid_impact_detailed(data: Dict[str, Dict[str, pd.DataFrame]]) -> Dict[str, pd.DataFrame]:
    """
    Perform a detailed analysis of COVID-19 impact on admissions.
    
    Args:
        data: Nested dictionary of admissions data by demographic category and group
        
    Returns:
        Dictionary of DataFrames with detailed analysis for each metric
    """
    results = {}
    
    pre_covid = ['2018-19', '2019-20']
    during_covid = ['2020-21', '2021-22']
    post_covid = ['2022-23', '2023-24', '2024-25']
    
    for metric in METRICS:
        columns = pd.MultiIndex.from_product([
            ['Pre-COVID', 'During-COVID', 'Post-COVID', 'Changes'],
            ['Value', 'Std Dev', 'During vs Pre (%)', 'Post vs During (%)', 'Post vs Pre (%)']
        ])
        result_df = pd.DataFrame(columns=columns)
        
        for category, groups_data in data.items():
            for group, df in groups_data.items():
                if metric in df.index:
                    row_label = f"{category.replace('_', ' ').title()} - {group.replace('_', ' ').title()}"
                    result_row = {}
                    
                    pre_values = df.loc[metric, pre_covid]
                    during_values = df.loc[metric, during_covid]
                    post_values = df.loc[metric, post_covid]
                    
                    pre_avg = pre_values.mean()
                    during_avg = during_values.mean()
                    post_avg = post_values.mean()
                    
                    pre_std = pre_values.std()
                    during_std = during_values.std()
                    post_std = post_values.std()
                    
                    during_vs_pre = ((during_avg - pre_avg) / pre_avg * 100) if pre_avg != 0 else np.nan
                    post_vs_during = ((post_avg - during_avg) / during_avg * 100) if during_avg != 0 else np.nan
                    post_vs_pre = ((post_avg - pre_avg) / pre_avg * 100) if pre_avg != 0 else np.nan
                    
                    result_row[('Pre-COVID', 'Value')] = pre_avg
                    result_row[('Pre-COVID', 'Std Dev')] = pre_std
                    result_row[('During-COVID', 'Value')] = during_avg
                    result_row[('During-COVID', 'Std Dev')] = during_std
                    result_row[('Post-COVID', 'Value')] = post_avg
                    result_row[('Post-COVID', 'Std Dev')] = post_std
                    result_row[('Changes', 'During vs Pre (%)')] = during_vs_pre
                    result_row[('Changes', 'Post vs During (%)')] = post_vs_during
                    result_row[('Changes', 'Post vs Pre (%)')] = post_vs_pre
                    
                    result_df.loc[row_label] = result_row
        
        results[metric] = result_df
    
    return results

def analyze_enrollment_yield_optimization(data: Dict[str, Dict[str, pd.DataFrame]]) -> pd.DataFrame:
    """
    Analyze yield optimization potential by demographic group.
    
    Args:
        data: Nested dictionary of admissions data by demographic category and group
        
    Returns:
        DataFrame with yield optimization strategies
    """
    columns = ['Category', 'Group', 'Current Yield (%)', 'Target Yield (%)', 
              'Gap (%)', 'Current Offers', 'Current Enrolled', 
              'Potential Additional Enrollments', 'Strategy Priority']
    results = pd.DataFrame(columns=columns)
    
    overall_yield = None
    if 'overall' in data and 'all' in data['overall']:
        overall_yield = data['overall']['all'].loc['Yield', YEARS[-1]]
    
    row_idx = 0
    for category, groups_data in data.items():
        if category == 'overall':
            continue
            
        for group, df in groups_data.items():
            if 'Yield' in df.index and 'Offered' in df.index and 'Enrolled' in df.index:
                current_yield = df.loc['Yield', YEARS[-1]]
                current_offers = df.loc['Offered', YEARS[-1]]
                current_enrolled = df.loc['Enrolled', YEARS[-1]]
                
                target_yield = overall_yield if overall_yield is not None else current_yield * 1.1
                
                gap = target_yield - current_yield
                potential_additional = int(current_offers * gap / 100)
                
                if gap > 5:
                    priority = 'High'
                elif gap > 2:
                    priority = 'Medium'
                else:
                    priority = 'Low'
                
                results.loc[row_idx] = [
                    category.replace('_', ' ').title(),
                    group.replace('_', ' ').title(),
                    current_yield,
                    target_yield,
                    gap,
                    current_offers,
                    current_enrolled,
                    potential_additional,
                    priority
                ]
                
                row_idx += 1
    
    results = results.sort_values('Potential Additional Enrollments', ascending=False)
    
    return results

def analyze_geographic_diversity(data: Dict[str, Dict[str, pd.DataFrame]]) -> Dict[str, pd.DataFrame]:
    """
    Analyze geographic diversity trends in admissions.
    
    Args:
        data: Nested dictionary of admissions data by demographic category and group
        
    Returns:
        Dictionary of DataFrames with geographic diversity analysis
    """
    results = {}
    
    if 'residence' not in data:
        raise ValueError("Residence data not found")
    
    metrics_to_analyze = ['Applied', 'Offered', 'Enrolled']
    
    for metric in metrics_to_analyze:
        comp_df = pd.DataFrame(index=YEARS, columns=['In-State (%)', 'Out-of-State (%)', 'International (%)'])
        
        totals = {}
        for year in YEARS:
            year_total = 0
            for group in ['in_state', 'out_of_state', 'international']:
                if group in data['residence'] and metric in data['residence'][group].index:
                    value = data['residence'][group].loc[metric, year]
                    if pd.notna(value):
                        year_total += value
            totals[year] = year_total
        
        for year in YEARS:
            if totals[year] > 0:
                for group, column in zip(['in_state', 'out_of_state', 'international'], 
                                        ['In-State (%)', 'Out-of-State (%)', 'International (%)']):
                    if group in data['residence'] and metric in data['residence'][group].index:
                        value = data['residence'][group].loc[metric, year]
                        if pd.notna(value):
                            comp_df.loc[year, column] = (value / totals[year]) * 100
        
        change_df = pd.DataFrame(index=['Change 2018-2025 (pp)'], 
                                columns=['In-State (%)', 'Out-of-State (%)', 'International (%)'])
        
        for column in comp_df.columns:
            first_value = comp_df.loc[YEARS[0], column]
            last_value = comp_df.loc[YEARS[-1], column]
            if pd.notna(first_value) and pd.notna(last_value):
                change_df.loc['Change 2018-2025 (pp)', column] = last_value - first_value
        
        results[metric] = {
            'composition': comp_df,
            'changes': change_df
        }
    
    return results

def build_predictive_model(data: Dict[str, Dict[str, pd.DataFrame]]) -> Dict:
    """
    Build a predictive model for future admissions.
    
    Args:
        data: Nested dictionary of admissions data by demographic category and group
        
    Returns:
        Dictionary with model parameters and predictions
    """
    model_results = {}
    
    years_numeric = list(range(len(YEARS)))
    
    future_years = []
    last_year = YEARS[-1]
    start_year, end_year = map(int, last_year.split('-'))
    
    for i in range(1, 4):
        future_start = start_year + i
        future_end = end_year + i
        future_years.append(f"{future_start}-{future_end}")
    
    future_x = list(range(len(YEARS), len(YEARS) + len(future_years)))
    
    for category, groups_data in data.items():
        category_results = {}
        
        for group, df in groups_data.items():
            group_results = {}
            
            for metric in METRICS:
                if metric in df.index:
                    metric_results = {}
                    
                    y = df.loc[metric, :].values
                    
                    slope, intercept, r_value, p_value, std_err = stats.linregress(years_numeric, y)
                    
                    metric_results['model'] = {
                        'slope': slope,
                        'intercept': intercept,
                        'r_squared': r_value**2,
                        'p_value': p_value,
                        'std_err': std_err
                    }
                    
                    predictions = {}
                    for i, year in enumerate(future_years):
                        pred_value = slope * future_x[i] + intercept
                        
                        if metric in ['Offered Rate', 'Yield']:
                            pred_value = max(0, min(100, pred_value))
                        else:
                            pred_value = max(0, pred_value)
                            if metric != 'Offered Rate' and metric != 'Yield':
                                pred_value = int(pred_value)
                        
                        predictions[year] = pred_value
                    
                    metric_results['predictions'] = predictions
                    
                    confidence_intervals = {}
                    for i, year in enumerate(future_years):
                        se = std_err * np.sqrt(1 + 1/len(years_numeric) + 
                                             ((future_x[i] - np.mean(years_numeric))**2 / 
                                              sum((x - np.mean(years_numeric))**2 for x in years_numeric)))
                        
                        margin = 1.96 * se
                        lower_bound = predictions[year] - margin
                        upper_bound = predictions[year] + margin
                        
                        if metric in ['Offered Rate', 'Yield']:
                            lower_bound = max(0, min(100, lower_bound))
                            upper_bound = max(0, min(100, upper_bound))
                        else:
                            lower_bound = max(0, lower_bound)
                            upper_bound = max(0, upper_bound)
                            if metric != 'Offered Rate' and metric != 'Yield':
                                lower_bound = int(lower_bound)
                                upper_bound = int(upper_bound)
                        
                        confidence_intervals[year] = {
                            'lower': lower_bound,
                            'upper': upper_bound
                        }
                    
                    metric_results['confidence_intervals'] = confidence_intervals
                    
                    group_results[metric] = metric_results
            
            category_results[group] = group_results
        
        model_results[category] = category_results
    
    return model_results

def prepare_data_summary(data: Dict[str, Dict[str, pd.DataFrame]]) -> pd.DataFrame:
    """
    Prepare a summary of the admissions data for quick reference.
    
    Args:
        data: Nested dictionary of admissions data by demographic category and group
        
    Returns:
        DataFrame with summary statistics
    """
    columns = ['Category', 'Group', 'Applications (Latest)', 'Offers (Latest)', 
              'Enrolled (Latest)', 'Acceptance Rate (%)', 'Yield Rate (%)',
              'Application Growth (%)', 'Enrollment Growth (%)']
    summary = pd.DataFrame(columns=columns)
    
    latest_year = YEARS[-1]
    first_year = YEARS[0]
    
    row_idx = 0
    for category, groups_data in data.items():
        for group, df in groups_data.items():
            if not all(metric in df.index for metric in ['Applied', 'Offered', 'Enrolled']):
                continue
                
            applications = df.loc['Applied', latest_year]
            offers = df.loc['Offered', latest_year]
            enrolled = df.loc['Enrolled', latest_year]
            
            acceptance_rate = df.loc['Offered Rate', latest_year] if 'Offered Rate' in df.index else (offers / applications * 100)
            yield_rate = df.loc['Yield', latest_year] if 'Yield' in df.index else (enrolled / offers * 100)
            
            app_first = df.loc['Applied', first_year]
            app_latest = df.loc['Applied', latest_year]
            app_growth = ((app_latest - app_first) / app_first * 100) if app_first > 0 else np.nan
            
            enr_first = df.loc['Enrolled', first_year]
            enr_latest = df.loc['Enrolled', latest_year]
            enr_growth = ((enr_latest - enr_first) / enr_first * 100) if enr_first > 0 else np.nan
            
            summary.loc[row_idx] = [
                category.replace('_', ' ').title(),
                group.replace('_', ' ').title(),
                int(applications),
                int(offers),
                int(enrolled),
                acceptance_rate,
                yield_rate,
                app_growth,
                enr_growth
            ]
            
            row_idx += 1
    
    return summary

def run_comprehensive_analysis(data_dir: str = 'data/admission'):
    """
    Run a comprehensive analysis of the admissions data.
    
    Args:
        data_dir: Directory containing the CSV files
        
    Returns:
        Dictionary with analysis results
    """
    print("Loading data...")
    data = load_all_data(data_dir)
    
    print("Preparing data summary...")
    summary = prepare_data_summary(data)
    
    print("Analyzing acceptance rates...")
    acceptance_rates = compare_acceptance_rates(data)
    
    print("Analyzing yield rates...")
    yield_rates = compare_yield_rates(data)
    
    print("Analyzing demographic ratios...")
    demographic_ratios = calculate_demographic_ratios(data)
    
    print("Analyzing COVID-19 impact...")
    covid_impact = analyze_covid_impact(data)
    covid_impact_detailed = analyze_covid_impact_detailed(data)
    
    print("Analyzing yield optimization potential...")
    yield_optimization = analyze_enrollment_yield_optimization(data)
    
    print("Analyzing geographic diversity...")
    geographic_diversity = analyze_geographic_diversity(data)
    
    print("Building predictive model...")
    predictive_model = build_predictive_model(data)
    
    print("Finding significant trends...")
    significant_trends = find_significant_trends(data)
    
    results = {
        'data': data,
        'summary': summary,
        'acceptance_rates': acceptance_rates,
        'yield_rates': yield_rates,
        'demographic_ratios': demographic_ratios,
        'covid_impact': covid_impact,
        'covid_impact_detailed': covid_impact_detailed,
        'yield_optimization': yield_optimization,
        'geographic_diversity': geographic_diversity,
        'predictive_model': predictive_model,
        'significant_trends': significant_trends
    }
    
    print("Analysis complete!")
    return results