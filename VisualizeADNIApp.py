#Goal: visualize ADNI analyses and data with an interactive app
import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np
from process_data import process_all_data

# # Functions for Plotting
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def plot_distribution_by_diagnosis_plotly(df_filtered, data_column):
    """
    Create an interactive histogram showing the distribution of a specified data column
    across different diagnosis groups using Plotly.
    
    Parameters:
    -----------
    df_filtered : pandas.DataFrame
        The dataframe containing the diagnosis and data column
    data_column : str
        The name of the column to plot (e.g., 'Hippocampus')
        
    Returns:
    --------
    fig : plotly.graph_objects.Figure
        The plotly figure object
    """
    # Specify the order we want
    diagnosis_order = ['CN', 'MCI', 'Dementia']
    colors = ['blue', 'orange', 'red']
    
    # Create a plotly figure
    fig = go.Figure()
    
    # Add histogram traces for each diagnosis in our specified order
    for dx, color in zip(diagnosis_order, colors):
        subset = df_filtered[df_filtered['DX'] == dx]
        # Skip empty subsets
        if len(subset) > 0:
            # Add histogram trace for this diagnosis group
            fig.add_trace(go.Histogram(
                x=subset[data_column].dropna(),
                name=dx,
                marker_color=color,
                opacity=0.6,
                nbinsx=30
            ))
    
    # Update layout for better appearance
    fig.update_layout(
        title=f'Distribution of {data_column} by Diagnosis',
        xaxis_title=f'{data_column} Value',
        yaxis_title='Count',
        barmode='overlay',  # This ensures histograms overlay each other
        legend_title='Diagnosis',
        template='plotly_white',  # Clean template
        height=500,
        width=800
    )
    
    # Return the figure object
    return fig

def plot_diagnosis_comparison_plotly(df_filtered, data_column, error_type='std'):
    """
    Create an interactive point plot with error bars showing the mean values of a specified data column
    across different diagnosis groups using Plotly.
    
    Parameters:
    -----------
    df_filtered : pandas.DataFrame
        The dataframe containing the diagnosis and data column
    data_column : str
        The name of the column to plot (e.g., 'Hippocampus')
    error_type : str, optional
        Type of error to display, either 'std' for standard deviation or 'sem' for standard error
        Default is 'std'
        
    Returns:
    --------
    fig : plotly.graph_objects.Figure
        The plotly figure object
    """
    import numpy as np
    import plotly.graph_objects as go
    
    # Specify order and colors
    diagnosis_order = ['CN', 'MCI', 'Dementia']
    colors = ['blue', 'orange', 'red']

    # Calculate means and errors for each group
    stats = []
    for dx in diagnosis_order:
        subset = df_filtered[df_filtered['DX'] == dx]
        values = subset[data_column].dropna()  # Handle NaN values
        
        # Only calculate statistics if we have data
        if len(values) > 0:
            mean = values.mean()
            std = values.std()
            sem = std / np.sqrt(len(values))
            stats.append({
                'dx': dx, 
                'mean': mean, 
                'sem': sem,
                'std': std,
                'n': len(values)
            })
        else:
            stats.append({
                'dx': dx, 
                'mean': np.nan, 
                'sem': np.nan,
                'std': np.nan,
                'n': 0
            })

    # Create a Plotly figure
    fig = go.Figure()
    
    # X-coordinates for each diagnosis
    x = list(range(len(diagnosis_order)))
    
    # Extract means and errors
    means = [s['mean'] for s in stats]
    errors = [s[error_type] for s in stats]
    
    # Add error bars and points
    for i, (dx, color) in enumerate(zip(diagnosis_order, colors)):
        if not np.isnan(means[i]):
            # Add error bars with points
            fig.add_trace(go.Scatter(
                x=[i],
                y=[means[i]],
                mode='markers',
                name=f"{dx} (n={stats[i]['n']})",
                marker=dict(
                    color=color,
                    size=12,
                    line=dict(
                        color='black',
                        width=1
                    )
                ),
                error_y=dict(
                    type='data',
                    array=[errors[i]],
                    visible=True,
                    thickness=2,
                    width=6
                )
            ))
    
    # Customize the layout
    error_label = 'Standard Error of Mean' if error_type == 'sem' else 'Standard Deviation'
    
    fig.update_layout(
        title=f'Mean {data_column} Value by Diagnosis (±{error_label})',
        xaxis=dict(
            title='Diagnosis',
            ticktext=diagnosis_order,
            tickvals=list(range(len(diagnosis_order))),
            tickmode='array'
        ),
        yaxis=dict(
            title=f'{data_column} Value'
        ),
        template='plotly_white',
        height=500,
        width=800,
        showlegend=True,
        legend=dict(
            title='Diagnosis Group'
        )
    )
    
    # Add grid lines
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.1)')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.1)')
    
    # Create a table of summary statistics to return alongside the chart
    summary_stats = {}
    for s in stats:
        if s['n'] > 0:
            summary_stats[s['dx']] = {
                'n': s['n'],
                'Mean': round(s['mean'], 2),
                'SEM': round(s['sem'], 2),
                'SD': round(s['std'], 2)
            }
        else:
            summary_stats[s['dx']] = {
                'n': 0,
                'Mean': 'No valid data',
                'SEM': 'No valid data',
                'SD': 'No valid data'
            }
    
    return fig, summary_stats

def plot_correlation_by_diagnosis_plotly(df_filtered, data_column1, data_column2):
    """
    Create an interactive correlation plot showing the relationship between two 
    data columns across different diagnosis groups using Plotly.
    
    Parameters:
    -----------
    df_filtered : pandas.DataFrame
        The dataframe containing the diagnosis and data column
    data_column1 : str
        The name of the X column to plot (e.g., 'Hippocampus')
    data_column2 : str
        The name of the Y column to plot (e.g., 'Age')
        
    Returns:
    --------
    fig : plotly.graph_objects.Figure
        The plotly figure object
    """
    import numpy as np
    import plotly.graph_objects as go
    from scipy import stats
    
    # Specify order and colors
    diagnosis_order = ['CN', 'MCI', 'Dementia']
    colors = ['blue', 'orange', 'red']

    # Create a Plotly figure
    fig = go.Figure()

    # Add traces for each diagnosis group
    for i, dx in enumerate(diagnosis_order):
        subset = df_filtered[df_filtered['DX'] == dx]
        # Extract values and handle NaN values by using dropna
        valid_data = subset[[data_column1, data_column2]].dropna()
        
        # Only proceed if we have enough valid data points
        if len(valid_data) > 1:  # Need at least 2 points for regression
            values1 = valid_data[data_column1]
            values2 = valid_data[data_column2]
            
            # Calculate linear regression
            slope, intercept, r_value, p_value, std_err = stats.linregress(values1, values2)
            r_squared = r_value**2
            
            # Add scatter points
            fig.add_trace(go.Scatter(
                x=values1,
                y=values2,
                mode='markers',
                marker=dict(
                    color=colors[i],
                    size=8,
                    opacity=0.7
                ),
                name=f"{dx}: R² = {r_squared:.2f}"
            ))
            
            # Create trend line data
            x_range = np.linspace(min(values1), max(values1), 100)
            y_pred = slope * x_range + intercept
            
            # Add trend line
            fig.add_trace(go.Scatter(
                x=x_range,
                y=y_pred,
                mode='lines',
                line=dict(color=colors[i], width=2),
                showlegend=False  # Don't show in legend to avoid duplication
            ))
            
        elif len(valid_data) == 1:
            # If only one valid point, just plot it without regression
            values1 = valid_data[data_column1]
            values2 = valid_data[data_column2]
            
            fig.add_trace(go.Scatter(
                x=values1,
                y=values2,
                mode='markers',
                marker=dict(
                    color=colors[i],
                    size=8,
                    opacity=0.7
                ),
                name=f"{dx}: insufficient data"
            ))
        else:
            # No data for this diagnosis group - add empty trace just for the legend
            fig.add_trace(go.Scatter(
                x=[None],
                y=[None],
                mode='markers',
                marker=dict(color=colors[i]),
                name=f"{dx}: no valid data"
            ))
 
    # Customize the layout
    fig.update_layout(
        title=f'Correlation between {data_column1} and {data_column2} by Diagnosis',
        xaxis_title=f'{data_column1} Value',
        yaxis_title=f'{data_column2} Value',
        template='plotly_white',
        height=600,
        width=800,
        legend=dict(
            title='Diagnosis Group',
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99
        )
    )
    
    # Add grid lines
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.1)')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.1)')

    return fig
# # End of functions for plotting

# Page config
st.set_page_config(page_title="ADNI: Visualized", layout="wide")

# Title and introduction
st.title('Interactive Data Visualization for ADNI')
st.write('This app visualizes processed imaging/clinical features from the ADNI dataset, focusing on Hippocampal CBF.')

# Cache the data processing function
@st.cache_data
def load_processed_data():
    return process_all_data()

# Get all DataFrames - will only run once
df_main, df_filtered, df_cbf, df_cbf_logical, df_filtered_logical, df_filtered_forCN, df_filtered_forMCI, df_filtered_forRecovery = load_processed_data()

# Create sidebar for datasets
st.sidebar.header('Dataset Selection')

# Add a section for dataset selection
st.subheader('Select Dataset to Visualize')

# Create a dictionary with dataset options and their descriptions
dataset_options = {
    'Main Dataset': 'Complete pre-processed dataset with all subjects and variables',
    'Filtered Dataset': 'Dataset with non-diagnosis timepoints, and "chaotic subjects", removed',
    'CBF Dataset': 'Filtered Dataset containing only visits with Cerebral Blood Flow measurements',
    'CBF Logical Dataset': 'CBF dataset with cognitively stable and declining subjects included',
    'Filtered Logical Dataset': 'Filtered dataset with cognitively stable and declining subjects included',
    'CN Filtered Dataset': 'Dataset filtered for Stable CN and CN to MCI subjects only',
    'MCI Filtered Dataset': 'Dataset filtered for Stable MCI and MCI to Dementia subjects only',
    'Recovery Filtered Dataset': 'Dataset filtered for MCI to CN recovery subjects only'
}

# Dictionary mapping option names to actual dataframes
dataset_mapping = {
    'Main Dataset': df_main,
    'Filtered Dataset': df_filtered,
    'CBF Dataset': df_cbf,
    'CBF Logical Dataset': df_cbf_logical,
    'Filtered Logical Dataset': df_filtered_logical,
    'CN Filtered Dataset': df_filtered_forCN,
    'MCI Filtered Dataset': df_filtered_forMCI,
    'Recovery Filtered Dataset': df_filtered_forRecovery
}

# Create two columns for the selection box and description
col1, col2 = st.columns([1, 2])

# Add the selection box in the first column
with col1:
    selected_dataset = st.selectbox(
        'Choose a dataset:',
        options=list(dataset_options.keys())
    )

# Display the description in the second column
with col2:
    st.info(dataset_options[selected_dataset])

# Get the selected dataframe
selected_df = dataset_mapping[selected_dataset]

# Now you can use selected_df for your visualizations
st.subheader(f'Visualization for {selected_dataset}')

# Create tabs for different visualizations
tab1, tab2, tab3 = st.tabs(["Dot Plot", "Histogram", "Correlation"])

with tab1:
    st.subheader('Interactive Dot Plot')
    
    # Check if the selected dataframe has the required column
    if 'DX' in selected_df.columns:
        # Get appropriate numeric columns for visualization
        numeric_columns = selected_df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        available_columns = [col for col in numeric_columns if selected_df[col].notna().sum() > 0]
        
        # Column selector
        selected_column = st.selectbox(
            'Select column to visualize:',
            options=available_columns,
            key='dotplot_column_selector'  # Unique key to avoid conflicts
        )
        
        # Error type selector
        error_type = st.radio(
            "Select error type to display:",
            options=["std", "sem"],
            format_func=lambda x: "Standard Deviation" if x == "std" else "Standard Error of Mean",
            horizontal=True,
            key='error_type_selector'  # Unique key
        )
        
        # Generate the plot
        if st.button('Generate Dot Plot', key='generate_dotplot_button'):
            # Create the plot
            fig, summary_stats = plot_diagnosis_comparison_plotly(selected_df, selected_column, error_type)
            
            # Display the plot
            st.plotly_chart(fig, use_container_width=True)
            
            # Display the summary statistics as a nice table
            st.subheader(f'Summary Statistics for {selected_column}')
            
            # Convert summary stats to a DataFrame for better display
            stats_df = pd.DataFrame.from_dict(summary_stats, orient='index')
            st.dataframe(stats_df)
    else:
        st.warning("The selected dataset doesn't contain diagnosis information (DX column) required for this visualization.")

with tab2:
    st.subheader('Distribution Histogram')
    
    # Check which columns are available in the selected dataset
    numeric_columns = selected_df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    available_columns = [col for col in numeric_columns if selected_df[col].notna().sum() > 0]

    # Only show visualization options if we have DX column and some numeric columns
    if 'DX' in selected_df.columns and len(available_columns) > 0:
        # Let user select which column to visualize
        selected_column = st.selectbox(
            'Select column to visualize:',
            options=available_columns
        )
        
        # Create the visualization
        if st.button('Generate Distribution Plot'):
            st.subheader(f'Distribution of {selected_column} by Diagnosis')
            
            # Create and display the plotly figure
            fig = plot_distribution_by_diagnosis_plotly(selected_df, selected_column)
            st.plotly_chart(fig, use_container_width=True)
            
            # Display summary statistics
            st.subheader(f'Summary Statistics for {selected_column} by Diagnosis')
            st.dataframe(selected_df.groupby('DX')[selected_column].describe())
    else:
        st.warning('The selected dataset does not have the required columns for this visualization.')

with tab3:
    st.subheader('Correlation')
    # plot_correlation_by_diagnosis_plotly(df_filtered, data_column1, data_column2)
    # Check which columns are available in the selected dataset
    numeric_columns = selected_df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    available_columns = [col for col in numeric_columns if selected_df[col].notna().sum() > 0]

    # Only show visualization options if we have DX column and two or more numeric columns
    if 'DX' in selected_df.columns and len(available_columns) > 1:
        # Let user select which column to visualize
        selected_column1 = st.selectbox(
            'Select column to visualize on X axis:',
            options=available_columns,
            key='correlation_x_column'  # Added unique key
        )
        # Set default for second dropdown to be different from first
        default_index = 0 if len(available_columns) > 1 else 0
        selected_column2 = st.selectbox(
            'Select column to visualize on Y axis:',
            options=available_columns,
            index=1 if len(available_columns) > 1 else 0,  # Default to second column if available
            key='correlation_y_column'
        )
        # Create the visualization
        if st.button('Generate Correlation Plot', key='generate_correlation_button'):  # Added unique key
            st.subheader(f'Correlation of {selected_column1} by {selected_column2}, by Diagnosis')
            
            # Create and display the plotly figure
            fig = plot_correlation_by_diagnosis_plotly(selected_df, selected_column1, selected_column2)
            st.plotly_chart(fig, use_container_width=True)
            
            # Display summary statistics 1
            st.subheader(f'Summary Statistics for {selected_column1} by Diagnosis')
            st.dataframe(selected_df.groupby('DX')[selected_column1].describe())
            # Display summary statistics 2
            st.subheader(f'Summary Statistics for {selected_column2} by Diagnosis')
            st.dataframe(selected_df.groupby('DX')[selected_column2].describe())
    else:
        st.warning('The selected dataset does not have the required columns for this visualization.')