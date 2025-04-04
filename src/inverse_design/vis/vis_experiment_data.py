import pandas as pd
import plotly.graph_objects as go
import numpy as np
from scipy.stats import gaussian_kde

# Read the NCI60 dataset
df = pd.read_csv('../../../NCI60.csv')

# Define the panel order to match the figure
panel_order = [
    'All Cell Lines',  # New entry for density plot
    'Leukemia',
    'Lung',  # Using simplified name
    'Colon',
    'CNS',
    'Melanoma',
    'Ovarian',
    'Renal',
    'Prostate',
    'Breast'
]

# Map from original panel names to simplified panel names
panel_mapping = {
    'Non-Small Cell Lung': 'Lung',
    'Leukemia': 'Leukemia',
    'Colon': 'Colon',
    'CNS': 'CNS',
    'Melanoma': 'Melanoma',
    'Ovarian': 'Ovarian',
    'Renal': 'Renal',
    'Prostate': 'Prostate',
    'Breast': 'Breast'
}

# Map panel names in the dataframe
df['Panel'] = df['Panel Name'].map(lambda x: panel_mapping.get(x, x))

# Create figure with subplots
fig = go.Figure()

# Define y-axis positions for each panel (bottom to top)
y_positions = {panel: i for i, panel in enumerate(panel_order)}

# Set the x range
x_min, x_max = 10, 85
x_grid = np.linspace(x_min, x_max, 300)

# Add horizontal grid lines
for i in range(len(panel_order)):
    fig.add_shape(
        type="line",
        x0=x_min,
        y0=i,
        x1=x_max,
        y1=i,
        line=dict(
            color="lightgray",
            width=1,
        ),
    )

# Add vertical grid lines
for x_val in [30, 50, 70]:
    fig.add_shape(
        type="line",
        x0=x_val,
        y0=0,
        x1=x_val,
        y1=len(panel_order)-1,
        line=dict(
            color="lightgray",
            width=1,
        ),
    )

# First add the density plot at the top position
all_doubling_times = df['Doubling Time'].dropna().values
kde = gaussian_kde(all_doubling_times, bw_method='silverman')
y_kde = kde(x_grid)
# Scale the KDE to fit nicely in the row
scale_factor = 0.5 / np.max(y_kde)  # Scale to fit within the y-axis range
y_kde_scaled = y_kde * scale_factor
base_position = y_positions['All Cell Lines']

fig.add_trace(
    go.Scatter(
        x=x_grid,
        y=[base_position + y for y in y_kde_scaled],
        fill='tonexty',
        mode='none',
        fillcolor='rgba(100, 100, 100, 0.5)',
        showlegend=False,
        hoverinfo='skip'
    )
)

# Then add scatter plots for each cancer panel (excluding 'All Cell Lines')
for panel in panel_order[1:]:  # Skip the first entry which is for density plot
    panel_data = df[df['Panel'] == panel]
    
    fig.add_trace(
        go.Scatter(
            x=panel_data['Doubling Time'],
            y=[y_positions[panel]] * len(panel_data),
            mode='markers',
            name=panel,
            showlegend=False,
            marker=dict(
                color='rgba(100, 100, 100, 0.8)',
                size=6
            ),
            text=[f"Cell Line: {row['Cell Line Name']}<br>Doubling Time: {row['Doubling Time']} hours<br>Inoculation Density: {row['Inoculation Density']}" 
                  for _, row in panel_data.iterrows()],
            hoverinfo='text'
        )
    )

# Update layout
fig.update_layout(
    title=None,
    xaxis=dict(
        title='Doubling Time (hours)',
        range=[x_min, x_max],
        tickvals=[10, 30, 50, 70],
        ticktext=['10', '30', '50', '70'],
        gridcolor='white',
        zerolinecolor='white',
        ticks='outside',
    ),
    yaxis=dict(
        range=[-0.5, len(panel_order) - 0.5],
        tickvals=list(y_positions.values()),
        ticktext=panel_order,
        gridcolor='white',
        zeroline=False,
        ticks='',
    ),
    plot_bgcolor='white',
    width=800,
    height=650,  # Slightly increased height to accommodate the density plot
    margin=dict(l=120, r=20, t=50, b=50),
    hovermode='closest',
    showlegend=False
)

# Save the figure
fig.write_html("NCI60_doubling_time_interactive.html")

# Show the figure
fig.show()
print("The plot is saved as 'NCI60_doubling_time_interactive.html'")