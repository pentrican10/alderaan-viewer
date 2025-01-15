import numpy as np
import plotly.graph_objects as go

# Example data (actual values) and model predictions
x = np.linspace(0, 10, 100)
y_data = np.sin(x)  # Example data (sin curve)

# Generate multiple models (for example, different versions of the model)
models = [
    0.9 * np.sin(x),  # Model 1
    0.8 * np.sin(x),  # Model 2
    1.1 * np.sin(x)   # Model 3
]

# Calculate residuals for each model
residuals = [y_data - model for model in models]

# Define traces dynamically for the models and residuals
data = []
num_models = len(models)

# Create traces for data and each model
for i in range(num_models):
    trace_data = go.Scatter(
        x=x,
        y=y_data,
        mode='markers',
        name="Data",
        marker=dict(color="blue"),
        yaxis=f"y{i*2+1}"  # Assign to unique y-axis for data (odd axes)
    )

    trace_model = go.Scatter(
        x=x,
        y=models[i],
        mode='lines',
        name=f"Model {i+1}",
        line=dict(color=f"rgb({255-i*50},{100+i*50},200)"),  # Different colors for models
        yaxis=f"y{i*2+1}"  # Assign to unique y-axis for model
    )

    trace_residuals = go.Scatter(
        x=x,
        y=residuals[i],
        mode='markers',
        name=f"Residuals {i+1}",
        marker=dict(color="green"),
        yaxis=f"y{i*2+2}"  # Assign to unique y-axis for residuals (even axes)
    )

    data.extend([trace_data, trace_model, trace_residuals])

# Create layout dynamically based on the number of models
y_axes = []
num_axes = num_models * 2  # Each model gets two axes (one for model and one for residuals)
axis_height = 1 / num_axes  # Height available per axis

# Create y-axes for models and residuals, ensuring correct order
for i in range(num_models):
    # Y-axis for models (odd numbered y-axes)
    y_axes.append(dict(
        title=f"Model {i+1}",
        domain=[(i * 2) * axis_height, (i * 2 + 1) * axis_height],  # Space them out evenly
        anchor=f"x{i+1}",
    ))

    # Y-axis for residuals (even numbered y-axes)
    y_axes.append(dict(
        title=f"Residuals {i+1}",
        domain=[(i * 2 + 1) * axis_height, (i * 2 + 2) * axis_height],
        anchor=f"x{i+1}",
    ))

# Define x-axis (shared for all plots)
x_axis = dict(
    domain=[0, 1],  # Full width for the x-axis
)

# Set up layout for the figure
layout = go.Layout(
    title="Data, Models, and Residuals",
    xaxis=x_axis,
    # Dynamically add multiple y-axes for models and residuals
    **{f"yaxis{i+1}": y_axes[i] for i in range(len(y_axes))}
)

# Create the figure and show it
fig = go.Figure(data=data, layout=layout)
fig.show()