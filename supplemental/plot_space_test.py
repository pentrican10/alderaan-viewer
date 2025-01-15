
'''
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# Create subplots
rows = 6
subplot_titles = ['mod1', 'res1', 'mod2', 'res2', 'mod3', 'res3']
fig = make_subplots(rows=rows, cols=1,subplot_titles=subplot_titles)

# Create a dictionary to hold dynamic yaxis settings
yaxis_dict = {}

systems = int(rows/2)
space_per_system = 1/systems # system contains the main plot and the residual plot
space_above = 0.15 * space_per_system
space_between_systems = 0.05 * space_per_system
plot_space = 0.60 * space_per_system 
residual_space = 0.20 * space_per_system
system_domains = [[0,space_per_system]]
for i in range(0,systems-1):
    sys_dom = system_domains[i][1] + space_per_system
    system_domains.append([system_domains[i][1], sys_dom])
print(system_domains)
print(plot_space)
print(residual_space)
print(space_above)
plot_domains = [[0,residual_space]]
yaxis_dict[f'yaxis{1}'] = [0, plot_domains[0][1]]
print(plot_domains[0][1])
for i in range(0,rows-1):
    if i==rows-2:
        domain_ = plot_domains[i][1] + plot_space
        dom = [plot_domains[i][1]+space_above/3, domain_]
    elif (i % 2 == 0):
        domain_ =  plot_domains[i][1] + plot_space
        dom = [plot_domains[i][1] + space_above/3, domain_]
    else:
        domain_ = space_above + plot_domains[i][1] + residual_space
        dom = [plot_domains[i][1] + space_above, domain_]
    
    plot_domains.append(dom)
    # Store the domain in the yaxis_dict dynamically
    yaxis_dict[f'yaxis{i+2}'] = dom



print(plot_domains)

print(yaxis_dict)


traces_dict = {}
traces_dict[f'trace1'] = go.Scatter(x=[1, 2, 3], y=[4, 5, 6])
traces_dict[f'trace2'] = go.Scatter(x=[1, 2, 3], y=[6, 5, 4])
traces_dict[f'trace3'] = go.Scatter(x=[1, 2, 3], y=[7, 8, 9])
traces_dict[f'trace4'] = go.Scatter(x=[1, 2, 3], y=[9, 8, 7])
traces_dict[f'trace5'] = go.Scatter(x=[1, 2, 3], y=[9, 10, 11])
traces_dict[f'trace6'] = go.Scatter(x=[1, 2, 3], y=[11, 10, 9])


# Loop to dynamically update layout for each subplot
for i in range(1, rows+1):
    fig.add_trace(traces_dict[f'trace{i}'], row=i, col=1)
    fig.update_layout(
        **{
            f'xaxis{i}': dict(domain=[0, 1]),  # Horizontal span: 0 to 1
            f'yaxis{i}': dict(domain=yaxis_dict[f'yaxis{rows+1-i}'])  # Use dynamic yaxis from dictionary
        }
    )
    if not (i %2 ==0):
        fig.update_xaxes(showticklabels=False, row=i, col=1)

# Update layout
fig.update_layout(height=800, width=800, title_text="Manual Subplot Positioning")
fig.show()
'''

import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Create subplots
rows = 6
subplot_titles = ['mod1', 'res1', 'mod2', 'res2', 'mod3', 'res3']
fig = make_subplots(rows=rows, cols=1)

# Create a dictionary to hold dynamic yaxis settings
yaxis_dict = {}

systems = int(rows / 2)
space_per_system = 1 / systems  # system contains the main plot and the residual plot
space_above = 0.10 * space_per_system
space_between_systems = 0.05 * space_per_system
plot_space = 0.65 * space_per_system
residual_space = 0.20 * space_per_system
system_domains = [[0, space_per_system]]

# Generate system domains for the subplots
for i in range(0, systems - 1):
    sys_dom = system_domains[i][1] + space_per_system
    system_domains.append([system_domains[i][1], sys_dom])
print(system_domains)

# Initialize the first plot domain
plot_domains = [[0, residual_space]]
yaxis_dict[f'yaxis{1}'] = [0, plot_domains[0][1]]

# Generate plot domains and yaxis settings
for i in range(0, rows - 1):
    if i == rows - 2:
        domain_ = space_between_systems + plot_domains[i][1] + plot_space
        dom = [space_between_systems + plot_domains[i][1], 1]
    elif (i % 2 == 0):
        domain_ = space_between_systems + plot_domains[i][1] + plot_space
        dom = [space_between_systems + plot_domains[i][1], domain_]
    else:
        domain_ = space_above + plot_domains[i][1] + residual_space
        dom = [space_above + plot_domains[i][1], domain_]
    plot_domains.append(dom)
    yaxis_dict[f'yaxis{i + 2}'] = dom

print(plot_domains)

# Set up traces
traces_dict = {}
traces_dict[f'trace1'] = go.Scatter(x=[1, 2, 3], y=[4, 5, 6])
traces_dict[f'trace2'] = go.Scatter(x=[1, 2, 3], y=[6, 5, 4])
traces_dict[f'trace3'] = go.Scatter(x=[1, 2, 3], y=[7, 8, 9])
traces_dict[f'trace4'] = go.Scatter(x=[1, 2, 3], y=[9, 8, 7])
traces_dict[f'trace5'] = go.Scatter(x=[1, 2, 3], y=[9, 10, 11])
traces_dict[f'trace6'] = go.Scatter(x=[1, 2, 3], y=[11, 10, 9])

# Loop to dynamically update layout for each subplot
for i in range(1, rows + 1):
    fig.add_trace(traces_dict[f'trace{i}'], row=i, col=1)
    fig.update_layout(
        **{
            f'xaxis{i}': dict(domain=[0, 1]),  # Horizontal span: 0 to 1
            f'yaxis{i}': dict(domain=yaxis_dict[f'yaxis{rows + 1 - i}'])  # Use dynamic yaxis from dictionary
        }
    )
    if not (i % 2 == 0):
        fig.update_xaxes(showticklabels=False, row=i, col=1)

# Add manual titles using annotations in figure coordinates
# Reverse the order of subplot titles
reversed_subplot_titles = list(reversed(subplot_titles))



# Update layout
fig.update_layout(
    height=800, width=800, title_text="Manual Subplot Positioning",
    showlegend=False  # Hide legend if not needed
)

fig.show()