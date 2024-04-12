import numpy as np
import plotly.graph_objs as go
from sklearn.neighbors import KernelDensity

# Generate sample data
np.random.seed(0)
x = np.random.randn(1000)
y = np.random.randn(1000)

# Combine x and y into a single array
data = np.vstack([x, y]).T

# Perform kernel density estimation
kde = KernelDensity(bandwidth=0.5, kernel='gaussian')
kde.fit(data)

# Generate grid points for density estimation
x_grid, y_grid = np.meshgrid(np.linspace(min(x), max(x), 100),
                              np.linspace(min(y), max(y), 100))
grid_points = np.vstack([x_grid.ravel(), y_grid.ravel()]).T

# Calculate density values for the grid points
log_dens = kde.score_samples(grid_points)
dens = np.exp(log_dens)

# Calculate threshold for the top 90% percentile based on density
threshold = np.percentile(dens, 90)

# Separate data into top 90% and last 10% percentiles based on density
x_top = grid_points[dens >= threshold, 0]
y_top = grid_points[dens >= threshold, 1]

# Sub-sample scatter points for last 10% percentile data
sample_indices = np.random.choice(np.sum(dens < threshold), size=min(100, np.sum(dens < threshold)), replace=False)
x_bottom = grid_points[dens < threshold][sample_indices, 0]
y_bottom = grid_points[dens < threshold][sample_indices, 1]

# Create density contour plot for top 90% percentile data
density_contour = go.Contour(x=x_top, y=y_top, z=dens[dens >= threshold], colorscale='Viridis', showscale=False)

# Add scatter points for last 10% percentile data
scatter_points = go.Scatter(x=x_bottom, y=y_bottom, mode='markers', marker=dict(color='red', size=5), name='Last 10%')

# Create layout
layout = go.Layout(title='Density Contour Plot with Scatter for Last 10% Percentile (based on density)',
                   xaxis=dict(title='X'),
                   yaxis=dict(title='Y'))

# Plot
fig = go.Figure(data=[density_contour, scatter_points], layout=layout)
fig.show()


'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from   scipy import stats
import seaborn as sns

# simulate data
N = int(1e5)
mu = np.random.uniform(1,2,size=2)
cov = np.eye(2) + np.random.uniform(0,1)*(1.0-np.eye(2))

x, y = stats.multivariate_normal(mu, cov).rvs(size=N).T
df = pd.DataFrame({'x':x, 'y':y})

# make 2D density plot
thin = 100

X = df.x[::thin].values
Y = df.y[::thin].values

density = stats.gaussian_kde([X,Y])([df.x,df.y])
cutoff  = 4
low_density = density < np.percentile(density, cutoff)

plt.figure()
sns.kdeplot(df[::thin], x='x', y='y', fill=True, levels=4)
plt.scatter(df.x[low_density], df.y[low_density], color='C0', s=3)
plt.show()
'''


"""
import os
import numpy as np
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from astropy.io import fits
import plotly.figure_factory as ff
from scipy.stats import gaussian_kde

PROJECT_DIR = 'c:\\Users\\Paige\\Projects\\data\\alderaan_results'

file = os.path.join(PROJECT_DIR, '2023-05-19_singles\\S00333\\S00333-results.fits')

# def load_posteriors(f):
#     with fits.open(f) as hduL:
#         data = hduL['SAMPLES'].data
#         keys = data.names
#         _posteriors = []
#         for k in keys:
#             _posteriors.append(data[k])

#         # LD_U1 = np.ones(len(data.C0_0))
#         # # Add calculated values as new columns
#         # keys.append('LD_U1')
#         # _posteriors.append([LD_U1])
#         return pd.DataFrame(np.array(_posteriors).T, columns=keys)


def load_posteriors(f,n,koi_id):
    ''' gets params for planet number n
        f = file path
        n = planet number
        koi_id = koi id

    '''
    with fits.open(f) as hduL:
        data = hduL['SAMPLES'].data
        C0 = data[f'C0_{n}']
        C1 = data[f'C1_{n}']
        ROR = data[f'ROR_{n}']
        IMPACT = data[f'IMPACT_{n}']
        DUR14 = data[f'DUR14_{n}']
        LD_Q1 = data[f'LD_Q1']
        LD_Q2 = data[f'LD_Q2']
        LN_WT = data[f'LN_WT']
        LN_LIKE = data[f'LN_LIKE']

        ### calculate P, T0, U1, U2
        LD_U1 = 2*np.sqrt(LD_Q1)*LD_Q2
        LD_U2 = np.sqrt(LD_Q1)*(1-2*LD_Q2)

        data_return = np.vstack([C0, C1, ROR, IMPACT, DUR14, LD_Q1, LD_Q2, LD_U1, LD_U2, LN_WT, LN_LIKE]).T
        labels = f'C0_{n} C1_{n} ROR_{n} IMPACT_{n} DUR14_{n} LD_Q1 LD_Q2 LD_U1 LD_U2 LN_WT LN_LIKE'.split()
        df = pd.DataFrame(data_return, columns=labels)
        return df




# with fits.open(file) as hduL:
#         data = hduL['SAMPLES'].data
#         keys = data.names
#         print(data['C0_0'])
#         #print(keys)
#         print(data)


data = load_posteriors(file,0,2)
print(data['LN_WT'].mean())
assert 1==0
selected_columns = ['C0_0','C1_0','ROR_0','IMPACT_0','DUR14_0','LD_U1','LD_Q1']

data = data[selected_columns]

labels = data.columns.tolist()

fig = make_subplots(rows=len(selected_columns), cols=len(selected_columns))

for i in range(len(selected_columns)):
    for j in range(i, len(selected_columns)):
        # x = data[selected_columns[i]]
        # y = data[selected_columns[j]]
        x = data[selected_columns[i]][::5]
        y = data[selected_columns[j]][::5]

        if i != j:
            # x = data[selected_columns[i]][::30]
            # y = data[selected_columns[j]][::30]
            fig.add_trace(go.Scatter(x=x, y=y, mode='markers', marker=dict(color='gray', size=1), showlegend=False), row=j + 1, col=i + 1)
            fig.add_trace(go.Histogram2dContour(x=x, y=y, colorscale='Blues', reversescale=False, showscale=False, ncontours=4, contours=dict(coloring='fill'), line=dict(width=1)), row=j + 1, col=i + 1)
        else:
            kde = gaussian_kde(x)
            x_vals = np.linspace(min(x), max(x), 1000)
            y_vals = kde(x_vals)
            fig.add_trace(go.Scatter(x=x_vals, y=y_vals, mode='lines', line=dict(color='blue'), name=labels[i]), row=j + 1, col=i + 1)

        if (i == 0) and (i != j):
            fig.update_yaxes(title_text=labels[j], row=j + 1, col=i + 1)
        if j == len(selected_columns) - 1:
            fig.update_xaxes(title_text=labels[i], row=j + 1, col=i + 1)

        fig.update_xaxes(showline=True, linewidth=1, linecolor='black', mirror=True, row=j + 1, col=i + 1)
        fig.update_yaxes(showline=True, linewidth=1, linecolor='black', mirror=True, row=j + 1, col=i + 1)

fig.show()
"""

'''
Nvar = 5  # Set the number of variables to 5

# Slice the DataFrame to include only the first 5 columns
data = data.iloc[:, :Nvar]
# Subsample the data to every 100th data point
#data = data.iloc[::30, :]
labels = data.columns.tolist()

fig = make_subplots(rows=Nvar, cols=Nvar)

for i in range(1, Nvar + 1):
    for j in range(i, Nvar + 1):
        x = data.iloc[:, i - 1]
        y = data.iloc[:, j - 1]

        # plot the data
        if i != j:
            x = data.iloc[::30, i-1]
            y=data.iloc[::30, j-1]
            fig.add_trace(go.Scatter(x=x, y=y, mode='markers', marker=dict(color='gray', size=1), showlegend=False), row=j, col=i)
            fig.add_trace(go.Histogram2dContour(x=x,y=y,colorscale='Blues',reversescale=False,showscale=False,ncontours=8, contours=dict(coloring='fill'),line=dict(width=1)),row=j,col=i)
            
            

        else:
            # here's where you put the histogram/kde
            #fig.add_trace(go.Histogram(x=x), row=j, col=i)
            kde = gaussian_kde(x)
            x_vals = np.linspace(min(x), max(x), 1000)
            y_vals = kde(x_vals)
            fig.add_trace(go.Scatter(x=x_vals, y=y_vals, mode='lines', line=dict(color='blue'),name=labels[i-1]), row=j, col=i)

        # add axes labels
        if (i == 1) and (i != j):
            fig.update_yaxes(title_text=labels[j - 1], row=j, col=i)
        if j == Nvar:
            fig.update_xaxes(title_text=labels[i - 1], row=j, col=i)
        # Add border to each subplot
        fig.update_xaxes(showline=True, linewidth=1, linecolor='black', mirror=True, row=j, col=i)
        fig.update_yaxes(showline=True, linewidth=1, linecolor='black', mirror=True, row=j, col=i)

# Set the plot background color to transparent
#fig.update_layout(plot_bgcolor='rgba(0,0,0,0)')
fig.show() 
'''

