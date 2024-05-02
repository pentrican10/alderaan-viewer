
import batman
import numpy as np
import matplotlib.pyplot as plt, mpld3

params = batman.TransitParams()
params.t0 = 0.                       #time of inferior conjunction
params.per = 1.                      #orbital period
params.rp = 0.1                      #planet radius (in units of stellar radii)
params.b = 0.8                       #impact
params.T14 = .5                      #transit duration
#params.a = 15.                       #semi-major axis (in units of stellar radii)
#params.inc = 87.                     #orbital inclination (in degrees)
#params.ecc = 0.                      #eccentricity
#params.w = 90.                       #longitude of periastron (in degrees)
params.u = [0.1, 0.3]                #limb darkening coefficients [u1, u2]
params.limb_dark = "quadratic"       #limb darkening model
t = np.linspace(-0.05, 0.05, 100)
m = batman.TransitModel(params, t)    #initializes model
flux = m.light_curve(params)          #calculates light curve
plt.plot(t, flux)
plt.xlabel("Time from central transit")
plt.ylabel("Relative flux")
#plt.show()
plt.show()
'''
from flask import Flask, render_template

import matplotlib.pyplot as plt, mpld3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from   scipy import stats
import seaborn as sns
from scipy.stats import gaussian_kde


app = Flask(__name__)

@app.route('/')
def index():
    
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

    Nvar = 2   # Number of variables

    data = [df['x'], df['y']]
    labels = ['x', 'y']
    fig, axs = plt.subplots(Nvar, Nvar, figsize=(10, 10))
    # Iterate over each subplot
    for i in range(Nvar):
        for j in range(Nvar):
            ax = axs[i, j]
            
            # Diagonal plots are histograms
            if i == j:
                #ax.hist(data[i], bins=10, color='skyblue')
                
                kde = gaussian_kde(data[i]) #, weights=weights
                x_vals = np.linspace(min(data[i])*0.95, max(data[i])*1.05, 1000)
                y_vals = kde(x_vals)
                ax.plot(x_vals,y_vals)
                #ax.set_xlabel(labels[i]) 
                
            # Lower triangle plots are scatter plots or line plots
            elif i > j:
                #ax.plot(data[j], data[i], 'o', color='orange')  # Scatter plot
                sns.kdeplot(df[::thin], x=f'{labels[j]}', y=f'{labels[i]}', fill=True, levels=4, ax=axs[i,j])
                ax.scatter(df.x[low_density], df.y[low_density], color='C0', s=3)
                ax.set_xlabel(labels[j])
                ax.set_ylabel(labels[i])
                #ax.set_xlim(-1,1)
                
            else:
                ax.remove()
            
            ### add labels to x and y axes
            if (i == 0) and (i != j):
                #fig.update_yaxes(title_text=labels[j], row=j + 1, col=i + 1)
                ax.set_ylabel(labels[i])
            if j == Nvar - 1:
                #fig.update_xaxes(title_text=labels[i], row=j + 1, col=i + 1)
                ax.set_xlabel(labels[j])
              

    # Adjust layout to prevent overlapping
    plt.tight_layout()

    # Convert the plot to an mpld3 object
    mpld3_plot = mpld3.fig_to_html(fig)

    # Render the HTML template with the mpld3 plot
    return render_template('plot.html', mpld3_plot=mpld3_plot)

if __name__ == '__main__':
    app.run(debug=True)
'''
##########################################################################################
'''
import matplotlib.pyplot as plt, mpld3
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

Nvar = 2   # Number of variables

data = [df['x'], df['y']]
labels = ['x', 'y']

fig, axs = plt.subplots(Nvar, Nvar, figsize=(10, 10))

for i in range(Nvar):
    for j in range(Nvar):
        

        # plot the data
        if i == j:
            # Add histogram/kde here
            axs[i, j].hist(data[i], bins=10, color='skyblue')
            axs[i, j].set_xlabel(labels[i])
        else:
            #axs[i, j].scatter(data[j], data[i], color='orange')
            sns.kdeplot(df[::thin], x=f'{labels[j]}', y=f'{labels[i]}', fill=True, levels=4, ax=[i,j])
            axs[i, j].scatter(df.x[low_density], df.y[low_density], color='C0', s=3)
            axs[i, j].set_xlabel(labels[j])
            axs[i, j].set_ylabel(labels[i])

plt.tight_layout()




#plt.figure()
#sns.kdeplot(df[::thin], x='x', y='y', fill=True, levels=4)
#plt.scatter(df.x[low_density], df.y[low_density], color='C0', s=3)
mpld3.show()
'''

"""
import plotly.graph_objects as go
import numpy as np

# Sample data (replace with your actual data)
np.random.seed(0)
x = np.random.normal(0, 1, 1000)
y = np.random.normal(0, 1, 1000)

# Calculate the density of points
hist, xedges, yedges = np.histogram2d(x, y, bins=50, density=True)

# Define contour levels
contour_levels = [10, 25, 50, 75, 90]  # Adjust as needed

# Create the contour plot
fig = go.Figure()

# Add histogram2dcontour trace
fig.add_trace(go.Histogram2dContour(
    x=x,
    y=y,
    z=hist.T,  # Transpose the histogram to match Plotly's expectations
    colorscale='Blues',
    reversescale=True,
    xaxis='x',
    yaxis='y',
    contours=dict(
        start=min(contour_levels),
        end=max(contour_levels),
        size=(max(contour_levels) - min(contour_levels)) / len(contour_levels),
    ),
))

# Update layout
fig.update_layout(
    xaxis=dict(title='X-axis'),
    yaxis=dict(title='Y-axis'),
    title='Histogram 2D Contour Plot with Manual Contours',
)

# Show the plot
fig.show()
"""
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
#print(data['LN_LIKE'])
# print(data['LN_LIKE'].max())
# max_index = data['LN_LIKE'].idxmax()
# print(max_index)
# print(data['IMPACT_0'][max_index])
print(data['DUR14_0'])
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

"""
import numpy as np
import plotly.graph_objects as go
from sklearn.neighbors import KernelDensity
from scipy.spatial import ConvexHull

# Generate random data centered around 0
np.random.seed(42)  # for reproducibility
num_points = 1000
x = np.random.randn(num_points)
y = np.random.randn(num_points)

# Fit kernel density estimation
data = np.column_stack((x, y))
kde = KernelDensity(bandwidth=0.5, kernel='gaussian')
kde.fit(data)

# Evaluate KDE on a grid
x_grid, y_grid = np.meshgrid(np.linspace(x.min(), x.max(), 100),
                             np.linspace(y.min(), y.max(), 100))
xy_grid = np.column_stack((x_grid.ravel(), y_grid.ravel()))
density = np.exp(kde.score_samples(xy_grid))
density_grid = density.reshape(x_grid.shape)

# Determine threshold densities corresponding to the percentiles
percentiles = [25, 50, 75, 90]  # Reversed order
threshold_densities = np.percentile(density, percentiles)

# Create a plotly figure
fig = go.Figure()

# Add scatter plot for all points
fig.add_trace(go.Scatter(
    x=x,
    y=y,
    mode='markers',
    marker=dict(
        color='blue',
        opacity=0.3
    ),
    name='All Points'
))

# Define shades of blue for each percentile
blue_shades = ['rgba(0, 0, 100, 1)', 'rgba(0, 0, 150, 1)', 'rgba(0, 0, 200, 1)', 'rgba(0, 0, 255, 1)']  # Adjusted order

# Iterate over each percentile in reversed order
for percentile, threshold_density, shade in zip(percentiles, threshold_densities, blue_shades):
    # Select points within the percentile density
    selected_points = (density >= threshold_density)

    # Find the convex hull of the selected points
    selected_x = xy_grid[selected_points][:, 0]
    selected_y = xy_grid[selected_points][:, 1]
    hull = ConvexHull(np.column_stack((selected_x, selected_y)))

    # Extract the vertices of the convex hull
    hull_vertices_x = selected_x[hull.vertices]
    hull_vertices_y = selected_y[hull.vertices]

    # Add shaded region for the convex hull
    fig.add_trace(go.Scatter(
        x=hull_vertices_x,
        y=hull_vertices_y,
        fill='toself',
        fillcolor=shade,  # Varying shades of blue
        line=dict(color=shade),  # Match fill color
        mode='lines',
        name=f'{percentile}th Percentile Density Region'
    ))

# Update layout
fig.update_layout(
    title='Shaded Regions Enclosing Percentiles of Points by Density',
    xaxis_title='X',
    yaxis_title='Y',
    showlegend=True,
    legend=dict(x=0.02, y=0.98),
    hovermode='closest'
)

# Show plot
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

