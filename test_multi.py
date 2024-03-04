'''
import os
import sys
PROJECT_DIR = 'c:\\Users\\Paige\\Projects\\data\\alderaan_results'
sys.path.append(PROJECT_DIR)
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from astropy.io import fits
import plotly.express as px
from plotly.subplots import make_subplots

file = os.path.join(PROJECT_DIR, '2023-05-19_singles\\S00333\\S00333-results.fits')
def load_posteriors(f):
    with fits.open(f) as hduL:
        data = hduL['SAMPLES'].data
        keys = data.names
        _posteriors = []
        for k in keys:
            _posteriors.append(data[k])
        return pd.DataFrame(np.array(_posteriors).T, columns=keys)
data = load_posteriors(file)
#data
import corner
#_ = corner.corner(data, var_names=['C0_0', 'C1_0', 'ROR_0', 'IMPACT_0', 'DUR14_0'])
fig = make_subplots(rows=2,cols=2)
contour = px.density_contour(data, x="C0_0", y="C1_0")#, marginal_x="histogram", marginal_y="histogram")
hist1 = px.histogram(data, x="C0_0")
hist2 = px.histogram(data,x="C1_0")

for trace in contour.data:
    fig.add_trace(trace, row=2, col=1)

for trace in hist1.data:
    fig.add_trace(trace, row=1, col=1)

for trace in hist2.data:
    fig.add_trace(trace, row=2, col=2)

# fig.add_trace(contour, row=2,col=1)
# fig.add_trace(hist1, row=1, col=1)
# fig.add_trace(hist2, row=2,col=2)
fig.show()
#plt.show()

'''

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

def load_posteriors(f):
    with fits.open(f) as hduL:
        data = hduL['SAMPLES'].data
        keys = data.names
        _posteriors = []
        for k in keys:
            _posteriors.append(data[k])
        return pd.DataFrame(np.array(_posteriors).T, columns=keys)

data = load_posteriors(file)

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
#scatter = go.Scatter(x=x, y=y, mode='markers',marker=dict(size=1))
            #fig.add_trace(scatter, row=j, col=i)
            #fig.add_trace(go.Scatter(x=x, y=y, mode='markers',marker=dict(size=1)), row=j, col=i)

            # Add density contour
            #hist_data = np.histogram2d(x, y, bins=6, density=True)
            #fig.add_trace(go.Contour(z=hist_data[0], x=hist_data[1][:-1], y=hist_data[2][:-1], contours_coloring='lines', showscale=False), row=j, col=i)
            # Calculate density using Gaussian KDE
            kde = gaussian_kde(np.vstack([x, y]))
            xi, yi = np.mgrid[x.min():x.max():50j, y.min():y.max():50j]
            zi = kde(np.vstack([xi.flatten(), yi.flatten()]))

            # Transpose zi for correct orientation of contours
            zi = zi.reshape(xi.shape).T

            # Plot contours
            contour = go.Contour(z=zi, x=xi[:, 0], y=yi[0, :], contours_coloring='lines', line_width=1, ncontours=8)
            fig.add_trace(contour, row=j, col=i)
            #fig.add_trace(scatter, row=j, col=i)
            # Select scatter points outside the contour lines
            threshold = np.percentile(zi, 90)  # Adjust the percentile threshold as needed
            x_outside = []
            y_outside = []

            for m in range(len(xi)):
                for n in range(len(yi)):
                    if zi[m, n] < threshold:
                        x_outside.append(xi[m, n])
                        y_outside.append(yi[m, n])

            # Plot scatter points outside the contour lines
            fig.add_trace(go.Scatter(x=x_outside, y=y_outside, mode='markers', marker=dict(size=1)), row=j, col=i)
'''