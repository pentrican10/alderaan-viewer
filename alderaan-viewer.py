from flask import Flask, render_template, jsonify, request, session, redirect, url_for
import os
import csv
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from astropy.io import fits
import pandas as pd
import json
import plotly.utils
import numpy as np
import sys
import lightkurve as lk
import numpy.polynomial.polynomial as poly
import data_load
import glob
import plotly.graph_objects as go
import re
from scipy.stats import gaussian_kde
from scipy import stats
from sklearn.neighbors import KernelDensity
from sklearn.decomposition import PCA

from plotly.subplots import make_subplots


#sys.path.append('c:\\Users\\Paige\\Projects\\alderaan\\')
data_directory = 'c:\\Users\\Paige\\Projects\\data\\alderaan_results'


app = Flask(__name__)
app.secret_key = 'super_secret'

@app.route('/')
def index():
    session.clear()
    return redirect(url_for('login'))


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        if username:
            session['username'] = username
            return redirect(url_for('display_table_data'))
    return render_template('login.html')

@app.route('/logout',methods=['POST'])
def logout():
    session.pop('username',None) #removes username
    return redirect(url_for('login'))

@app.route('/home')
def display_table_data():
    """
    Assigns each html file to their respective locations
    Renders Index Template
    """
    table = request.args.get('table', '2023-05-19_singles.csv')
    update_data_directory(table)
    table_data = data_load.read_table_data(table)
    left_content = render_template('left.html', table_data=table_data)
    right_top_content = render_template('right_top.html')
    right_bottom_content = render_template('right_bottom.html')
    return render_template('index.html',left_content=left_content, right_top_content=right_top_content, right_bottom_content=right_bottom_content)

def update_data_directory(selected_table):
    global data_directory
    data_directory = os.path.join('c:\\Users\\Paige\\Projects\\data\\alderaan_results', selected_table[:-4])

@app.route('/star/<koi_id>')
def display_comment_file(koi_id):
    """
    Function to display comment file associated with KOI ID
    args: 
        koi_id: string, format K00000
    """
    star_id = koi_id.replace("K","S")
    path_extension = os.path.join(star_id, f'{star_id}_comments.txt')
    file_path = os.path.join(data_directory, path_extension)
    if os.path.isfile(file_path):
        with open(file_path, 'r') as file:
                file_content = file.read()
    else:
        file_content = f'Comment file for {koi_id} not found.'
    return file_content

@app.route('/star/<koi_id>/save_comment', methods=['POST'])
def save_comment(koi_id):
    """
    function saves the comment input by user to the associated comment file
    Saves with username, date, comment
    Returns the function to display the updated comment file
    """
    star_id = koi_id.replace("K","S")
    path_extension = os.path.join(star_id, f'{star_id}_comments.txt')
    file_path = os.path.join(data_directory, path_extension)
    comment = request.form.get('comment').strip()
    username = session.get('username')
    with open(file_path, 'a') as file:
        file.write("\n")
        file.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        file.write(f"User: {username}\n")
        file.write(f"Comment: {comment}\n")
    return display_comment_file(koi_id)

@app.route('/star/<koi_id>/edit_file', methods=['POST'])
def save_file(koi_id):
    """
    saves the file and displays the updated file or an error message
    """
    star_id = koi_id.replace("K","S")
    path_extension = os.path.join(star_id, f'{star_id}_comments.txt')
    file_path = os.path.join(data_directory, path_extension)
    content = request.form.get('content')
    # Normalize line endings to Unix-style (\n)
    content = content.replace('\r\n', '\n')
    try:
        with open(file_path, 'w') as file:
            file.writelines(content)
            file.write('\n')
        return display_comment_file(koi_id)
    except Exception as e:
        return f"An error occurred: {str(e)}"

@app.route('/generate_plot/<koi_id>')
def generate_plot_Detrended_Light_Curve(koi_id):
    star_id = koi_id.replace("K","S")
    file_name_lc = star_id + '_lc_filtered.fits'
    file_path_lc = os.path.join(data_directory,star_id,file_name_lc)
    file_name_sc = star_id + '_sc_filtered.fits'
    file_path_sc = os.path.join(data_directory, star_id, file_name_sc)

    ### initialize figure
    fig = make_subplots(rows=1, cols=1)

    file_name = star_id + '_*_quick.ttvs'
    file_paths = glob.glob(os.path.join(data_directory,star_id, file_name))
    

    ### get data and create detrended light curve
    if os.path.isfile(file_path_lc) and os.path.isfile(file_path_sc):
        data_lc = data_load.read_data_from_fits(file_path_lc)
        data_sc = data_load.read_data_from_fits(file_path_sc)

        lc = px.scatter(data_lc, x="TIME",y="FLUX").data[0]
        lc.marker.update(symbol="circle", size=4, color="blue")
        lc.name = "Long Cadence"
        fig.add_trace(lc, row=1, col=1)

        ### trim short cadence data
        sc_time_trim = data_sc['TIME'][::30]
        sc_flux_trim = data_sc['FLUX'][::30]
        sc = px.scatter(x=sc_time_trim, y=sc_flux_trim).data[0]
        sc.marker.update(symbol="circle", size=4, color="gray")
        sc.name = "Short Cadence"
        fig.add_trace(sc, row=1, col=1)

        ###################
        colors = ['orange','green','blue','pink','red','purple']

        ttv_lines = []  # Store all TTV lines for manual legend item creation

        # Iterate through file paths
        for i, file_path in enumerate(file_paths):
            if os.path.isfile(file_path):
                index, center_time, model, out_prob, out_flag = data_load.get_ttv_file(koi_id, file_path)

                # Add a dot for each center time
                offset = 0.0001*i
                y_pts = 0.998* np.ones(len(center_time)) + offset
                c_time = px.scatter(x=center_time, y=y_pts).data[0]
                color = colors[i]
                c_time.marker.update(symbol="circle", size=4, color=color)
                c_time.name = f"ttime 0{i}"
                fig.add_trace(c_time, row=1, col=1)
                

        # Update x-axis label with units
        fig.update_traces(showlegend=True, row=1, col=1)
        fig.update_layout(xaxis_title=f"TIME (DAYS)", yaxis_title="FLUX")
        fig.update_layout(legend=dict(traceorder="normal"))
        graph1JSON= json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder) 
        return jsonify(graph1JSON)
    
    elif os.path.isfile(file_path_lc):
        data_lc = data_load.read_data_from_fits(file_path_lc)
        lc = px.scatter(data_lc, x="TIME",y="FLUX").data[0]
        lc.marker.update(symbol="circle", size=4, color="blue")
        lc.name = "Long Cadence"
        fig.add_trace(lc, row=1, col=1)
        # Update x-axis label with units
        fig.update_layout(xaxis_title=f"TIME (DAYS)", yaxis_title="FLUX")
        graph1JSON= json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder) 
        return jsonify(graph1JSON)
    
    elif os.path.isfile(file_path_sc):
        data_sc = data_load.read_data_from_fits(file_path_sc)
        sc_time_trim = data_sc['TIME'][::30]
        sc_flux_trim = data_sc['FLUX'][::30]
        sc = px.scatter(x=sc_time_trim, y=sc_flux_trim).data[0]
        sc.marker.update(symbol="circle", size=4, color="gray")
        sc.name = "Short Cadence"
        fig.add_trace(sc, row=1, col=1)
        # Update x-axis label with units
        fig.update_layout(xaxis_title=f"TIME (DAYS)", yaxis_title="FLUX")
        graph1JSON= json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder) 
        return jsonify(graph1JSON)
    
    else:
        error_message = f'No data found for {koi_id}'
        return jsonify(error_message=error_message)


@app.route('/generate_plot_single_transit/<koi_id>/<int:line_number>/<planet>')
def generate_plot_single_transit(koi_id, line_number,planet):
    star_id = koi_id.replace("K","S")
    ttv_file = star_id + planet
    ext = os.path.basename(data_directory) +'.csv'
    csv_file_path = os.path.join(data_directory, ext)

    period = data_load.get_periods_for_koi_id(csv_file_path, koi_id)

    planet_num = re.findall(r'\d+', planet)
    num = planet_num[0][1]
    int_num = int(num)
    title = period[int_num]
    
    ### initialize figure
    fig = make_subplots(rows=1, cols=1)

    if (data_load.single_transit_data(koi_id, line_number,ttv_file)):
        photometry_data_lc,photometry_data_sc, transit_number, center_time = data_load.single_data(koi_id, line_number,ttv_file)
        transit_lc = px.scatter(photometry_data_lc, x="TIME", y="FLUX").data[0]

        fig.add_trace(transit_lc, row=1, col=1)

        transit_sc = px.scatter(photometry_data_sc, x="TIME", y="FLUX").data[0]
        fig.add_trace(transit_sc,row=1,col=1)

        lc_min,lc_max,sc_min,sc_max = data_load.get_min_max(koi_id)

        if len(photometry_data_sc)>0:
            fig.update_layout(xaxis_title=f"TIME (DAYS)", 
                          yaxis_title="FLUX",
                          yaxis=dict(range=[sc_min, sc_max])
            )
        else:
            fig.update_layout(xaxis_title=f"TIME (DAYS)", 
                          yaxis_title="FLUX",
                          yaxis=dict(range=[lc_min, lc_max])
            )
       
        fig.update_layout(title=title, title_x=0.5)
        
        graphJSON= json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder) 
        response_data = {
            'graphJSON': graphJSON,
            'transit_number': transit_number
        }
        return jsonify(response_data)
    else:
        error_message = f'No data found for {koi_id}'
        return jsonify(error_message=error_message)
    

    
@app.route('/get_transit_file_options/<koi_id>')
def planet_options(koi_id):
    star_id = koi_id.replace("K","S")
    file_name = star_id + '_*_quick.ttvs'
    file_paths = glob.glob(os.path.join(data_directory,star_id, file_name))
    options = []
    for i in range(len(file_paths)):
        option_value = f"_{i:02d}_quick.ttvs"
        option = {'number': f'{i:02d}', 'value': option_value}
        options.append(option)
    return jsonify(options)

@app.route('/get_transit_file_options_corner/<koi_id>')
def planet_options_corner(koi_id):
    star_id = koi_id.replace("K","S")
    file_name = star_id + '_*_quick.ttvs'
    file_paths = glob.glob(os.path.join(data_directory,star_id, file_name))
    options = []
    for i in range(len(file_paths)):
        option_value =  f'{i}'
        option = {'number': f'{i:02d}', 'value': option_value}
        options.append(option)
    return jsonify(options)

@app.route('/generate_plot_folded_light_curve/<koi_id>')
def generate_plot_folded_light_curve(koi_id):
    star_id = koi_id.replace("K","S")
    file_name = star_id + '_*_quick.ttvs'
    file_paths = glob.glob(os.path.join(data_directory,star_id, file_name))
    ext = os.path.basename(data_directory) +'.csv'
    csv_file_path = os.path.join(data_directory, ext)

    ### number of planets from number of ttv files
    npl = len(file_paths)
    subplot_height=350
    titles = data_load.get_periods_for_koi_id(csv_file_path, koi_id)
    fig = make_subplots(rows=npl, cols=1,
                        subplot_titles = titles,
                        row_heights=[subplot_height]*npl,
                        vertical_spacing=0.15)
    
    for i, file_path in enumerate(file_paths):
        fold_data_lc, fold_data_sc, binned_avg = data_load.folded_data(koi_id,file_path)

        if fold_data_lc is not None and fold_data_sc is not None:
            ### short cadence
            fold_sc = go.Scatter(x=fold_data_sc.TIME, y=fold_data_sc.FLUX, mode='markers')
            fold_sc.marker.update(symbol="circle", size=4, color="gray")
            fold_sc.name = "Short Cadence"
            fold_sc.legendgroup=f'{i}'
            fig.add_trace(fold_sc, row=i+1, col=1)
            ### long cadence
            fold_lc = go.Scatter(x=fold_data_lc.TIME, y=fold_data_lc.FLUX, mode='markers')
            fold_lc.marker.update(symbol="circle", size=5, color="blue")
            fold_lc.name = "Long Cadence"
            fold_lc.legendgroup=f'{i}'
            fig.add_trace(fold_lc, row=i+1, col=1)
            ### binned avg
            bin_avg = go.Scatter(x=binned_avg.TIME, y=binned_avg.FLUX, mode='markers')
            bin_avg.marker.update(symbol="square", size=10, color="orange")
            bin_avg.name = "Binned Average"
            bin_avg.legendgroup=f'{i}'
            fig.add_trace(bin_avg, row=i+1, col=1)
            ### Update x-axis and y-axis labels for each subplot
            fig.update_xaxes(title_text="TIME (HOURS)", row=i+1, col=1)
            fig.update_yaxes(title_text="FLUX", row=i+1, col=1)
        
        elif fold_data_lc is not None and fold_data_sc is None:
            fold_lc = go.Scatter(x=fold_data_lc.TIME, y=fold_data_lc.FLUX, mode='markers')
            fold_lc.marker.update(symbol="circle", size=5, color="blue")
            fold_lc.name = "Long Cadence"
            fig.add_trace(fold_lc, row=i+1, col=1)
            ### binned avg
            bin_avg = go.Scatter(x=binned_avg.TIME, y=binned_avg.FLUX, mode='markers')
            bin_avg.marker.update(symbol="square", size=10, color="orange")
            bin_avg.name = "Binned Average"
            fig.add_trace(bin_avg, row=i+1, col=1)
            ### Update x-axis and y-axis labels for each subplot
            fig.update_xaxes(title_text="TIME (HOURS)", row=i+1, col=1)
            fig.update_yaxes(title_text="FLUX", row=i+1, col=1)

        elif fold_data_sc is not None and fold_data_lc is None:
            fold_sc = go.Scatter(x=fold_data_sc.TIME, y=fold_data_sc.FLUX, mode='markers')
            fold_sc.marker.update(symbol="circle", size=4, color="gray")
            fold_sc.name = "Short Cadence"
            fig.add_trace(fold_sc, row=i+1, col=1)
            ### binned avg
            bin_avg = go.Scatter(x=binned_avg.TIME, y=binned_avg.FLUX, mode='markers')
            bin_avg.marker.update(symbol="square", size=10, color="orange")
            bin_avg.name = "Binned Average"
            fig.add_trace(bin_avg, row=i+1, col=1)
            ### Update x-axis and y-axis labels for each subplot
            fig.update_xaxes(title_text="TIME (HOURS)", row=i+1, col=1)
            fig.update_yaxes(title_text="FLUX", row=i+1, col=1)
        
        else:
            error_message = f'No data found for {koi_id}'
            return jsonify(error_message=error_message)
        
    ### return whole fig to page
    if npl>1:
        fig.update_layout(height=npl * subplot_height,legend_tracegroupgap = 240)
    fig.update_traces(showlegend=True, row=1, col=1)
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder) 
    return jsonify(graphJSON)
    

@app.route('/generate_plot_OMC/<koi_id>')
def generate_plot_OMC(koi_id):
    star_id = koi_id.replace("K","S")
    file_name = star_id + '_*_quick.ttvs'
    file_paths = glob.glob(os.path.join(data_directory,star_id, file_name))
    ext = os.path.basename(data_directory) +'.csv'
    csv_file_path = os.path.join(data_directory, ext)
    ### number of planets from number of ttv files
    npl = len(file_paths)
    titles = data_load.get_periods_for_koi_id(csv_file_path, koi_id)
    fig = make_subplots(rows=npl, cols=1,
                        subplot_titles=titles)

    for i, file_path in enumerate(file_paths):
        omc_data, omc_model, out_prob, out_flag = data_load.OMC_data(koi_id, file_path)
        show_outliers = True

        if omc_data is not None:
            mask = [bool(flag) for flag in out_flag]
            if show_outliers:
                omc = px.scatter(omc_data,  
                                x='TIME', 
                                y='OMC', 
                                color=out_prob, 
                                color_continuous_scale='viridis').data[0]
                line_trace = px.line(omc_model,x='TIME', y='OMC_MODEL').data[0]
                line_trace.line.color = 'red'
                fig.add_trace(omc, row=(i+1), col=1)
                fig.add_trace(line_trace, row=(i+1), col=1)

                # Add a new scatter trace for outliers with 'x' shape markers
                scatter_outliers = px.scatter(omc_data[mask], x='TIME', y='OMC').update_traces(
                    marker=dict(symbol='x', color='orange'),
                    line=dict(width=0.7))

                fig.add_trace(scatter_outliers.data[0], row=(i+1), col=1)
                
                # Update x-axis and y-axis labels for each subplot
                fig.update_xaxes(title_text="TIME (DAYS)", row=i+1, col=1)
                fig.update_yaxes(title_text="O-C (MINUTES)", row=i+1, col=1)
                fig.update_coloraxes(colorbar_title_text='Out Probability', colorbar_len=0.2, row=i+1, col=1)

            else:
                mask_arr = np.array(mask)
                omc = px.scatter(omc_data[~mask_arr], x="TIME",y="OMC")
                # Add a line plot for OMC_MODEL
                line_trace = px.line(omc_model[~mask_arr], x="TIME", y="OMC_MODEL").data[0]
                line_trace.line.color = 'red' 
                fig.add_trace(omc, row=(i+1), col=1)
                fig.add_trace(line_trace, row=(i+1), col=1)
                ### update axes and colorbar
                fig.update_xaxes(title_text="TIME (DAYS)", row=i+1, col=1)
                fig.update_yaxes(title_text="O-C (MINUTES)", row=i+1, col=1)
                fig.update_coloraxes(colorbar_title_text='Out Probability', colorbar_len=0.2, row=i+1, col=1)
        
        else: 
            error_message = f'No data found for {koi_id}'
            return jsonify(error_message=error_message)
    ### return whole figure to page
    fig.update_coloraxes(colorbar_title_text='Out Probability')#, colorbar_len=0.2)
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder) 
    return jsonify(graphJSON)
        
@app.route('/generate_plot_corner/<koi_id>/<selected_columns>/<planet_num>')
def generate_plot_corner(koi_id,selected_columns, planet_num):
    selected_columns = selected_columns.split(',')
    star_id = koi_id.replace("K","S")
    file =star_id + '-results.fits'
    file_path = os.path.join(data_directory, star_id, file)

    if os.path.isfile(file_path):
        data = data_load.load_posteriors(file_path,planet_num,koi_id)
        LN_WT = data['LN_WT'][::5].values
        weight = np.exp(LN_WT- LN_WT.max())
        w = weight/ np.sum(weight)
        index = np.arange(len(LN_WT))
        rand_index = np.random.choice(index,p=w,size=len(LN_WT))

        data = data[selected_columns]

        labels = data.columns.tolist()

        fig = make_subplots(rows=len(selected_columns), cols=len(selected_columns))
        tick_values_y_c0 = None
        tick_values_x_c0 = None
        for i in range(len(selected_columns)):
            for j in range(i, len(selected_columns)):
                # x = data[selected_columns[i]]
                # y = data[selected_columns[j]]
                
                x1 = data[selected_columns[i]][::5].values
                y1 = data[selected_columns[j]][::5].values
                ### trim with random indicies 
                x = x1[rand_index]
                y = y1[rand_index]
                

                if i != j:
                    # Calculate the density of the points
                    xy = np.vstack([x, y])
                    z = gaussian_kde(xy)(xy)
                    # Select the top 90th percentile based on density
                    threshold = np.percentile(z, 10)
                    # Plot points below the threshold as scatter plot
                    fig.add_trace(go.Scatter(x=x[z < threshold], y=y[z < threshold], mode='markers', marker=dict(color='gray', size=1), showlegend=False), row=j + 1, col=i + 1)
                    # Plot points above the threshold in the contour
                    fig.add_trace(go.Histogram2dContour(x=x[z >= threshold], y=y[z >= threshold], colorscale='Blues', reversescale=False, showscale=False, ncontours=8, contours=dict(coloring='fill'), line=dict(width=1)), row=j + 1, col=i + 1)
                    
                    
                    #fig.add_trace(go.Scatter(x=x, y=y, mode='markers', marker=dict(color='gray', size=1), showlegend=False), row=j + 1, col=i + 1)
                    #fig.add_trace(go.Scatter(x=x, y=y, mode='markers', marker=dict(color='gray', size=1), showlegend=False), row=j + 1, col=i + 1)
                    #fig.add_trace(go.Histogram2dContour(x=x, y=y, colorscale='Blues', reversescale=False, showscale=False, ncontours=8, contours=dict(coloring='fill'), line=dict(width=1)), row=j + 1, col=i + 1)
                else:
                    kde = gaussian_kde(x, weights=weight) #, weights=weights
                    x_vals = np.linspace(min(x)*0.95, max(x)*1.05, 1000)
                    y_vals = kde(x_vals)
                    fig.add_trace(go.Scatter(x=x_vals, y=y_vals, mode='lines', line=dict(color='blue'), name=labels[i], showlegend=False), row=j + 1, col=i + 1)

                ### add labels to x and y axes
                if (i == 0) and (i != j):
                    fig.update_yaxes(title_text=labels[j], row=j + 1, col=i + 1)
                if j == len(selected_columns) - 1:
                    fig.update_xaxes(title_text=labels[i], row=j + 1, col=i + 1)

                ### only have 3 ticks and only show axis at left-most column and bottom-most row
                if i != j:
                    #tick_values_x = np.linspace(min(x), max(x), 3)
                    #tick_values_y = np.linspace(min(y), max(y), 3)
                    tick_format = '.2f'
                    #tick_text_x = [f"{val:{tick_format}}" for val in tick_values_x]
                    #tick_text_y = [f"{val:{tick_format}}" for val in tick_values_y]
                    ### update which axes show
                    plot_range=None
                    if (i==0):
                        if labels[j]==f'IMPACT_{planet_num}':
                            tick_values_y = np.linspace(0, 1.2, 3)
                            plot_range = [0,1.2]
                        elif labels[j]==f'LD_Q1':
                            tick_values_y = np.linspace(0, 1, 3)
                            plot_range = [0,1]
                        elif labels[j]==f'LD_Q2':
                            tick_values_y = np.linspace(0, 1, 3)
                            plot_range = [0,1]
                        elif labels[j]==f'LD_U1':
                            tick_values_y = np.linspace(0, 2, 3)
                            plot_range = [0,2]
                        elif labels[j]==f'LD_U2':
                            tick_values_y = np.linspace(-1, 1, 3)
                            plot_range = [-1,1]
                        elif labels[j]==f'C0_{planet_num}' or labels[j]==f'C1_{planet_num}':
                            #tick_values_y_c0 = np.linspace(min(y), max(y), 3)
                            if tick_values_y_c0 is None:
                                tick_values_y_c0 = np.linspace(min(y), max(y), 3)
                                tick_values_y = tick_values_y_c0  # Use the same tick values for C0 and C1
                            tick_values_y = tick_values_y_c0
                        else:
                            tick_values_y = np.linspace(min(y), max(y), 3)
                            plot_range = [min(y),max(y)]
                        tick_text_y = [f"{val:{tick_format}}" for val in tick_values_y]
                        fig.update_yaxes(tickvals=tick_values_y, ticktext=tick_text_y, row=j + 1, col=i + 1, tickangle=0)
                    else:
                        if labels[i]==f'IMPACT_{planet_num}':
                            tick_values_x = np.linspace(0, 1.2, 3)
                            plot_range = [0,1.2]
                        elif labels[i]==f'LD_Q1':
                            tick_values_x = np.linspace(0, 1, 3)
                            plot_range = [0,1]
                        elif labels[i]==f'LD_Q2':
                            tick_values_x = np.linspace(0, 1, 3)
                            plot_range = [0,1]
                        elif labels[i]==f'LD_U1':
                            tick_values_x = np.linspace(0, 2, 3)
                            plot_range = [0,2]
                        elif labels[i]==f'LD_U2':
                            tick_values_x = np.linspace(-1, 1, 3)
                            plot_range = [-1,1]
                        elif labels[i]==f'C0_{planet_num}' or labels[j]==f'C1_{planet_num}':
                            #tick_values_y_c0 = np.linspace(min(y), max(y), 3)
                            if tick_values_x_c0 is None:
                                tick_values_x_c0 = np.linspace(min(x), max(x), 3)
                                tick_values_x = tick_values_x_c0  # Use the same tick values for C0 and C1
                            tick_values_x = tick_values_x_c0
                        else:
                            tick_values_x = np.linspace(min(x), max(x), 3)
                            plot_range = [min(x),max(x)]
                        tick_text_x = [f"{val:{tick_format}}" for val in tick_values_x]
                        fig.update_yaxes(showticklabels=False, tickvals=tick_values_y, ticktext=tick_text_y, row=j + 1, col=i + 1, tickangle=0)
                    if j == len(selected_columns) - 1:
                        fig.update_xaxes(tickvals=tick_values_x, ticktext=tick_text_x, row=j + 1, col=i + 1, tickangle=0)
                    else:
                        fig.update_xaxes(showticklabels=False, tickvals=tick_values_x, ticktext=tick_text_x, row=j + 1, col=i + 1, tickangle=0)
                else:
                    ### histograms only have x axes
                    tick_values_x = np.linspace(min(x_vals), max(x_vals), 3)
                    tick_values_y = np.linspace(min(y_vals), max(y_vals), 3)
                    tick_format = '.2f'
                    tick_text_x = [f"{val:{tick_format}}" for val in tick_values_x]
                    tick_text_y = [f"{val:{tick_format}}" for val in tick_values_y]
                    # showticklabels=False,
                    fig.update_xaxes(tickvals=tick_values_x, ticktext=tick_text_x, row=j + 1, col=i + 1, tickangle=0)
                    fig.update_yaxes(showticklabels=False,tickvals=tick_values_y, row=j + 1, col=i + 1, tickangle=0)
                # if i!=j:
                #     fig.update_layout(plot_bgcolor='white')
                fig.update_xaxes(showline=True, linewidth=1, linecolor='black', mirror=True, row=j + 1, col=i + 1, tickangle=0)
                fig.update_yaxes(showline=True, linewidth=1, linecolor='black', mirror=True, row=j + 1, col=i + 1, tickangle=0)
        
        
        fig.update_layout(height=800, width=900)
        graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder) 
        return jsonify(graphJSON)
    else:
        error_message = f'No data found for {koi_id}'
        return jsonify(error_message=error_message)
    
    
    
# def generate_plot_corner(koi_id):
#     star_id = koi_id.replace("K","S")
#     file =star_id + '-results.fits'
#     file_path = os.path.join(data_directory, star_id, file)
#     if os.path.isfile(file_path):
#         data = data_load.load_posteriors(file_path)
#         Nvar = 5  # Set the number of variables to 5

#         # Slice the DataFrame to include only the first 5 columns
#         data = data.iloc[:, :Nvar]
#         # Subsample the data to every 100th data point
#         #data = data.iloc[::30, :]
#         labels = data.columns.tolist()

#         fig = make_subplots(rows=Nvar, cols=Nvar, horizontal_spacing=0.04, vertical_spacing=0.05)

#         for i in range(1, Nvar + 1):
#             for j in range(i, Nvar + 1):
#                 #x = data.iloc[:, i - 1]
#                 #y = data.iloc[:, j - 1]
#                 x = data.iloc[::30, i-1]
#                 y=data.iloc[::30, j-1]

#                 # plot the data
#                 if i != j:
#                     # x = data.iloc[::30, i-1]
#                     # y=data.iloc[::30, j-1]
#                     fig.add_trace(go.Scatter(x=x, y=y, mode='markers', marker=dict(color='gray', size=1), showlegend=False), row=j, col=i)
#                     fig.add_trace(go.Histogram2dContour(x=x,y=y,colorscale='Blues',reversescale=False,showscale=False,ncontours=8, contours=dict(coloring='fill'),line=dict(width=1)),row=j,col=i)
                    
#                 else:
#                     # here's where you put the histogram/kde
#                     #fig.add_trace(go.Histogram(x=x), row=j, col=i)
#                     kde = gaussian_kde(x)
#                     x_vals = np.linspace(min(x), max(x), 1000)
#                     y_vals = kde(x_vals)
#                     #fig.add_trace(go.Scatter(x=x_vals, y=y_vals, mode='lines', line=dict(color='blue'),name=labels[i-1]), row=j, col=i)
#                     fig.add_trace(go.Scatter(x=x_vals, y=y_vals, mode='lines', line=dict(color='blue'), showlegend=False), row=j, col=i)

#                 # add axes labels
#                 if (i == 1) and (i != j):
#                     fig.update_yaxes(title_text=labels[j - 1], row=j, col=i)
#                 if j == Nvar:
#                     fig.update_xaxes(title_text=labels[i - 1], row=j, col=i)
#                 # Add border to each subplot
#                 fig.update_xaxes(showline=True, linewidth=1, linecolor='black', mirror=True, row=j, col=i, tickangle=0)
#                 fig.update_yaxes(showline=True, linewidth=1, linecolor='black', mirror=True, row=j, col=i)
#         fig.update_layout(height=800, width=900)
#         graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder) 
#         return jsonify(graphJSON)
#     else:
#         error_message = f'No data found for {koi_id}'
#         return jsonify(error_message=error_message)

if __name__ == '__main__':
    app.run(debug=True)
