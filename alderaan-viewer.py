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

        # Update x-axis label with units
        fig.update_traces(showlegend=True, row=1, col=1)
        fig.update_layout(xaxis_title=f"TIME (DAYS)", yaxis_title="FLUX")
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
        


if __name__ == '__main__':
    app.run(debug=True)
