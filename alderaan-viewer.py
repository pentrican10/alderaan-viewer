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

#sys.path.append('c:\\Users\\Paige\\Projects\\alderaan\\')

data_directory = 'c:\\Users\\Paige\\Projects\\data\\'


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
    table_data = data_load.read_table_data()
    left_content = render_template('left.html', table_data=table_data)
    right_top_content = render_template('right_top.html')
    right_bottom_content = render_template('right_bottom.html')
    return render_template('index.html',left_content=left_content, right_top_content=right_top_content, right_bottom_content=right_bottom_content)

@app.route('/star/<koi_id>')
def display_comment_file(koi_id):
    """
    Function to display comment file associated with KOI ID
    args: 
        koi_id: string, format K00000
    """
    path_extension = os.path.join('comment_files', f'{koi_id}_comments.txt')
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
    path_extension = os.path.join('comment_files', f'{koi_id}_comments.txt')
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
    path_extension = os.path.join('comment_files', f'{koi_id}_comments.txt')
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
    file_name_lc = star_id + '_lc_detrended.fits'
    file_path_lc = os.path.join(data_directory,'kepler_lightcurves_for_paige',file_name_lc)
    file_name_sc = star_id + '_sc_filtered.fits'
    file_path_sc = os.path.join(data_directory, file_name_sc)

    ### get data and create detrended light curve
    if os.path.isfile(file_path_lc) and os.path.isfile(file_path_sc):
        data_lc = data_load.read_data_from_fits(file_path_lc)
        data_sc = data_load.read_data_from_fits(file_path_sc)
        fig = px.scatter(data_lc, x="TIME", y="FLUX")#, 
                    #title="Kepler Detrended Light Curve")
        fig.add_scatter(x=data_sc['TIME'], y=data_sc['FLUX'],mode='markers')
        graph1JSON= json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder) 
        return jsonify(graph1JSON)
    
    if os.path.isfile(file_path_lc):
        data_lc = data_load.read_data_from_fits(file_path_lc)
        fig = px.scatter(data_lc, x="TIME", y="FLUX")
        graph1JSON= json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder) 
        return jsonify(graph1JSON)
    
    elif os.path.isfile(file_path_sc):
        data_sc = data_load.read_data_from_fits(file_path_sc)
        fig = px.scatter(data_sc, x="TIME", y="FLUX")
        graph1JSON= json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder) 
        return jsonify(graph1JSON)
    
    else:
        error_message = f'No data found for {koi_id}'
        return jsonify(error_message=error_message)

''' 
@app.route('/generate_plot_single_transit/<koi_id>/<int:line_number>')
def generate_plot_single_transit(koi_id, line_number):
    if (data_load.fetch_data(koi_id, line_number)):
        photometry_data, true_transit_number, center_time = data_load.fetch_data(koi_id, line_number)

        fig = px.scatter(photometry_data, x="TIME", y="FLUX") #only want to plot data within selected time window
        #fig.update_traces(marker=dict(
        #        color='red'))
        
        # plot the center time on the single transit block
        fig.add_trace(
            go.Scatter(x=[center_time, center_time], y=[min(fig.data[0].y), max(fig.data[0].y)],
               mode='lines',
               line=dict(color="red", width=2),
               name="Center time",
               showlegend=True)
        )

        graph2JSON= json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder) 
        response_data = {
            'graphJSON': graph2JSON,
            'transit_number': true_transit_number
        }
        return jsonify(response_data)
        #return jsonify(graph2JSON), jsonify(true_transit_number)
    else: 
        error2 = f'No data found for {koi_id}'
        return jsonify(error2=error2)

@app.route('/generate_plot_folded_light_curve/<koi_id>')
def generate_plot_folded_light_curve(koi_id):
    x = np.linspace(1,4,1000)
    y = np.linspace(3,8,1000)
    fig = px.scatter(x,y)
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder) 
    return jsonify(graphJSON)
'''

#@app.route('/generate_plot/<koi_id>')




if __name__ == '__main__':
    app.run(debug=True)
