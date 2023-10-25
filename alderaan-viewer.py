# add short cadence 
# get results and clone whole repository and get to work in jupyter
# import results class from results.py
# launch jupyter
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
sys.path.append('c:\\Users\\Paige\\Projects\\alderaan\\')
import myResults

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
    table_data = read_table_data()
    left_content = render_template('left.html', table_data=table_data)
    right_top_content = render_template('right_top.html')
    right_bottom_content = render_template('right_bottom.html')
    return render_template('index1.html',left_content=left_content, right_top_content=right_top_content, right_bottom_content=right_bottom_content)


def read_table_data():
    file_path = os.path.join(os.path.dirname(__file__), 'data', '2023-05-19_singles.csv')
    table_data = []
    with open(file_path, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            #round table values
            row['kep_mag'] = round(float(row['kep_mag']), 2)
            row['Rstar'] = round(float(row['Rstar']), 2)
            row['logrho'] = round(float(row['logrho']), 2)
            row['Teff'] = round(float(row['Teff']))
            row['logg'] = round(float(row['logg']), 2)
            table_data.append(row)
    return table_data

#environment variable -- mini app
@app.route('/star/<koi_id>')
def display_comment_file(koi_id):
    file_path = os.path.join('C:\\Users\\Paige\\Projects','miniflask','comment_files',f'{koi_id}_comments.txt')
    if os.path.isfile(file_path):
        with open(file_path, 'r') as file:
                file_content = file.read()
    else:
        file_content = f'Comment file for {koi_id} not found.'
    return file_content


# DATA 
def read_data_from_fits(file_path):
    with fits.open(file_path) as fits_file:
        time = np.array(fits_file[1].data, dtype=float)
        flux = np.array(fits_file[2].data, dtype=float)
        df = pd.DataFrame(dict(
            TIME=time,
            FLUX=flux
        ))
    return df

def fetch_data(koi_id, line_number):
    star_id = koi_id.replace("K","S")
    file_name = star_id + '_lc_detrended.fits'
    file_path = os.path.join('C:\\Users\\Paige\\Projects','miniflask','kepler_lightcurves_for_paige',file_name)
    #get data and create detrended light curve
    if os.path.isfile(file_path):
        photometry_data = read_data_from_fits(file_path) #descriptive names
        transit_values, center_time_values = read_center_time_values_from_file(koi_id)
        if line_number < len(transit_values):
            center_time = center_time_values[line_number]
            transit_number = transit_values[line_number]
        
            start_time = float(center_time) - 0.25
            end_time= float(center_time) + 0.25

            use = (photometry_data['TIME'] > start_time) & (photometry_data['TIME'] < end_time)
            filtered_data = photometry_data[use]
            return filtered_data, transit_number, center_time ############
    else:
        return None, None, None
    

def read_center_time_values_from_file(koi_id): #read_ct_and_transit_number_from_file(koi_id)
    star_id = koi_id.replace("K","S")
    file_name = star_id + '_00_quick.ttvs'
    file_path = os.path.join('C:\\Users\\Paige\\Projects','miniflask','quick_ttvs_for_paige',file_name)

    center_time_values=[] 
    transit_value =[]
    # Open the file for reading
    with open(file_path, 'r') as file:
        # Iterate through each line in the file
        for line in file:
            # Split the line into columns based on the delimiter
            columns = line.strip().split('\t')

            center_time_values.append(columns[1])
            transit_value.append(columns[0])
        
    #print(center_time_values)
    return transit_value, center_time_values
    

@app.route('/star/<koi_id>/save_comment', methods=['POST'])
def save_comment(koi_id):
    file_path = os.path.join('C:\\Users\\Paige\\Projects','miniflask','comment_files',f'{koi_id}_comments.txt')
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
    file_path = os.path.join('C:\\Users\\Paige\\Projects', 'miniflask', 'comment_files', f'{koi_id}_comments.txt')
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
    file_name = star_id + '_lc_detrended.fits'
    file_path = os.path.join('C:\\Users\\Paige\\Projects','miniflask','kepler_lightcurves_for_paige',file_name)
    #get data and create detrended light curve
    if os.path.isfile(file_path):
        data = read_data_from_fits(file_path)
        fig = px.scatter(data, x="TIME", y="FLUX")#, 
                    #title="Kepler Detrended Light Curve")
        graph1JSON= json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder) 
        return jsonify(graph1JSON)
    else:
        error_message = f'No data found for {koi_id}'
        return jsonify(error_message=error_message)
        
# light curve file: has quarter, use this to separate viewing, dashed line
# function that generates plot 2, individual transits
@app.route('/plot2/<koi_id>/<int:line_number>')#<int:start_time>/<int:end_time>')
def plot2(koi_id, line_number): 
    if (fetch_data(koi_id, line_number)):
        photometry_data, true_transit_number, center_time = fetch_data(koi_id, line_number)

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

@app.route('/get_transit_numbers/<koi_id>')
def get_transit_numbers(koi_id):
    transit_values, center_time_values = read_center_time_values_from_file(koi_id)
    return jsonify(transit_values)


def o_minus_c_plot():

    return

def folded_transit_plot():

    return


if __name__ == '__main__':
    app.run(debug=True)


# to plotly function to take data and make interactive 
