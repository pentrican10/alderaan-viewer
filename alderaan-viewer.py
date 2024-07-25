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
from scipy.spatial import ConvexHull
import batman
import matplotlib.pyplot as plt, mpld3
import seaborn as sns
import tempfile
import base64
import io
from plotly.subplots import make_subplots
from scipy.interpolate import interp1d



#sys.path.append('c:\\Users\\Paige\\Projects\\alderaan\\')
data_directory = 'c:\\Users\\Paige\\Projects\\data\\alderaan_results'
K_id = True
table = '2023-05-19_singles.csv'

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
    global K_id
    table = request.args.get('table', '2023-05-19_singles.csv')
    update_data_directory(table)
    ### set switch to use K versus S(simulation data) based on table selected
    if (table == '2023-05-19_singles.csv') or (table == '2023-05-15_doubles.csv'):
        K_id = False
    else: 
        K_id = True
    table_data = data_load.read_table_data(table)
    left_content = render_template('left.html', table_data=table_data)
    right_top_content = render_template('right_top.html')
    right_bottom_content = render_template('right_bottom.html')
    return render_template('index.html',left_content=left_content, right_top_content=right_top_content, right_bottom_content=right_bottom_content)

def update_data_directory(selected_table):
    global data_directory
    global table
    data_directory = os.path.join('c:\\Users\\Paige\\Projects\\data\\alderaan_results', selected_table[:-4])
    table = selected_table

@app.route('/review_status/<koi_id>', methods=['POST'])
def review_status(koi_id):
    global data_directory
    global table
    ### get review status from dropdown
    data = request.json
    review_status = data['reviewStatus']
    file_path = os.path.join(data_directory, table)
    table_data = []
    with open(file_path, 'r') as file:
        reader = csv.DictReader(file)
        fieldnames = reader.fieldnames
        for row in reader:
            if row['koi_id'] == koi_id:
                row['review'] = review_status
            table_data.append(row)
    with open(file_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(table_data)
    
    return jsonify({'status': 'success'})


@app.route('/table_color/')
def table_color():
    global data_directory
    global table
    file_path = os.path.join(data_directory, table)
    
    # Read CSV and prepare review status data
    table_data = []
    with open(file_path, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            table_data.append({
                'koi_id': row['koi_id'],
                'review': row['review']
            })
    
    return jsonify(table_data)


@app.route('/star/<koi_id>')
def display_comment_file(koi_id):
    """
    Function to display comment file associated with KOI ID
    args: 
        koi_id: string, format K00000
    """
    global K_id
    if K_id == False:
        star_id = koi_id.replace("K","S")
    else:
        star_id = koi_id
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
    global K_id
    if K_id==False:
        star_id = koi_id.replace("K","S")
    else:
        star_id = koi_id
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
    global K_id
    if K_id==False:
        star_id = koi_id.replace("K","S")
    else:
        star_id = koi_id
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
    global K_id
    if K_id==False:
        star_id = koi_id.replace("K","S")
    else:
        star_id = koi_id
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

        # Iterate through file paths
        for i, file_path in enumerate(file_paths):
            if os.path.isfile(file_path):
                index, center_time, model, out_prob, out_flag = data_load.get_ttv_file(koi_id, file_path)

                # Add a dot for each center time
                minimum = data_sc.FLUX.min() -0.00001
                offset = 0.0001*i
                y_pts = minimum* np.ones(len(center_time)) + offset
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

        ###################
        colors = ['orange','green','blue','pink','red','purple']

        # Iterate through file paths
        for i, file_path in enumerate(file_paths):
            if os.path.isfile(file_path):
                index, center_time, model, out_prob, out_flag = data_load.get_ttv_file(koi_id, file_path)

                # Add a dot for each center time
                minimum = data_lc.FLUX.min() -0.00001
                offset = 0.0001*i
                y_pts = minimum* np.ones(len(center_time)) + offset
                c_time = px.scatter(x=center_time, y=y_pts).data[0]
                color = colors[i]
                c_time.marker.update(symbol="circle", size=4, color=color)
                c_time.name = f"ttime 0{i}"
                fig.add_trace(c_time, row=1, col=1)
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

        ###################
        colors = ['orange','green','blue','pink','red','purple']

        # Iterate through file paths
        for i, file_path in enumerate(file_paths):
            if os.path.isfile(file_path):
                index, center_time, model, out_prob, out_flag = data_load.get_ttv_file(koi_id, file_path)

                # Add a dot for each center time
                minimum = data_sc.FLUX.min() -0.00001
                offset = 0.0001*i
                y_pts = minimum* np.ones(len(center_time)) + offset
                c_time = px.scatter(x=center_time, y=y_pts).data[0]
                color = colors[i]
                c_time.marker.update(symbol="circle", size=4, color=color)
                c_time.name = f"ttime 0{i}"
                fig.add_trace(c_time, row=1, col=1)
        # Update x-axis label with units
        fig.update_layout(xaxis_title=f"TIME (DAYS)", yaxis_title="FLUX")
        graph1JSON= json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder) 
        return jsonify(graph1JSON)
    
    else:
        error_message = f'No data found for {koi_id}'
        return jsonify(error_message=error_message)

@app.route('/generate_plot_single_transit/<koi_id>/<int:line_number>/<planet>')
def generate_plot_single_transit(koi_id, line_number,planet):
    global K_id
    if K_id==False:
        star_id = koi_id.replace("K","S")
    else:
        star_id = koi_id
    ttv_file = star_id + planet
    ext = os.path.basename(data_directory) +'.csv'
    csv_file_path = os.path.join(data_directory, ext)

    period = data_load.get_periods_for_koi_id(csv_file_path, koi_id)

    planet_num = re.findall(r'\d+', planet)
    num = planet_num[0][1]
    int_num = int(num)
    title = period[int_num]

    file_name_lc = star_id + '_lc_filtered.fits'
    file_path_lc = os.path.join(data_directory,star_id,file_name_lc)
    
    file_name_sc = star_id + '_sc_filtered.fits'
    file_path_sc = os.path.join(data_directory, star_id, file_name_sc)

    ### posteriors for most likely model
    file_results =star_id + '-results.fits'
    file_path_results = os.path.join(data_directory, star_id, file_results)
    data_post = data_load.load_posteriors(file_path_results,num,koi_id)
    ### get max likelihood
    max_index = data_post['LN_LIKE'].idxmax()
    ### get most likely params {P, t0, Rp/Rs, b, T14, q1, q2}
    theta = batman.TransitParams()
    theta.per = data_post[f'P'][max_index]
    theta.t0 = 0.
    theta.rp = data_post[f'ROR_{num}'][max_index]
    theta.b = data_post[f'IMPACT_{num}'][max_index]
    theta.T14 = data_post[f'DUR14_{num}'][max_index]
    LD_U1 = data_post[f'LD_U1'][max_index]
    LD_U2 = data_post[f'LD_U2'][max_index]
    theta.u = [LD_U1, LD_U2]
    theta.limb_dark = 'quadratic'
    
    ### initialize figure
    fig = make_subplots(rows=1, cols=1)

    if (data_load.single_data(koi_id, line_number,num,ttv_file)):
        photometry_data_lc,photometry_data_sc, transit_number, center_time = data_load.single_data(koi_id, line_number, num, ttv_file)
        center_time = np.asarray(center_time, dtype=np.float64)
        #transit_lc = px.scatter(photometry_data_lc, x="TIME", y="FLUX").data[0]
        # transit_lc = go.Scatter(x=photometry_data_lc.TIME, y=photometry_data_lc.FLUX, mode='markers')
        # transit_lc.marker.update(color="blue")
        # fig.add_trace(transit_lc, row=1, col=1)

        # #transit_sc = px.scatter(photometry_data_sc, x="TIME", y="FLUX").data[0]
        # transit_sc = go.Scatter(x=photometry_data_sc.TIME, y=photometry_data_sc.FLUX, mode='markers')
        # transit_sc.marker.update(color="blue")
        # fig.add_trace(transit_sc,row=1,col=1)
        if os.path.isfile(file_path_lc) and os.path.isfile(file_path_sc):
            transit_lc = go.Scatter(x=photometry_data_lc.TIME, y=photometry_data_lc.FLUX, mode='markers')
            transit_lc.marker.update(color="blue")
            fig.add_trace(transit_lc, row=1, col=1)

            transit_sc = go.Scatter(x=photometry_data_sc.TIME, y=photometry_data_sc.FLUX, mode='markers')
            transit_sc.marker.update(color="blue")
            fig.add_trace(transit_sc,row=1,col=1)
            lc_min,lc_max,sc_min,sc_max = data_load.get_min_max(koi_id)
            if len(photometry_data_sc)>0:
                ### transit model
                scit = 1.15e-5
                t = np.arange(photometry_data_sc.TIME.min(), photometry_data_sc.TIME.max(),scit)
                m = batman.TransitModel(theta, t-center_time)    #initializes model
                flux = m.light_curve(theta)          #calculates light curve
                mod = go.Scatter(x=t, y=flux, mode="lines", line=dict(color='red'))
                fig.add_trace(mod,row=1,col=1)


                fig.update_layout(xaxis_title=f"TIME (DAYS)", 
                            yaxis_title="FLUX",
                            yaxis=dict(range=[sc_min, sc_max])
                )
            else:
                ### transit model
                scit = 1.15e-5
                t = np.arange(photometry_data_lc.TIME.min(), photometry_data_lc.TIME.max(),scit)
                m = batman.TransitModel(theta, t-center_time)    #initializes model
                flux = m.light_curve(theta)          #calculates light curve
                mod = go.Scatter(x=t, y=flux, mode="lines", line=dict(color='red'))
                fig.add_trace(mod,row=1,col=1)

                fig.update_layout(xaxis_title=f"TIME (DAYS)", 
                            yaxis_title="FLUX",
                            yaxis=dict(range=[lc_min, lc_max])
                )
        elif os.path.isfile(file_path_lc) and not os.path.isfile(file_path_sc):
            transit_lc = go.Scatter(x=photometry_data_lc.TIME, y=photometry_data_lc.FLUX, mode='markers')
            transit_lc.marker.update(color="blue")
            fig.add_trace(transit_lc, row=1, col=1)
            lc_min,lc_max = data_load.get_min_max(koi_id)
            ### transit model
            scit = 1.15e-5
            t = np.arange(photometry_data_lc.TIME.min(), photometry_data_lc.TIME.max(),scit)
            m = batman.TransitModel(theta, t-center_time)    #initializes model
            flux = m.light_curve(theta)          #calculates light curve
            mod = go.Scatter(x=t, y=flux, mode="lines", line=dict(color='red'))
            fig.add_trace(mod,row=1,col=1)

            fig.update_layout(xaxis_title=f"TIME (DAYS)", 
                        yaxis_title="FLUX",
                        yaxis=dict(range=[lc_min, lc_max])
            )
        elif os.path.isfile(file_path_sc) and not os.path.isfile(file_path_lc):
            transit_sc = go.Scatter(x=photometry_data_sc.TIME, y=photometry_data_sc.FLUX, mode='markers')
            transit_sc.marker.update(color="blue")
            fig.add_trace(transit_sc,row=1,col=1)
            sc_min,sc_max = data_load.get_min_max(koi_id)
            ### transit model
            scit = 1.15e-5
            t = np.arange(photometry_data_sc.TIME.min(), photometry_data_sc.TIME.max(),scit)
            m = batman.TransitModel(theta, t-center_time)    #initializes model
            flux = m.light_curve(theta)          #calculates light curve
            mod = go.Scatter(x=t, y=flux, mode="lines", line=dict(color='red'))
            fig.add_trace(mod,row=1,col=1)

            fig.update_layout(xaxis_title=f"TIME (DAYS)", 
                        yaxis_title="FLUX",
                        yaxis=dict(range=[sc_min, sc_max])
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
    global K_id
    if K_id==False:
        star_id = koi_id.replace("K","S")
    else:
        star_id = koi_id
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
    global K_id
    if K_id==False:
        star_id = koi_id.replace("K","S")
    else:
        star_id = koi_id
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
    global K_id
    if K_id==False:
        star_id = koi_id.replace("K","S")
    else:
        star_id = koi_id
    file_name_lc = star_id + '_lc_filtered.fits'
    file_path_lc = os.path.join(data_directory, star_id, file_name_lc)
    file_name_sc = star_id + '_sc_filtered.fits'
    file_path_sc = os.path.join(data_directory, star_id, file_name_sc)
    file_name = star_id + '_*_quick.ttvs'
    file_paths = glob.glob(os.path.join(data_directory,star_id, file_name))
    ext = os.path.basename(data_directory) +'.csv'
    csv_file_path = os.path.join(data_directory, ext)

    file_results =star_id + '-results.fits'
    file_path_results = os.path.join(data_directory, star_id, file_results)

    ### number of planets from number of ttv files
    npl = len(file_paths)
    subplot_height=350
    titles = data_load.get_periods_for_koi_id(csv_file_path, koi_id)
    
    fig = make_subplots(rows=npl*2, cols=1,
                        subplot_titles = titles,
                        row_heights=[subplot_height, subplot_height*0.4]*npl,
                        vertical_spacing=0.15)
    
    for i, file_path in enumerate(file_paths):
        planet_num = 0+i
        fold_data_lc, fold_data_sc, binned_avg,center_time = data_load.folded_data(koi_id,planet_num,file_path)
        N_samp = 1000
        if len(fold_data_lc) > N_samp:
            fold_data_lc = fold_data_lc.sample(N_samp)
        if len(fold_data_sc) > N_samp:
            fold_data_sc = fold_data_sc.sample(N_samp)
        ### posteriors for most likely model
        
        data_post = data_load.load_posteriors(file_path_results,planet_num,koi_id)
        ### get max likelihood
        max_index = data_post['LN_LIKE'].idxmax()
        data_post = data_post.sort_values(by='LN_LIKE', ascending=False) 
        row = data_post.iloc[0] # pick row with highest likelihood
        ### get most likely params {P, t0, Rp/Rs, b, T14, q1, q2}
        theta = batman.TransitParams()
        # theta.per = data_post[f'P'][max_index]
        # theta.t0 = 0.
        # theta.rp = data_post[f'ROR_{planet_num}'][max_index]
        # theta.b = data_post[f'IMPACT_{planet_num}'][max_index]
        # theta.T14 = data_post[f'DUR14_{planet_num}'][max_index]*24
        # LD_U1 = data_post[f'LD_U1'][max_index]
        # LD_U2 = data_post[f'LD_U2'][max_index]
        # theta.u = [LD_U1, LD_U2]
        # theta.limb_dark = 'quadratic'

        theta.per = row[f'P']
        theta.t0 = 0.
        theta.rp = row[f'ROR_{planet_num}']
        theta.b = row[f'IMPACT_{planet_num}']
        theta.T14 = row[f'DUR14_{planet_num}']*24
        LD_U1 = row[f'LD_U1']
        LD_U2 = row[f'LD_U2']
        theta.u = [LD_U1, LD_U2]
        theta.limb_dark = 'quadratic'


        all_residuals = []
        if os.path.exists(file_path_lc) and os.path.exists(file_path_sc):
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

            ### model
            scit = 1.15e-5
            t = np.arange(fold_data_lc.TIME.min(), fold_data_lc.TIME.max(),scit)
            m = batman.TransitModel(theta, t)    #initializes model
            flux = (m.light_curve(theta))        #calculates light curve
            mod = go.Scatter(x=t, y=flux, mode="lines", line=dict(color='red'))
            mod.name = "Model"
            mod.legendgroup=f'{i}'
            fig.add_trace(mod, row=i+1, col=1)

            # Interpolate model flux to match observed times
            interp_model_flux_lc = interp1d(t, flux, kind='linear', fill_value='extrapolate')
            model_flux_lc = interp_model_flux_lc(fold_data_lc.TIME)
            interp_model_flux_sc = interp1d(t, flux, kind='linear', fill_value='extrapolate')
            model_flux_sc = interp_model_flux_sc(fold_data_sc.TIME)
            interp_model_flux_bin = interp1d(t, flux, kind='linear', fill_value='extrapolate')
            model_flux_bin = interp_model_flux_bin(binned_avg.TIME)

            ### Compute residuals
            residuals_lc = fold_data_lc.FLUX - model_flux_lc
            residuals_sc = fold_data_sc.FLUX - model_flux_sc
            residuals_bin = binned_avg.FLUX - model_flux_bin

            # Collect all residuals
            all_residuals.extend(residuals_lc)
            all_residuals.extend(residuals_sc)
            all_residuals.extend(residuals_bin)

            residuals_plot_lc = go.Scatter(x=fold_data_lc.TIME, y=residuals_lc, mode='markers', showlegend=False)
            residuals_plot_lc.marker.update(symbol="circle", size=5, color="blue")
            fig.add_trace(residuals_plot_lc, row=i+2, col=1)

            residuals_plot_sc = go.Scatter(x=fold_data_sc.TIME, y=residuals_sc, mode='markers', showlegend=False)
            residuals_plot_sc.marker.update(symbol="circle", size=4, color="gray")
            fig.add_trace(residuals_plot_sc, row=i+2, col=1)

            residuals_plot_bin = go.Scatter(x=binned_avg.TIME, y=residuals_bin, mode='markers', showlegend=False)
            residuals_plot_bin.marker.update(symbol="square", size=10, color="orange")
            fig.add_trace(residuals_plot_bin, row=i+2, col=1)

            # Add horizontal line at 0 in residual plot
            fig.add_shape(type="line", x0=fold_data_lc.TIME.min(), x1=fold_data_lc.TIME.max(), y0=0, y1=0,
                          line=dict(color="Red"), row= i + 2, col=1)

            ### Update x-axis and y-axis labels for each subplot
            # fig.update_xaxes(title_text="TIME (HOURS)", row=i+1, col=1)
            
            residuals_min = np.percentile(all_residuals, 20)
            residuals_max = np.percentile(all_residuals, 80)
            max_abs_residual = max(abs(residuals_min), abs(residuals_max))
            fig.update_yaxes(range=[-max_abs_residual,max_abs_residual], row= i + 2, col=1)

            fig.update_yaxes(title_text="FLUX", row=i+1, col=1)
            fig.update_xaxes(title_text="TIME (HOURS)", row=i+2, col=1)
            fig.update_yaxes(title_text="Residuals", row=i+2, col=1)
            fig.update_layout(height=700, width=1000)
        
        elif os.path.exists(file_path_lc) and not os.path.exists(file_path_sc):
            fold_lc = go.Scatter(x=fold_data_lc.TIME, y=fold_data_lc.FLUX, mode='markers')
            fold_lc.marker.update(symbol="circle", size=5, color="blue")
            fold_lc.name = "Long Cadence"
            fig.add_trace(fold_lc, row=i+1, col=1)
            ### binned avg
            bin_avg = go.Scatter(x=binned_avg.TIME, y=binned_avg.FLUX, mode='markers')
            bin_avg.marker.update(symbol="square", size=10, color="orange")
            bin_avg.name = "Binned Average"
            fig.add_trace(bin_avg, row=i+1, col=1)
            ### model
            scit = 1.15e-5
            t = np.arange(fold_data_lc.TIME.min(), fold_data_lc.TIME.max(),scit)
            m = batman.TransitModel(theta, t)    #initializes model
            flux = m.light_curve(theta)          #calculates light curve
            mod = go.Scatter(x=t, y=flux, mode="lines")
            mod.line.update(color="red")
            mod.name = "Model"
            mod.legendgroup=f'{i}'
            fig.add_trace(mod, row=i+1, col=1)

            # Interpolate model flux to match observed times
            interp_model_flux_lc = interp1d(t, flux, kind='linear', fill_value='extrapolate')
            model_flux_lc = interp_model_flux_lc(fold_data_lc.TIME)
            interp_model_flux_bin = interp1d(t, flux, kind='linear', fill_value='extrapolate')
            model_flux_bin = interp_model_flux_bin(binned_avg.TIME)

            ### Compute residuals
            residuals_lc = fold_data_lc.FLUX - model_flux_lc
            residuals_bin = binned_avg.FLUX - model_flux_bin

            # Collect all residuals
            all_residuals.extend(residuals_lc)
            all_residuals.extend(residuals_bin)

            residuals_plot_lc = go.Scatter(x=fold_data_lc.TIME, y=residuals_lc, mode='markers', showlegend=False)
            residuals_plot_lc.marker.update(symbol="circle", size=5, color="blue")
            fig.add_trace(residuals_plot_lc, row=i+2, col=1)

            residuals_plot_bin = go.Scatter(x=binned_avg.TIME, y=residuals_bin, mode='markers', showlegend=False)
            residuals_plot_bin.marker.update(symbol="square", size=10, color="orange")
            fig.add_trace(residuals_plot_bin, row=i+2, col=1)

            # Add horizontal line at 0 in residual plot
            fig.add_shape(type="line", x0=fold_data_lc.TIME.min(), x1=fold_data_lc.TIME.max(), y0=0, y1=0,
                          line=dict(color="Red"), row= i + 2, col=1)

            ### Update x-axis and y-axis labels for each subplot
            # fig.update_xaxes(title_text="TIME (HOURS)", row=i+1, col=1)
            
            residuals_min = np.percentile(all_residuals, 20)
            residuals_max = np.percentile(all_residuals, 80)
            max_abs_residual = max(abs(residuals_min), abs(residuals_max))
            fig.update_yaxes(range=[-max_abs_residual,max_abs_residual], row= i + 2, col=1)


            ### Update x-axis and y-axis labels for each subplot
            fig.update_yaxes(title_text="FLUX", row=i+1, col=1)
            fig.update_xaxes(title_text="TIME (HOURS)", row=i+2, col=1)
            fig.update_yaxes(title_text="Residuals", row=i+2, col=1)
            fig.update_layout(height=700, width=1000)
            

        elif not os.path.exists(file_path_lc) and os.path.exists(file_path_sc):
            fold_sc = go.Scatter(x=fold_data_sc.TIME, y=fold_data_sc.FLUX, mode='markers')
            fold_sc.marker.update(symbol="circle", size=4, color="gray")
            fold_sc.name = "Short Cadence"
            fig.add_trace(fold_sc, row=i+1, col=1)
            ### binned avg
            bin_avg = go.Scatter(x=binned_avg.TIME, y=binned_avg.FLUX, mode='markers')
            bin_avg.marker.update(symbol="square", size=10, color="orange")
            bin_avg.name = "Binned Average"
            fig.add_trace(bin_avg, row=i+1, col=1)
            ### model
            scit = 1.15e-5
            t = np.arange(fold_data_sc.TIME.min(), fold_data_sc.TIME.max(),scit)
            m = batman.TransitModel(theta, t)    #initializes model
            flux = (m.light_curve(theta))        #calculates light curve
            mod = go.Scatter(x=t, y=flux, mode="lines", line=dict(color='red'))
            mod.line.update(color="red")
            mod.name = "Model"
            mod.legendgroup=f'{i}'
            fig.add_trace(mod, row=i+1, col=1)

            # Interpolate model flux to match observed times
            interp_model_flux_sc = interp1d(t, flux, kind='linear', fill_value='extrapolate')
            model_flux_sc = interp_model_flux_sc(fold_data_sc.TIME)
            interp_model_flux_bin = interp1d(t, flux, kind='linear', fill_value='extrapolate')
            model_flux_bin = interp_model_flux_bin(binned_avg.TIME)

            ### Compute residuals
            residuals_sc = fold_data_sc.FLUX - model_flux_sc
            residuals_bin = binned_avg.FLUX - model_flux_bin

            # Collect all residuals
            all_residuals.extend(residuals_sc)
            all_residuals.extend(residuals_bin)

            residuals_plot_sc = go.Scatter(x=fold_data_sc.TIME, y=residuals_sc, mode='markers', showlegend=False)
            residuals_plot_sc.marker.update(symbol="circle", size=4, color="gray")
            fig.add_trace(residuals_plot_sc, row=i+2, col=1)

            residuals_plot_bin = go.Scatter(x=binned_avg.TIME, y=residuals_bin, mode='markers', showlegend=False)
            residuals_plot_bin.marker.update(symbol="square", size=10, color="orange")
            fig.add_trace(residuals_plot_bin, row=i+2, col=1)

            # Add horizontal line at 0 in residual plot
            fig.add_shape(type="line", x0=fold_data_lc.TIME.min(), x1=fold_data_lc.TIME.max(), y0=0, y1=0,
                          line=dict(color="Red"), row= i + 2, col=1)

            ### Update x-axis and y-axis labels for each subplot
            # fig.update_xaxes(title_text="TIME (HOURS)", row=i+1, col=1)
            
            residuals_min = np.percentile(all_residuals, 20)
            residuals_max = np.percentile(all_residuals, 80)
            max_abs_residual = max(abs(residuals_min), abs(residuals_max))
            fig.update_yaxes(range=[-max_abs_residual,max_abs_residual], row= i + 2, col=1)


            ### Update x-axis and y-axis labels for each subplot
            fig.update_yaxes(title_text="FLUX", row=i+1, col=1)
            fig.update_xaxes(title_text="TIME (HOURS)", row=i+2, col=1)
            fig.update_yaxes(title_text="Residuals", row=i+2, col=1)
            fig.update_layout(height=700, width=1000)
        
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
    global K_id
    if K_id==False:
        star_id = koi_id.replace("K","S")
    else:
        star_id = koi_id
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

"""   
@app.route('/generate_plot_corner/<koi_id>/<selected_columns>/<planet_num>')
def generate_plot_corner(koi_id,selected_columns, planet_num):
    try:
        selected_columns = selected_columns.split(',')
        global K_id
        if K_id==False:
            star_id = koi_id.replace("K","S")
        else:
            star_id = koi_id
        file =star_id + '-results.fits'
        file_path = os.path.join(data_directory, star_id, file)

        if os.path.isfile(file_path):
            data = data_load.load_posteriors(file_path,planet_num,koi_id)
            # Drop the 'period' column if it exists
            if f'P_{planet_num}' in data.columns:
                data = data.drop(columns=[f'P_{planet_num}'])

            LN_WT = data['LN_WT'][::5].values
            weight = np.exp(LN_WT- LN_WT.max())
            w = weight/ np.sum(weight)
            index = np.arange(len(LN_WT))
            rand_index = np.random.choice(index,p=w,size=len(LN_WT), replace=True)

            data = data[selected_columns]

            labels = data.columns.tolist()
            
            # Create a subplot grid for the corner plot
            fig, axs = plt.subplots(len(selected_columns), len(selected_columns), figsize=(12, 12))
            
            for i in range(len(selected_columns)):
                for j in range(len(selected_columns)):
                    if i == j:
                        # Diagonal plot - histogram or KDE plot
                        x=[1,2,3,4]
                        y=[1,2,3,4]
                        ax = axs[i, j]
                        ax.scatter(x,y)
                        #ax.set_xlabel(labels[i])
                        ax.set_ylabel('Density')
                    elif i > j:
                        # Lower triangle plot - scatter plot or line plot
                        ax = axs[i, j]
                        x = [5,6,7,8]
                        y=[5,6,7,8]
                        ax.scatter(x,y)
                        #ax.set_xlabel(labels[j])
                        #ax.set_ylabel(labels[i])
                    else:
                        # Upper triangle plot - remove axis
                        axs[i, j].remove()
            
            # Adjust layout and convert plot to HTML
            plt.tight_layout()
            mpld3_plot = mpld3.fig_to_html(fig)
            
            # Convert mpld3 plot to JSON format and return as response
            #graphJSON = json.dumps(mpld3_plot)
            # Convert mpld3 plot to JSON format and return as response
            return jsonify(graphJSON=mpld3_plot)
            #return jsonify(graphJSON=graphJSON)
        else:
            error_message = f'No data found for {koi_id}'
            return jsonify(error_message=error_message)
    except Exception as e:
        error_message = f'An error occurred: {str(e)}'
        print(error_message)  # Log the error
        return jsonify(error_message=error_message)
    
"""
@app.route('/generate_plot_corner/<koi_id>/<selected_columns>/<planet_num>')
def generate_plot_corner(koi_id,selected_columns, planet_num):
    selected_columns = selected_columns.split(',')
    global K_id
    if K_id==False:
        star_id = koi_id.replace("K","S")
    else:
        star_id = koi_id
    file =star_id + '-results.fits'
    file_path = os.path.join(data_directory, star_id, file)

    if os.path.isfile(file_path):
        data = data_load.load_posteriors(file_path,planet_num,koi_id)
        
        #set target # samples
        N_samp = 1000
        LN_WT = data['LN_WT'].values
        weight = np.exp(LN_WT- LN_WT.max())
        w = weight/ np.sum(weight)

        data = data.sample(N_samp, replace=True, ignore_index=True, weights=w)
        #data = data[selected_columns]
        LN_WT = data['LN_WT'].values
        weight = np.exp(LN_WT- LN_WT.max())
        w = weight/ np.sum(weight)
        data['WEIGHTS'] = w

        labels = data[selected_columns].columns.tolist()

        fig = make_subplots(rows=len(selected_columns), cols=len(selected_columns))
        tick_values_y_c0 = None
        plot_range_y_c0 = None
        
        for i in range(len(selected_columns)):
            for j in range(i, len(selected_columns)):
                
                x = data[selected_columns[i]].values
                y = data[selected_columns[j]].values
                

                if i != j:
                    # Calculate the density of the points
                    # xy = np.vstack([x, y])
                    
                    # z = gaussian_kde(xy)(xy)
                    # ### plotting threshold
                    # threshold_p=np.percentile(z,1)
                    # # Select the top 90th percentile based on density
                    # threshold_s = np.percentile(z, 10) # scatter threshhold
                    # mask = (z >= threshold_s) 
                    # mask_s = (z >= threshold_p) & (z < threshold_s)
                    #Plot points below the threshold as scatter plot
                    fig.add_trace(go.Scatter(
                        x=x,#[mask_s], 
                        y=y,#[mask_s], 
                        mode='markers', 
                        marker=dict(color='gray', size=1), 
                        showlegend=False
                        ), row=j + 1, col=i + 1)
                    
                    # Determine threshold densities corresponding to the percentiles
                    # percentiles = [10, 25, 50, 75, 90]  # Reversed order

                    # # Define contour levels
                    # contour_levels = [10, 68, 98, 99]  # Adjust as needed
                    # fig.add_trace(go.Histogram2dContour(
                    #     x=x[mask],
                    #     y=y[mask],
                    #     colorscale='Blues',
                    #     reversescale=False,
                    #     xaxis='x',
                    #     yaxis='y',
                    #     contours=dict(
                    #         start=min(contour_levels),
                    #         end=max(contour_levels),
                    #         size=(max(contour_levels) - min(contour_levels)) / len(contour_levels),
                    #     ),
                    #     showscale=False,
                    # ), row=j + 1, col=i + 1)
                    
                else:
                    kde = gaussian_kde(x, weights=data['WEIGHTS']) 
                    max1 = max(x)
                    min1=min(x)
                    scale = max1 - min1
                    buffer = 0.01 * scale
                    x_vals = np.linspace(min(x)-buffer, max(x)+buffer, 1000)
                    y_vals = kde(x_vals)
                    fig.add_trace(go.Scatter(x=x_vals, y=y_vals, mode='lines', line=dict(color='blue'), name=labels[i], showlegend=False), row=j + 1, col=i + 1)
                    if labels[j]==f'IMPACT_{planet_num}':
                        # Add vertical line at x=1
                        fig.add_trace(go.Scatter(x=[1, 1], y=[0, np.max(y_vals)], mode='lines', line=dict(color='black', dash='dash')), row=j + 1, col=i + 1)

                ### add labels to x and y axes
                if (i == 0) and (i != j):
                    fig.update_yaxes(title_text=labels[j], row=j + 1, col=i + 1)
                if j == len(selected_columns) - 1:
                    fig.update_xaxes(title_text=labels[i], row=j + 1, col=i + 1)

                ### only have 3 ticks and only show axis at left-most column and bottom-most row
                if i != j:
                    tick_format = '.3f'
                    plot_range = None
                    ### set y axes
                    if labels[j] == f'IMPACT_{planet_num}':
                        if max(y)<1:
                            tick_values_y = np.linspace(0, 1.2, 3)
                            plot_range = [0, 1.2]
                            tick_format = '.1f'
                        else:
                            buffer = 0.05
                            maxy = max(y)
                            rng = buffer + maxy
                            tick_values_y = np.linspace(0, rng, 3)
                            plot_range = [0, rng]
                            tick_format = '.1f'
                    else:
                        tick_values_y = np.linspace(min(y), max(y), 3)
                        plot_range = [min(y), max(y)]

                    tick_text_y = [f"{val:{tick_format}}" for val in tick_values_y]
                    fig.update_yaxes(tickvals=tick_values_y, ticktext=tick_text_y, range=plot_range, row=j + 1, col=i + 1, tickangle=0)

                    ### set x axes
                    if labels[i] == f'IMPACT_{planet_num}':
                        if max(x)<1:
                            tick_values_x = np.linspace(0, 1.2, 3)
                            plot_range = [0, 1.2]
                            tick_format = '.1f'
                        else:
                            buffer = 0.05
                            maxx = max(x)
                            rng = buffer + maxx
                            tick_values_x = np.linspace(0, rng, 3)
                            plot_range = [0, rng]
                            tick_format = '.1f'
                    else:
                        tick_values_x = np.linspace(min(x), max(x), 3)
                        plot_range = [min(x), max(x)]
                    tick_text_x = [f"{val:{tick_format}}" for val in tick_values_x]

                    if (i!=0):
                        fig.update_yaxes(showticklabels=False, tickvals=tick_values_y, ticktext=tick_text_y, row=j + 1, col=i + 1, tickangle=0)
                    fig.update_xaxes(range=plot_range, row=j + 1, col=i + 1)
                        
                    if j == len(selected_columns) - 1:
                        fig.update_xaxes(tickvals=tick_values_x, ticktext=tick_text_x, row=j + 1, col=i + 1, tickangle=90)
                    else:
                        fig.update_xaxes(showticklabels=False, tickvals=tick_values_x, ticktext=tick_text_x, row=j + 1, col=i + 1, tickangle=90)
                else:
                    tick_format = '.3f'
                    ### histograms only have x axes
                    if labels[i] == f'IMPACT_{planet_num}':
                        if max(x)<1:
                            tick_values_x = np.linspace(0, 1.2, 3)
                            plot_range = [0, 1.2]
                            tick_format = '.1f'
                        else:
                            buffer = 0.05
                            maxx = max(x)
                            rng = buffer + maxx
                            tick_values_x = np.linspace(0, rng, 3)
                            plot_range = [0, rng]
                            tick_format = '.1f'
                    else:
                        tick_values_x = np.linspace(min(x), max(x), 3)
                        plot_range = [min(x), max(x)]
                    tick_values_y = np.linspace(min(y_vals), max(y_vals), 3)
                    tick_text_x = [f"{val:{tick_format}}" for val in tick_values_x]
                    tick_text_y = [f"{val:{tick_format}}" for val in tick_values_y]

                    fig.update_xaxes(tickvals=tick_values_x, ticktext=tick_text_x, row=j + 1, col=i + 1, tickangle=0)
                    fig.update_yaxes(showticklabels=False,tickvals=tick_values_y, row=j + 1, col=i + 1, tickangle=0)
                    fig.update_xaxes(range=plot_range, row=j + 1, col=i + 1)
                    

                fig.update_layout(plot_bgcolor='#F7FBFF') # match background with the back contour color
                fig.update_xaxes(showline=True, linewidth=1, linecolor='black', mirror=True, row=j + 1, col=i + 1, tickangle=30)
                fig.update_yaxes(showline=True, linewidth=1, linecolor='black', mirror=True, row=j + 1, col=i + 1, tickangle=0)
        
        
        fig.update_layout(height=800, width=900)
        graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder) 
        return jsonify(graphJSON)
    else:
        error_message = f'No data found for {koi_id}'
        return jsonify(error_message=error_message)



if __name__ == '__main__':
    app.run(debug=True)
