from flask import Flask, render_template, jsonify, request, session, redirect, url_for
import os
import csv
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import json
import plotly.utils
import numpy as np
import glob
import plotly.graph_objects as go
import re
from scipy.stats import gaussian_kde
import batman
from plotly.subplots import make_subplots
import random
import data_load

K_id = True
table = ''# 'ecc-all-LC.csv' 
app = Flask(__name__)
app.secret_key = 'super_secret'

data_directory = ''

### Dynamically determine the root directory of the Flask app
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))  # Root directory of the app
### Move one level up from the root directory
PARENT_DIR = os.path.dirname(ROOT_DIR)
### Set the default directory to the parent directory's 'alderaan/Results' path
default_directory = os.path.join(PARENT_DIR, 'alderaan', 'Results')


@app.route('/')
def index():
    """
    Initializes web app and sends to login screen
    """
    session.clear()
    return redirect(url_for('login'))


@app.route('/login', methods=['GET', 'POST'])
def login():
    """
    Takes username entered into input box on login screen
    Saves username for session, keeps track of comments made
    """
    if request.method == 'POST':
        username = request.form.get('username')
        if username:
            session['username'] = username
            return redirect(url_for('display_table_data'))
    return render_template('login.html')

@app.route('/logout',methods=['POST'])
def logout():
    """
    Button that ends session, logs out user
    """
    session.pop('username',None) #removes username
    return redirect(url_for('login'))

@app.route('/home')
def display_table_data():
    """
    Called after user logs in:
    Assigns each html file to their respective locations
    Renders Index Template
    """
    global K_id
    global table
    global default_directory
    ### List all items (files and folders) in the directory 
    options = os.listdir(default_directory) 
    table = request.args.get('table', options[0] + '.csv')  # default table option is first folder in directory
   
    update_data_directory(table)
    ### set switch to use K versus S(simulation data) based on table selected
    if 'SIMULATION' in table:
        K_id = False 
    else: 
        K_id = True
    if not table.startswith('.'):
        table_data = data_load.read_table_data(table)
    else:
        table_data = 'error'
    left_content = render_template('left.html', table_data=table_data)
    right_top_content = render_template('right_top.html')
    right_bottom_content = render_template('right_bottom.html')
    return render_template('index.html',left_content=left_content, right_top_content=right_top_content, right_bottom_content=right_bottom_content)

def update_data_directory(selected_table):
    """
    Function adds table name to data directory. 
    The data directory is then routed to the file containing the selected table and all corresponding data.

    args:
        selected_table: string of table name. This must be in the form 'table_name.csv'
    """
    global data_directory
    global table 
    data_directory = os.path.join(default_directory, selected_table[:-4])
    table = selected_table 


@app.route('/get_dropdown_options')
def get_dropdown_options():
    '''
    Function populates options based on folders in the default directory

    Returns:
        jsonify(options): list of options in json-readable format
    '''
    global default_directory
    ### List all directories in the default directory
    options = sorted([item for item in os.listdir(default_directory) if os.path.isdir(os.path.join(default_directory, item))])
    options_with_tables = []
    for option in options: 
        if os.path.isfile(os.path.join(default_directory, option, option + '.csv')):
            options_with_tables.append(option)
    ### Return the list of options as a JSON response
    return jsonify(options_with_tables)
    
    
@app.route('/get_selected_table', methods=['GET'])
def get_selected_table():
    """
    Function passes the selected table as JSON response to display
    """
    try:
        ### Return the selected table as a JSON response
        selected_table = {'selected_table': table}
        return jsonify(selected_table)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/planet_properties/<koi_id>', methods=['GET'])
def get_planet_properties(koi_id):
    """
    Function gets planet properties used in the table on the web app.

    args:
        koi_id: string in the form "K00000", KOI identification

    returns:
        jsonify(planet_data): planet properties table data in json passed to html/javascript
    """
    global table
    planet_data = data_load.get_planet_properties_table(koi_id,table)
    return jsonify(planet_data)

@app.route('/period_ratios/<koi_id>', methods=['GET'])
def get_period_ratios(koi_id):
    """
    Function gets period ratios used in the table on the web app.

    args:
        koi_id: string in the form "K00000", KOI identification

    returns:
        jsonify(ratio_data): period ratio table data in json passed to html/javascript
    """
    global table
    ratio_data = data_load.get_period_ratios_table(koi_id,table)
    return jsonify(ratio_data)


@app.route('/review_status/<koi_id>', methods=['POST'])
def review_status(koi_id):
    """
    Function handling the review status dropdown menu. 
    Assigns review status chosen to the associated koi_id in the csv table

    args:
        koi_id: string in the form "K00000" (KOI identification)

    returns: 
        jsonify({'status': 'success'}): sends a success message to console(see right_top.html) if review status was written to the table

    """
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
    """
    Function that retrieves review status for javascript function to assign colors to table based on review status value.
    
    returns:
        jsonify(review_data): passes review status and koi_id to html
    """
    global data_directory
    global table
    file_path = os.path.join(data_directory, table)
    
    # Read CSV and prepare review status data
    review_data = []
    with open(file_path, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            review_data.append({
                'koi_id': row['koi_id'],
                'review': row['review']
            })
    
    return jsonify(review_data)


@app.route('/star/<koi_id>')
def display_comment_file(koi_id):
    """
    Function to display comment file associated with KOI ID
    args: 
        koi_id: string in the form "K00000" (KOI identification)

    returns:
        file_content: string, content of comments file for koi_id passed to html to be displayed
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
    args: 
        koi_id: string in the form "K00000" (KOI identification)
    
    returns:
        Calls display_comment_file() function to display updated comment file
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
    Function saves the comment file and displays the updated file or an error message.
    A comment file will be created if there is not one in path data_directory/koi_id/koi_id_comments.txt

    args:
        koi_id: string in the form "K00000" (KOI identification)

    returns:
        Calls display_comment_file() function or gives error is something went wrong
    """
    global K_id
    if K_id==False:
        star_id = koi_id.replace("K","S")
    else:
        star_id = koi_id
    path_extension = os.path.join(star_id, f'{star_id}_comments.txt')
    file_path = os.path.join(data_directory, path_extension)
    content = request.form.get('content')
    ### Normalize line endings to Unix-style (\n)
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
    """
    Function generates the detrended light curve plot and passes the figure to be displayed on html

    args:
        koi_id: string in the form "K00000" (KOI identification)

    returns: 
        jsonify(graph1JSON): figure able to be read and displayed in html
    """
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
        data_lc = data_load.load_photometry_data(file_path_lc)
        data_sc = data_load.load_photometry_data(file_path_sc)

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

        ### mark quarters 
        ### Define a constant y-value for horizontal lines above the data
        horizontal_line_y = max(data_sc['FLUX']) + 0.0001 * max(data_sc['FLUX'])  # Adjust offset as needed
        quarter_colors = ['green','red','blue','orange'] 


        def get_color(quarter):
            # Use modulo 4 to group quarters
            return quarter_colors[quarter % 4]

        # Mark quarters for Long Cadence data
        unique_quarters_lc = data_lc['QUARTER'].unique()
        for idx, quarter in enumerate(unique_quarters_lc):
            times = data_lc.loc[data_lc['QUARTER'] == quarter, 'TIME']
            start = times.min()
            end = times.max()
            line_color = get_color(quarter)

            # Add horizontal lines at the top of the plot
            fig.add_shape(
                type="line",
                x0=start,
                x1=end,
                y0=horizontal_line_y,
                y1=horizontal_line_y,
                line=dict(color=line_color, width=4),
                name=f"Quarter {quarter}"
            )

            ### Add an invisible scatter trace for hover information
            hover_trace = go.Scatter(
                x=[(start + end) / 2],  # Position the hover text in the middle of the line
                y=[horizontal_line_y],
                mode='markers',
                marker=dict(opacity=0),  # Make the marker invisible
                showlegend=False,
                hoverinfo='text',
                text=f"Quarter {quarter}",
                hoverlabel=dict(bgcolor=line_color, font=dict(color='white'))  # Set hover text background color and font color
            )
            fig.add_trace(hover_trace)

        ### Mark quarters for Short Cadence data
        unique_quarters_sc = data_sc['QUARTER'].unique()
        for idx, quarter in enumerate(unique_quarters_sc):
            times = data_sc.loc[data_sc['QUARTER'] == quarter, 'TIME']
            start = times.min()
            end = times.max()
            line_color = get_color(quarter)

            ### Add horizontal lines at the top of the plot
            fig.add_shape(
                type="line",
                x0=start,
                x1=end,
                y0=horizontal_line_y,
                y1=horizontal_line_y,
                line=dict(color=line_color, width=4),
                name=f"Quarter {quarter}"
            )

            ### Add an invisible scatter trace for hover information
            hover_trace = go.Scatter(
                x=[(start + end) / 2],  # Position the hover text in the middle of the line
                y=[horizontal_line_y],
                mode='markers',
                marker=dict(opacity=0),  # Make the marker invisible
                showlegend=False,
                hoverinfo='text',
                text=f"Quarter {quarter}",
                hoverlabel=dict(bgcolor=line_color, font=dict(color='white'))  # Set hover text background color and font color
            )
            fig.add_trace(hover_trace)
        ###################
        colors = ['orange','green','red','orange','green','red','orange','green','red']

        ### Iterate through file paths
        for i, file_path in enumerate(file_paths):
            if os.path.isfile(file_path):
                index, center_time, model, out_prob, out_flag = data_load.load_ttv_data(koi_id, file_path)

                ### Add a dot for each center time
                minimum = data_sc.FLUX.min() -0.00001
                offset = 0.0001*i
                y_pts = minimum* np.ones(len(center_time)) + offset
                c_time = px.scatter(x=center_time, y=y_pts).data[0]
                color = colors[i]
                c_time.marker.update(symbol="circle", size=4, color=color)
                c_time.name = f"ttime 0{i}"
                fig.add_trace(c_time, row=1, col=1)
                

        ### Update x-axis label with units
        fig.update_traces(showlegend=True, row=1, col=1)
        fig.update_layout(xaxis_title=f"TIME (DAYS)", yaxis_title="FLUX")
        fig.update_layout(title=star_id, title_x=0.5)
        fig.update_layout(legend=dict(traceorder="normal"))
        graph1JSON= json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder) 
        return jsonify(graph1JSON)
    
    elif os.path.isfile(file_path_lc):
        data_lc = data_load.load_photometry_data(file_path_lc)
        lc = px.scatter(data_lc, x="TIME",y="FLUX").data[0]
        lc.marker.update(symbol="circle", size=4, color="blue")
        lc.name = "Long Cadence"
        fig.add_trace(lc, row=1, col=1)

        ### mark quarters 
        ### Define a constant y-value for horizontal lines above the data
        horizontal_line_y = max(data_lc['FLUX']) + 0.0001 * max(data_lc['FLUX'])  # Adjust offset as needed
        quarter_colors = ['green','red','blue','orange']

        def get_color(quarter):
            ### Use modulo 4 to group quarters
            return quarter_colors[quarter % 4]
        ### Mark quarters for Long Cadence data
        unique_quarters_lc = data_lc['QUARTER'].unique()
        for idx, quarter in enumerate(unique_quarters_lc):
            times = data_lc.loc[data_lc['QUARTER'] == quarter, 'TIME']
            start = times.min()
            end = times.max()
            line_color = get_color(quarter)

            ### Add horizontal lines at the top of the plot
            fig.add_shape(
                type="line",
                x0=start,
                x1=end,
                y0=horizontal_line_y,
                y1=horizontal_line_y,
                line=dict(color=line_color, width=4),
                name=f"Quarter {quarter}"
            )

            ### Add an invisible scatter trace for hover information
            hover_trace = go.Scatter(
                x=[(start + end) / 2],  # Position the hover text in the middle of the line
                y=[horizontal_line_y],
                mode='markers',
                marker=dict(opacity=0),  # Make the marker invisible
                showlegend=False,
                hoverinfo='text',
                text=f"Quarter {quarter}",
                hoverlabel=dict(bgcolor=line_color, font=dict(color='white'))  # Set hover text background color and font color
            )
            fig.add_trace(hover_trace)

        ###################
        colors = ['orange','green','red','orange','green','red','orange','green','red']

        ### Iterate through file paths
        for i, file_path in enumerate(file_paths):
            if os.path.isfile(file_path):
                index, center_time, model, out_prob, out_flag = data_load.load_ttv_data(koi_id, file_path)

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
        fig.update_traces(showlegend=True, row=1, col=1)
        fig.update_layout(xaxis_title=f"TIME (DAYS)", yaxis_title="FLUX")
        fig.update_layout(title=star_id, title_x=0.5)
        graph1JSON= json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder) 
        return jsonify(graph1JSON)
    
    elif os.path.isfile(file_path_sc):
        data_sc = data_load.load_photometry_data(file_path_sc)
        sc_time_trim = data_sc['TIME'][::30]
        sc_flux_trim = data_sc['FLUX'][::30]
        sc = px.scatter(x=sc_time_trim, y=sc_flux_trim).data[0]
        sc.marker.update(symbol="circle", size=4, color="gray")
        sc.name = "Short Cadence"
        fig.add_trace(sc, row=1, col=1)

        ### mark quarters 
        # Define a constant y-value for horizontal lines above the data
        horizontal_line_y = max(data_sc['FLUX']) + 0.0001 * max(data_sc['FLUX'])  # Adjust offset as needed
        # Use Plotly's predefined color scale
        quarter_colors = ['green','red','blue','orange']
        num_colors = len(quarter_colors)


        def get_color(quarter):
            # Use modulo 4 to group quarters
            return quarter_colors[quarter % 4]
        # Mark quarters for Short Cadence data
        unique_quarters_sc = data_sc['QUARTER'].unique()
        for idx, quarter in enumerate(unique_quarters_sc):
            times = data_sc.loc[data_sc['QUARTER'] == quarter, 'TIME']
            start = times.min()
            end = times.max()
            line_color = get_color(idx)

            # Add horizontal lines at the top of the plot
            fig.add_shape(
                type="line",
                x0=start,
                x1=end,
                y0=horizontal_line_y,
                y1=horizontal_line_y,
                line=dict(color=line_color, width=4),
                name=f"Quarter {quarter}"
            )

            # Add an invisible scatter trace for hover information
            hover_trace = go.Scatter(
                x=[(start + end) / 2],  # Position the hover text in the middle of the line
                y=[horizontal_line_y],
                mode='markers',
                marker=dict(opacity=0),  # Make the marker invisible
                showlegend=False,
                hoverinfo='text',
                text=f"Quarter {quarter}",
                hoverlabel=dict(bgcolor=line_color, font=dict(color='white'))  # Set hover text background color and font color
            )
            fig.add_trace(hover_trace)

        ###################
        colors = ['orange','green','red','orange','green','red','orange','green','red']

        # Iterate through file paths
        for i, file_path in enumerate(file_paths):
            if os.path.isfile(file_path):
                index, center_time, model, out_prob, out_flag = data_load.load_ttv_data(koi_id, file_path)

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
        fig.update_layout(title=star_id, title_x=0.5)
        graph1JSON= json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder) 
        return jsonify(graph1JSON)
    
    else:
        error_message = f'No data found for {koi_id}'
        return jsonify(error_message=error_message)

@app.route('/generate_plot_single_transit/<koi_id>/<int:line_number>/<planet>')
def generate_plot_single_transit(koi_id, line_number,planet):
    """
    Function generates a panel of single transit light curves and passes the figure to be displayed on html

    args:
        koi_id: string in the form "K00000" (KOI identification)
        line_number: integer index corresponding with the transit
        planet: integer planet number 

    returns: 
        jsonify(response_data): contains figure able to be read and displayed in html and transit number
    """
    global K_id
    if K_id==False:
        star_id = koi_id.replace("K","S")
    else:
        star_id = koi_id
    ttv_file = star_id + planet
    ext = os.path.basename(data_directory) +'.csv'
    csv_file_path = os.path.join(data_directory, ext)

    data_per = data_load.get_koi_identifiers(csv_file_path, koi_id)
    data_per = data_per.sort_values(by='periods') 
    koi_identifier = data_per.koi_identifiers.values
    period = data_per.period_title.values

    planet_num = re.findall(r'\d+', planet)
    num = planet_num[0][1]
    int_num = int(num)
    title = koi_identifier[int_num]
    period= period[int_num]

    file_name_lc = star_id + '_lc_filtered.fits'
    file_path_lc = os.path.join(data_directory,star_id,file_name_lc)
    
    file_name_sc = star_id + '_sc_filtered.fits'
    file_path_sc = os.path.join(data_directory, star_id, file_name_sc)

    ### posteriors for most likely model
    file_results =star_id + '-results.fits'
    file_path_results = os.path.join(data_directory, star_id, file_results)
    data_post = data_load.load_posteriors(file_path_results,num,koi_id)
    ### get max likelihood
    data_post = data_post.sort_values(by='LN_LIKE', ascending=False) 
    row = data_post.iloc[0] # pick row with highest likelihood
    ### get most likely params {P, t0, Rp/Rs, b, T14, q1, q2}
    theta = batman.TransitParams()
    theta.per = row[f'P']
    theta.t0 = 0.
    theta.rp = row[f'ROR_{num}']
    theta.b = row[f'IMPACT_{num}']
    theta.T14 = row[f'DUR14_{num}']#*24
    LD_U1 = row[f'LD_U1']
    LD_U2 = row[f'LD_U2']
    theta.u = [LD_U1, LD_U2]
    theta.limb_dark = 'quadratic'

    ### get next 9 line numbers for panel
    line_number_plots = np.arange(line_number, line_number+9)
    row = [1,1,1,2,2,2,3,3,3]
    col = [1,2,3,1,2,3,1,2,3]
    
    ### initialize figure
    fig = make_subplots(rows=3, cols=3) 

    ### Loop through the grid positions and corresponding line numbers
    for i, line_num in enumerate(line_number_plots):
        r = row[i]
        c = col[i]

        if (data_load.single_data(koi_id, line_num,num,ttv_file)):
            photometry_data_lc,photometry_data_sc, transit_number, center_time = data_load.single_data(koi_id, line_num, num, ttv_file)
            center_time = np.asarray(center_time, dtype=np.float64)
            
            if os.path.isfile(file_path_lc) and os.path.isfile(file_path_sc):
                transit_lc = go.Scatter(x=photometry_data_lc.TIME, y=photometry_data_lc.FLUX, mode='markers',showlegend=False)
                transit_lc.marker.update(color="blue")
                transit_lc.name = "lc data"
                fig.add_trace(transit_lc, row=r, col=c)

                transit_sc = go.Scatter(x=photometry_data_sc.TIME, y=photometry_data_sc.FLUX, mode='markers',showlegend=False)
                transit_sc.marker.update(color="gray")
                transit_sc.name="sc data"
                fig.add_trace(transit_sc,row=r,col=c)

                

                lc_min,lc_max,sc_min,sc_max = data_load.get_min_max(koi_id)
                if len(photometry_data_sc)>0:
                    ### transit model
                    scit = 1.15e-5
                    t = np.arange(photometry_data_sc.TIME.min(), photometry_data_sc.TIME.max(),scit)
                    m = batman.TransitModel(theta, t-center_time)    #initializes model
                    flux = m.light_curve(theta)          #calculates light curve
                    mod = go.Scatter(x=t, y=flux, mode="lines", line=dict(color='red'),showlegend=False)
                    mod.name='Model'
                    fig.add_trace(mod,row=r,col=c)
                    if r==3:
                        fig.update_xaxes(title_text="TIME (DAYS)", row=r, col=c)
                    if c==1:
                        fig.update_yaxes(title_text="FLUX", row=r, col=c, range=[sc_min, sc_max]) 

                    ### quarter 
                    quarter = photometry_data_sc.loc[photometry_data_sc['TIME'] == photometry_data_sc.TIME.min(), 'QUARTER']
                    
                else:
                    ### transit model
                    scit = 1.15e-5
                    t = np.arange(photometry_data_lc.TIME.min(), photometry_data_lc.TIME.max(),scit)
                    m = batman.TransitModel(theta, t-center_time)    #initializes model
                    flux = m.light_curve(theta)          #calculates light curve
                    mod = go.Scatter(x=t, y=flux, mode="lines", line=dict(color='red'),showlegend=False)
                    mod.name='Model'
                    fig.add_trace(mod,row=r,col=c)
                    
                    ### quarter 
                    quarter = photometry_data_lc.loc[photometry_data_lc['TIME'] == photometry_data_lc.TIME.min(), 'QUARTER']
                    

                    if r==3:
                        fig.update_xaxes(title_text="TIME (DAYS)", row=r, col=c)
                    if c==1:
                        fig.update_yaxes(title_text="FLUX", row=r, col=c, range=[lc_min, lc_max])


            elif len(photometry_data_lc.TIME) > 1 and (photometry_data_sc) < 1:
                transit_lc = go.Scatter(x=photometry_data_lc.TIME, y=photometry_data_lc.FLUX, mode='markers',showlegend=False)
                transit_lc.marker.update(color="blue")
                fig.add_trace(transit_lc, row=r, col=c)
                lc_min,lc_max = data_load.get_min_max(koi_id)
                ### transit model
                scit = 1.15e-5 
                t = np.arange(photometry_data_lc.TIME.min(), photometry_data_lc.TIME.max(),scit)
                m = batman.TransitModel(theta, t-center_time)    #initializes model
                flux = m.light_curve(theta)          #calculates light curve
                mod = go.Scatter(x=t, y=flux, mode="lines", line=dict(color='red'),showlegend=False)
                fig.add_trace(mod,row=r,col=c)

                ### quarter 
                quarter = photometry_data_lc.loc[photometry_data_lc['TIME'] == photometry_data_lc.TIME.min(), 'QUARTER']
            
            elif len(photometry_data_lc.TIME) > 1 and (photometry_data_sc) == 2:
                
                annotation = go.layout.Annotation(
                                    x=1,  # Positioning on the far right
                                    y=1,  # Positioning on the top
                                    text=f'No more transits', 
                                    showarrow=False,  # No arrow needed
                                    font=dict(size=14, color="black"),  # Customize font size and color
                                    align='center',
                                    xanchor='center',  # Anchor the text to the right
                                    yanchor='middle'  # Anchor the text to the top
                                )
                fig.add_annotation(annotation, row=r, col=c) 

            elif os.path.isfile(file_path_sc) and not os.path.isfile(file_path_lc):
                transit_sc = go.Scatter(x=photometry_data_sc.TIME, y=photometry_data_sc.FLUX, mode='markers',showlegend=False)
                transit_sc.marker.update(color="blue")
                fig.add_trace(transit_sc,row=r,col=c)
                sc_min,sc_max = data_load.get_min_max(koi_id)
                ### transit model
                scit = 1.15e-5
                t = np.arange(photometry_data_sc.TIME.min(), photometry_data_sc.TIME.max(),scit)
                m = batman.TransitModel(theta, t-center_time)    #initializes model
                flux = m.light_curve(theta)          #calculates light curve
                mod = go.Scatter(x=t, y=flux, mode="lines", line=dict(color='red'),showlegend=False)
                fig.add_trace(mod,row=r,col=c)

                ### quarter 
                quarter = photometry_data_sc.loc[photometry_data_sc['TIME'] == photometry_data_sc.TIME.min(), 'QUARTER']
        else:
            error_message = f'No data found for {koi_id}'
            return jsonify(error_message=error_message)
        
    fig.update_layout(height=700, width=1000, title=title, title_x=0.5)
            
    graphJSON= json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder) 
    response_data = {
        'graphJSON': graphJSON,
        'transit_number': str(transit_number) 
    }
    return jsonify(response_data)

    
@app.route('/get_transit_file_options/<koi_id>')
def planet_options(koi_id):
    """
    Function gets planet numbers to auto-populate the dropdown for single transit plot planet options.
    
    args:
        koi_id: string in the form "K00000" (KOI identification)

    returns:
        jsonify(options): dictionary containing planet number and path value associated with it as JSON response
    """
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


@app.route('/generate_plot_folded_light_curve/<koi_id>')
def generate_plot_folded_light_curve(koi_id):
    """
    Function generates folded light curves and passes the figure to be displayed on html

    args:
        koi_id: string in the form "K00000" (KOI identification) 

    returns: 
        jsonify(graphJSON): contains figure able to be read and displayed in html
    """
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
    ### mask out overlapping transits
    overlap = []
    for i in range(npl):
        df_i = data_load.load_results_model(file_path_results,i)
        df_post_i = data_load.load_posteriors(file_path_results, i, star_id)
        df_post_i = df_post_i.sort_values(by="LN_LIKE", ascending = False)
        row_i = df_post_i.iloc[0]
        dur_i = row_i[f'DUR14_{i}']
        overlap_mask = np.zeros(len(df_i.ttime), dtype='bool')
        for j in range(npl):
            if i != j:
                df_j = data_load.load_results_model(file_path_results,j)
                df_post_j = data_load.load_posteriors(file_path_results, j, star_id)
                df_post_j = df_post_j.sort_values(by="LN_LIKE", ascending = False)
                row_j = df_post_j.iloc[0]
                dur_j = row_j[f'DUR14_{j}']
                for ttj in df_j.ttime:
                    overlap_mask += np.abs(df_i.ttime - ttj)/ (dur_i + dur_j) < 1.5
        overlap.append(overlap_mask)

    subplot_height=550
    data_id = data_load.get_koi_identifiers(csv_file_path, koi_id)
    data_id = data_id.sort_values(by='periods') 
    koi_identifiers = data_id.koi_identifiers.values
    periods = data_id.period_title.values
    subplot_titles = []
    spacing = [0.2,0.13,0.09,0.07,0.05,0.04,0.03] # spacing between plots for different numbers of planets
    for k in range(len(koi_identifiers)):
        subplot_titles.append(f'{koi_identifiers[k]}, {periods[k]}') 
        subplot_titles.append('')
        
    ### initialize figure
    rows = npl*2
    fig = make_subplots(rows=rows, cols=1,
                        #subplot_titles = subplot_titles,
                        row_heights=[subplot_height, subplot_height*0.4]*npl,
                        #vertical_spacing=spacing[npl-1]
                        ) 
    annotations = []

    

    # Create a dictionary to hold dynamic yaxis settings
    yaxis_dict = {}
    # calculate spacing between plots
    systems = rows/2
    space_per_system = 1/systems # system contains the main plot and the residual plot
    space_above = 0.25 * space_per_system
    space_between_systems = 0.05 * space_per_system
    plot_space = 0.60 * space_per_system 
    residual_space = 0.20 * space_per_system
    plot_domains = [[0,residual_space]]
    yaxis_dict[f'yaxis{1}'] = [0, plot_domains[0][1]]
    for i in range(0,rows-1):
        if i==rows-2:
            domain_ = plot_domains[i][1] + space_between_systems + plot_space
            dom = [plot_domains[i][1] + space_between_systems, 1]
        elif (i % 2 == 0):
            domain_ =  plot_domains[i][1] + space_between_systems + plot_space
            dom = [plot_domains[i][1] + space_between_systems, domain_]
        else:
            domain_ = space_above + plot_domains[i][1] + residual_space
            dom = [plot_domains[i][1] + space_above, domain_]
        
        plot_domains.append(dom)
        # Store the domain in the yaxis_dict dynamically
        yaxis_dict[f'yaxis{i+2}'] = dom
        
    r_plot = 1
    r_residuals = r_plot+1
    
    for i, file_path in enumerate(file_paths):
        planet_num = i
        period=periods[i]
        current_overlap = overlap[i]
        current_overlap = current_overlap.astype(np.bool_).astype(current_overlap.dtype.newbyteorder('='))
        fold_data_lc, fold_data_sc, binned_avg = data_load.folded_data(koi_id,planet_num,file_path,current_overlap)
        
        ### posteriors for most likely model
        data_post = data_load.load_posteriors(file_path_results,planet_num,koi_id)
        ### get max likelihood
        data_post = data_post.sort_values(by='LN_LIKE', ascending=False) 
        row = data_post.iloc[0] # pick row with highest likelihood
        ### get most likely params {P, t0, Rp/Rs, b, T14, q1, q2}
        theta = batman.TransitParams()
        theta.per = row[f'P']
        theta.t0 = 0.
        theta.rp = row[f'ROR_{planet_num}']
        theta.b = row[f'IMPACT_{planet_num}']
        theta.T14 = row[f'DUR14_{planet_num}']#*24
        LD_U1 = row[f'LD_U1']
        LD_U2 = row[f'LD_U2']
        theta.u = [LD_U1, LD_U2]
        theta.limb_dark = 'quadratic'
        DUR14 = row[f'DUR14_{planet_num}']

        other_models = []
        num_models = 20

        ### Select 20 random samples from the posterior distribution
        random_indices = random.sample(range(len(data_post)), num_models)
        random_samples = data_post.iloc[random_indices]
        
        pink_transparent = "rgba(255, 105, 180, 0.5)"  # Pink color with 50% transparency

        all_residuals = []
        all_data = []
        if os.path.exists(file_path_lc) and os.path.exists(file_path_sc):
            scit = 1.15e-5
            t = np.arange(fold_data_lc.TIME.min(), fold_data_lc.TIME.max(),scit)
            m = batman.TransitModel(theta, t)    #initializes model
            flux = (m.light_curve(theta))        #calculates light curve

            t_lc = np.array(fold_data_lc.TIME)
            f_lc = np.array(fold_data_lc.FLUX)
            m_lc = batman.TransitModel(theta, t_lc, supersample_factor = 7, exp_time = 0.02)    #initializes model for lc times
            flux_m_lc = (m_lc.light_curve(theta))
            t_sc = np.array(fold_data_sc.TIME)
            f_sc = np.array(fold_data_sc.FLUX)
            m_sc = batman.TransitModel(theta, t_sc, supersample_factor = 1, exp_time = 0.0007)    #initializes model for sc times
            flux_m_sc = (m_sc.light_curve(theta))

            # Sort t_lc and flux_m_lc based on t_lc
            # combined_fold_data = pd.concat([fold_data_lc, fold_data_sc], ignore_index=True)
            # combined_fold_data = combined_fold_data.sort_values(by='TIME', ascending=True)
            sorted_indices = np.argsort(t_lc)
            t_lc_sorted = t_lc[sorted_indices]
            f_lc_sorted = f_lc[sorted_indices]
            flux_m_lc_sorted = flux_m_lc[sorted_indices]
            sorted_indices_sc = np.argsort(t_sc)
            t_sc_sorted = t_sc[sorted_indices_sc]
            f_sc_sorted = f_sc[sorted_indices_sc]
            flux_m_sc_sorted = flux_m_sc[sorted_indices_sc]

            ### Compute residuals
            residuals_lc = fold_data_lc.FLUX - flux_m_lc
            residuals_sc = fold_data_sc.FLUX - flux_m_sc
            combined_residuals = pd.concat([residuals_lc, residuals_sc], ignore_index=True)
            combined_residuals = combined_residuals.sort_values(by='TIME', ascending=True)
            fold_time = np.array(combined_residuals.TIME)
            fold_flux = np.array(combined_residuals.FLUX)
            residuals_bin = data_load.bin_data(fold_time,fold_flux, DUR14/14)
            #residuals_bin = binned_avg.FLUX - flux_m_bin

            # Collect all residuals
            all_residuals.extend(residuals_lc)
            all_residuals.extend(residuals_sc)
            all_residuals.extend(residuals_bin)
            ### collect all data
            all_data.extend(fold_data_lc.FLUX)
            all_data.extend(fold_data_sc.FLUX)
            all_data.extend(binned_avg.FLUX)

            N_samp = 1000
            inds = np.arange(len(t_lc_sorted), dtype='int')
            inds = np.random.choice(inds, size=np.min([N_samp,len(inds)]), replace=False)
            inds_sc = np.arange(len(t_sc_sorted), dtype='int')
            inds_sc = np.random.choice(inds_sc, size=np.min([N_samp,len(inds_sc)]), replace=False)

            if i==0:
                ### short cadence
                fold_sc = go.Scatter(x=t_sc_sorted[inds_sc]*24, y=f_sc_sorted[inds_sc], mode='markers')
                fold_sc.marker.update(symbol="circle", size=4, color="gray")
                fold_sc.name = "Short Cadence"
                fold_sc.legendgroup=f'{i}'
                fig.add_trace(fold_sc, row=r_plot, col=1)
                ### long cadence
                fold_lc = go.Scatter(x=t_lc_sorted[inds]*24, y=f_lc_sorted[inds], mode='markers')
                fold_lc.marker.update(symbol="circle", size=5, color="blue")
                fold_lc.name = "Long Cadence"
                fold_lc.legendgroup=f'{i}'
                fig.add_trace(fold_lc, row=r_plot, col=1)
                ### binned avg
                bin_avg = go.Scatter(x=binned_avg.TIME*24, y=binned_avg.FLUX, mode='markers')
                bin_avg.marker.update(symbol="square", size=10, color="orange")
                bin_avg.name = "Binned Average"
                bin_avg.legendgroup=f'{i}'
                fig.add_trace(bin_avg, row=r_plot, col=1) 

                ### model
                mod = go.Scatter(x=t*24, y=flux, mode="lines", line=dict(color='red'))
                mod.name = "Model"
                mod.legendgroup=f'{i}'
                fig.add_trace(mod, row=r_plot, col=1)
                ### LC model
                mod_lc = go.Scatter(x=t_lc_sorted*24, y=flux_m_lc_sorted, mode="lines", line=dict(color='green'))
                mod_lc.name = "LC Model"
                fig.add_trace(mod_lc, row=r_plot, col=1)
            else:
                ### short cadence
                fold_sc = go.Scatter(x=t_sc_sorted[inds_sc]*24, y=f_sc_sorted[inds_sc], mode='markers', shlowlegend=False)
                fold_sc.marker.update(symbol="circle", size=4, color="gray")
                fold_sc.name = "Short Cadence"
                fig.add_trace(fold_sc, row=r_plot, col=1)
                ### long cadence
                fold_lc = go.Scatter(x=t_lc_sorted[inds]*24, y=f_lc_sorted[inds], mode='markers', shlowlegend=False)
                fold_lc.marker.update(symbol="circle", size=5, color="blue")
                fold_lc.name = "Long Cadence"
                fig.add_trace(fold_lc, row=r_plot, col=1)
                ### binned avg
                bin_avg = go.Scatter(x=binned_avg.TIME*24, y=binned_avg.FLUX, mode='markers', shlowlegend=False)
                bin_avg.marker.update(symbol="square", size=10, color="orange")
                bin_avg.name = "Binned Average"
                fig.add_trace(bin_avg, row=r_plot, col=1) 

                ### model
                mod = go.Scatter(x=t*24, y=flux, mode="lines", line=dict(color='red'), shlowlegend=False)
                mod.name = "Inst Model"
                fig.add_trace(mod, row=r_plot, col=1)
                ### LC model
                mod_lc = go.Scatter(x=t_lc_sorted*24, y=flux_m_lc_sorted, mode="lines", line=dict(color='green'), shlowlegend=False)
                mod_lc.name = "LC Model"
                fig.add_trace(mod_lc, row=r_plot, col=1)

            # for j, row_ in random_samples.iterrows():
            #     #row_ = data_post.iloc[j] # pick row with highest likelihood
            #     ### get random params {P, t0, Rp/Rs, b, T14, q1, q2}
            #     theta_ = batman.TransitParams()
            #     theta_.per = row_[f'P']
            #     theta_.t0 = 0.
            #     theta_.rp = row_[f'ROR_{planet_num}']
            #     theta_.b = row_[f'IMPACT_{planet_num}']
            #     theta_.T14 = row_[f'DUR14_{planet_num}']
            #     LD_U1 = row_[f'LD_U1']
            #     LD_U2 = row_[f'LD_U2']
            #     theta_.u = [LD_U1, LD_U2]
            #     theta_.limb_dark = 'quadratic'
                
            #     m_ = batman.TransitModel(theta_, t)    #initializes model
            #     flux_ = (m_.light_curve(theta_))        #calculates light curve
            #     mod_ = go.Scatter(x=t*24, y=flux_, mode="lines", showlegend=False, line=dict(color=pink_transparent))
            #     mod_.name = f'Model {j}'
                
            #     fig.add_trace(mod_, row=i+1, col=1)

            residuals_plot_lc = go.Scatter(x=t_lc_sorted[inds]*24, y=residuals_lc[inds], mode='markers', showlegend=False)
            residuals_plot_lc.marker.update(symbol="circle", size=5, color="blue")
            fig.add_trace(residuals_plot_lc, row=r_residuals, col=1)

            residuals_plot_sc = go.Scatter(x=t_sc_sorted[inds_sc]*24, y=residuals_sc[inds_sc], mode='markers', showlegend=False)
            residuals_plot_sc.marker.update(symbol="circle", size=4, color="gray")
            fig.add_trace(residuals_plot_sc, row=r_residuals, col=1)

            residuals_plot_bin = go.Scatter(x=binned_avg.TIME*24, y=residuals_bin, mode='markers', showlegend=False)
            residuals_plot_bin.marker.update(symbol="square", size=10, color="orange")
            fig.add_trace(residuals_plot_bin, row=r_residuals, col=1)

            # Add horizontal line at 0 in residual plot
            t = t*24
            fig.add_shape(type="line", x0=(t.min()), x1=(-1*t.min()), y0=0, y1=0,
                          line=dict(color="Red"), row=r_residuals, col=1)

            ### set plotting range for folded lc
            data_min = np.percentile(all_data, 5)
            data_max = np.percentile(all_data, 95)
            fig.update_yaxes(range=[data_min,data_max], row= r_plot, col=1)
            
            residuals_min = np.percentile(all_residuals, 20)
            residuals_max = np.percentile(all_residuals, 80)
            max_abs_residual = max(abs(residuals_min), abs(residuals_max))
            fig.update_yaxes(range=[-max_abs_residual,max_abs_residual], row= r_residuals, col=1)
            
            fig.update_yaxes(title_text="FLUX", row=r_plot, col=1)
            fig.update_xaxes(title_text="TIME (HOURS)", row=r_plot, col=1)
            fig.update_yaxes(title_text="Residuals", row=r_residuals, col=1)
            fig.update_layout(height=700, width=1000)
            r_plot = r_residuals+1
            r_residuals = r_plot+1
            
        
        elif os.path.exists(file_path_lc) and not os.path.exists(file_path_sc):
            if len(fold_data_lc.TIME) < 1:
                non_overlapping_transit = False
            else:
                non_overlapping_transit=True

            if non_overlapping_transit==True:
                t_lc = np.array(fold_data_lc.TIME)
                f_lc = np.array(fold_data_lc.FLUX)
                m_lc = batman.TransitModel(theta, t_lc, supersample_factor = 29, exp_time = 0.02)    #initializes model for lc times ##############################################################################
                flux_m_lc = (m_lc.light_curve(theta))

                ### Sort t_lc and flux_m_lc based on t_lc
                sorted_indices = np.argsort(t_lc)
                t_lc_sorted = t_lc[sorted_indices]
                flux_m_lc_sorted = flux_m_lc[sorted_indices]
                
                ### Compute residuals
                residuals_lc = fold_data_lc.FLUX - flux_m_lc
                residuals_lc = np.array(residuals_lc)
                fold_time = np.array(fold_data_lc.TIME)
                bin_centers, residuals_bin = data_load.bin_data(fold_time,residuals_lc, DUR14/11)
            
                ### Collect all residuals
                all_residuals.extend(residuals_lc) 
                all_residuals.extend(residuals_bin)

                ### thinning data to 1000 points
                N_samp = 1000
                inds = np.arange(len(t_lc_sorted), dtype='int')
                inds = np.random.choice(inds, size=np.min([N_samp,len(inds)]), replace=False)

                
                if i ==0:
                    ### long cadence
                    fold_lc = go.Scatter(x=t_lc[inds]*24, y=f_lc[inds], mode='markers')
                    fold_lc.marker.update(symbol="circle", size=5, color="blue")
                    fold_lc.name = "Long Cadence"
                    fold_lc.legendgroup=f'{i}'
                    fig.add_trace(fold_lc, row=r_plot, col=1)
                    ### binned avg
                    bin_avg = go.Scatter(x=binned_avg.TIME*24, y=binned_avg.FLUX, mode='markers')
                    bin_avg.marker.update(symbol="square", size=10, color="orange")
                    bin_avg.name = "Binned Average"
                    bin_avg.legendgroup=f'{i}'
                    fig.add_trace(bin_avg, row=r_plot, col=1)
                    ### model
                    mod_lc = go.Scatter(x=t_lc_sorted*24, y=flux_m_lc_sorted, mode="lines", line=dict(color='red')) 
                    mod_lc.name = "LC Model"
                    fig.add_trace(mod_lc, row=r_plot, col=1)
                else:
                    ### long cadence
                    fold_lc = go.Scatter(x=t_lc[inds]*24, y=f_lc[inds], mode='markers',showlegend=False)
                    fold_lc.marker.update(symbol="circle", size=5, color="blue")
                    fold_lc.name = "Long Cadence"
                    fig.add_trace(fold_lc, row=r_plot, col=1)
                    ### binned avg
                    bin_avg = go.Scatter(x=binned_avg.TIME*24, y=binned_avg.FLUX, mode='markers',showlegend=False)
                    bin_avg.marker.update(symbol="square", size=10, color="orange")
                    bin_avg.name = "Binned Average"
                    fig.add_trace(bin_avg, row=r_plot, col=1)
                    ### model
                    mod_lc = go.Scatter(x=t_lc_sorted*24, y=flux_m_lc_sorted, mode="lines", line=dict(color='red'), showlegend=False)
                    mod_lc.name = "LC Model" 
                    fig.add_trace(mod_lc, row=r_plot, col=1)

                ### plot 20 random model draws
                for j, row_ in random_samples.iterrows():
                    ### get random params {P, t0, Rp/Rs, b, T14, q1, q2}
                    theta_ = batman.TransitParams()
                    theta_.per = row_[f'P']
                    theta_.t0 = 0.
                    theta_.rp = row_[f'ROR_{planet_num}']
                    theta_.b = row_[f'IMPACT_{planet_num}']
                    theta_.T14 = row_[f'DUR14_{planet_num}']
                    LD_U1 = row_[f'LD_U1']
                    LD_U2 = row_[f'LD_U2']
                    theta_.u = [LD_U1, LD_U2]
                    theta_.limb_dark = 'quadratic'
                    
                    m_ = batman.TransitModel(theta_, t_lc_sorted, supersample_factor = 29, exp_time = 0.02)    #initializes model
                    flux_ = (m_.light_curve(theta_))        #calculates light curve
                    mod_ = go.Scatter(x=t_lc_sorted*24, y=flux_, mode="lines", showlegend=False, line=dict(color=pink_transparent))
                    
                    fig.add_trace(mod_, row=r_plot, col=1)

                ### plot residuals
                residuals_plot_lc = go.Scatter(x=t_lc[inds]*24, y=residuals_lc[inds], mode='markers', showlegend=False)
                residuals_plot_lc.marker.update(symbol="circle", size=5, color="blue")
                fig.add_trace(residuals_plot_lc, row=r_residuals, col=1)

                residuals_plot_bin = go.Scatter(x=bin_centers*24, y=residuals_bin, mode='markers', showlegend=False)
                residuals_plot_bin.marker.update(symbol="square", size=10, color="orange")
                fig.add_trace(residuals_plot_bin, row=r_residuals, col=1)

                ### Add horizontal line at 0 in residual plot
                fig.add_shape(type="line", x0=fold_data_lc.TIME.min()*24, x1=fold_data_lc.TIME.max()*24, y0=0, y1=0,
                            line=dict(color="Red"), row= r_residuals, col=1)

                ### plot range
                residuals_min = np.percentile(all_residuals, 99) 
                max_abs_residual =(abs(residuals_min))  
                fig.update_yaxes(range=[-max_abs_residual,max_abs_residual], row=r_residuals, col=1)

                ### Update x-axis and y-axis labels for each subplot
                fig.update_yaxes(title_text="FLUX", row=r_plot, col=1)
                fig.update_xaxes(title_text="TIME (HOURS)", row=r_residuals, col=1)
                fig.update_yaxes(title_text="Residuals", row=r_residuals, col=1)
                fig.update_layout(height=700, width=1000)

                ### update rows
                r_plot = r_residuals + 1
                r_residuals = r_plot+1
            else:
                ### annotation stating there are no non-overlapping transits
                annotation = go.layout.Annotation(
                                    x=1,  # Positioning on the far right
                                    y=1,  # Positioning on the top
                                    text=f'No non-overlapping transit for {koi_identifiers[i]}', 
                                    showarrow=False,  # No arrow needed
                                    font=dict(size=14, color="black"),  # Customize font size and color
                                    align='center',
                                    xanchor='center',  # Anchor the text to the right
                                    yanchor='middle'  # Anchor the text to the top
                                )
                fig.add_annotation(annotation, row=r_plot, col=1)
                fig.add_annotation(annotation, row=r_residuals, col=1) 
                r_plot = r_residuals + 1
                r_residuals = r_plot+1
            

        elif not os.path.exists(file_path_lc) and os.path.exists(file_path_sc):
            if i ==0:
                fold_sc = go.Scatter(x=fold_data_sc.TIME, y=fold_data_sc.FLUX, mode='markers')
                fold_sc.marker.update(symbol="circle", size=4, color="gray")
                fold_sc.name = "Short Cadence"
                fig.add_trace(fold_sc, row=r_plot, col=1)
                ### binned avg
                bin_avg = go.Scatter(x=binned_avg.TIME, y=binned_avg.FLUX, mode='markers')
                bin_avg.marker.update(symbol="square", size=10, color="orange")
                bin_avg.name = "Binned Average"
                fig.add_trace(bin_avg, row=r_plot, col=1)
                ### model
                scit = 1.15e-5
                t = np.arange(fold_data_sc.TIME.min(), fold_data_sc.TIME.max(),scit)
                m = batman.TransitModel(theta, t)    #initializes model
                flux = (m.light_curve(theta))        #calculates light curve
                mod = go.Scatter(x=t, y=flux, mode="lines", line=dict(color='red'))
                mod.line.update(color="red")
                mod.name = "Model"
                mod.legendgroup=f'{i}'
                fig.add_trace(mod, row=r_plot, col=1)
            else:
                fold_sc = go.Scatter(x=fold_data_sc.TIME, y=fold_data_sc.FLUX, mode='markers',showlegend=False)
                fold_sc.marker.update(symbol="circle", size=4, color="gray")
                fold_sc.name = "Short Cadence"
                fig.add_trace(fold_sc, row=r_plot, col=1)
                ### binned avg
                bin_avg = go.Scatter(x=binned_avg.TIME, y=binned_avg.FLUX, mode='markers',showlegend=False)
                bin_avg.marker.update(symbol="square", size=10, color="orange")
                bin_avg.name = "Binned Average"
                fig.add_trace(bin_avg, row=r_plot, col=1)
                ### model
                scit = 1.15e-5
                t = np.arange(fold_data_sc.TIME.min(), fold_data_sc.TIME.max(),scit)
                m = batman.TransitModel(theta, t)    #initializes model
                flux = (m.light_curve(theta))        #calculates light curve
                mod = go.Scatter(x=t, y=flux, mode="lines", line=dict(color='red'),showlegend=False)
                mod.line.update(color="red")
                mod.name = "Model"
                mod.legendgroup=f'{i}'
                fig.add_trace(mod, row=r_plot, col=1)

            t_sc = np.array(fold_data_sc.TIME)
            m_sc = batman.TransitModel(theta, t_sc, supersample_factor = 1, exp_time = 0.0007)    #initializes model for sc times
            flux_m_sc = (m_sc.light_curve(theta))
            t_bin = np.array(binned_avg.TIME)
            m_bin = batman.TransitModel(theta, t_bin, supersample_factor = 7, exp_time = DUR14/14)    #initializes model for bin times
            flux_m_bin = (m_bin.light_curve(theta))

            ### Compute residuals
            residuals_sc = fold_data_sc.FLUX - flux_m_sc
            fold_time = np.array(fold_data_sc.TIME)
            fold_flux = np.array(residuals_sc)
            residuals_bin = data_load.bin_data(fold_time,fold_flux, DUR14/14)
            #residuals_bin = binned_avg.FLUX - flux_m_bin

            # Collect all residuals
            all_residuals.extend(residuals_sc)
            all_residuals.extend(residuals_bin)

            ### plot residuals
            residuals_plot_sc = go.Scatter(x=fold_data_sc.TIME, y=residuals_sc, mode='markers', showlegend=False)
            residuals_plot_sc.marker.update(symbol="circle", size=4, color="gray")
            fig.add_trace(residuals_plot_sc, row=r_residuals, col=1)

            residuals_plot_bin = go.Scatter(x=binned_avg.TIME, y=residuals_bin, mode='markers', showlegend=False)
            residuals_plot_bin.marker.update(symbol="square", size=10, color="orange")
            fig.add_trace(residuals_plot_bin, row=r_residuals, col=1)

            # Add horizontal line at 0 in residual plot
            fig.add_shape(type="line", x0=fold_data_lc.TIME.min(), x1=fold_data_lc.TIME.max(), y0=0, y1=0,
                          line=dict(color="Red"), row=r_residuals, col=1)

            residuals_min = np.percentile(all_residuals, 20)
            residuals_max = np.percentile(all_residuals, 80)
            max_abs_residual = max(abs(residuals_min), abs(residuals_max))
            fig.update_yaxes(range=[-max_abs_residual,max_abs_residual], row=r_residuals, col=1)

            ### Update x-axis and y-axis labels for each subplot
            fig.update_yaxes(title_text="FLUX", row=r_plot, col=1)
            fig.update_xaxes(title_text="TIME (HOURS)", row=r_residuals, col=1)
            fig.update_yaxes(title_text="Residuals", row=r_residuals, col=1)
            fig.update_layout(height=700, width=1000) 

            ### update rows
            r_plot = r_residuals+1
            r_residuals = r_plot+1
        
        else:
            error_message = f'No data found for {koi_id}'
            return jsonify(error_message=error_message)
        
    ### return whole fig to page
    if npl>1:
        fig.update_layout(height=npl * subplot_height,legend_tracegroupgap = 240)
    # Loop to dynamically update layout for each subplot
    for i in range(1, rows+1):
        fig.update_layout(
            **{
                f'xaxis{i}': dict(domain=[0, 1]),  # Horizontal span: 0 to 1
                f'yaxis{i}': dict(domain=yaxis_dict[f'yaxis{rows+1-i}'])  # Use dynamic yaxis from dictionary
            }
        )
        if not (i %2 ==0):
            fig.update_xaxes(showticklabels=False, row=i, col=1)
    
    # Add annotations (titles) above each main plot
    annotations = []
    for i in range(1, rows+1, 2):  # Loop through odd rows (main plots only)
        y_position = yaxis_dict[f'yaxis{rows+1-i}'][1] + 0.25*residual_space  # Get the top boundary of the yaxis and add offset
        annotations.append(dict(
            x=0.5, y=y_position,  # Centered horizontally
            xref="paper", yref="paper",
            text=subplot_titles[i-1],  # Use corresponding title
            showarrow=False,
            font=dict(size=18)
        ))

    fig.update_traces(showlegend=True, row=1, col=1)
    # Update layout to include annotations
    fig.update_layout(annotations=annotations)
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder) 
    return jsonify(graphJSON)
    

@app.route('/generate_plot_OMC/<koi_id>')
def generate_plot_OMC(koi_id):
    """
    Function generates Observed Minus Calculated (OMC) plots and passes the figure to be displayed on html

    args:
        koi_id: string in the form "K00000" (KOI identification) 

    returns: 
        jsonify(graphJSON): contains figure able to be read and displayed in html
    """
    global K_id
    if K_id==False:
        star_id = koi_id.replace("K","S")
    else:
        star_id = koi_id
    file_name = star_id + '_*_quick.ttvs'
    file_paths = (glob.glob(os.path.join(data_directory,star_id, file_name)))   
    file_path_sort = sorted(file_paths, key=lambda  x: int(re.search(r"_(\d+)_", x).group(1))) ### sort
    
    ext = os.path.basename(data_directory) +'.csv'
    csv_file_path = os.path.join(data_directory, ext)
    ### number of planets from number of ttv files
    npl = len(file_path_sort)
    data_id = data_load.get_koi_identifiers(csv_file_path, koi_id)
    data_id = data_id.sort_values(by='periods') 
    koi_identifiers = data_id.koi_identifiers.values
    periods = data_id.period_title.values
    subplot_height = 400
    subplot_titles = []
    spacing = [0.2,0.1,0.05,0.04,0.04,0.04,0.03] 
    ### create subplot titles
    for k in range(len(koi_identifiers)):
        subplot_titles.append(f'{koi_identifiers[k]}, {periods[k]}') 
        subplot_titles.append('here') # no title for residual plots
    
    ### initialize figure
    rows = npl*2
    fig = make_subplots(rows=rows, cols=1,
                        #subplot_titles=subplot_titles,
                        row_heights=[subplot_height, subplot_height*0.4]*npl,
                        #vertical_spacing=spacing[npl-1]
                        ) 
    
    # Create a dictionary to hold dynamic yaxis settings
    yaxis_dict = {}
    # calculate spacing between plots
    systems = rows/2
    space_per_system = 1/systems # system contains the main plot and the residual plot
    space_above = 0.25 * space_per_system
    space_between_systems = 0.05 * space_per_system
    plot_space = 0.60 * space_per_system 
    residual_space = 0.20 * space_per_system
    plot_domains = [[0,residual_space]]
    yaxis_dict[f'yaxis{1}'] = [0, plot_domains[0][1]]
    for i in range(0,rows-1):
        if i==rows-2:
            domain_ = plot_domains[i][1] + space_between_systems + plot_space
            dom = [plot_domains[i][1] + space_between_systems, 1]
        elif (i % 2 == 0):
            domain_ =  plot_domains[i][1] + space_between_systems + plot_space
            dom = [plot_domains[i][1] + space_between_systems, domain_]
        else:
            domain_ = space_above + plot_domains[i][1] + residual_space
            dom = [plot_domains[i][1] + space_above, domain_]
        
        plot_domains.append(dom)
        # Store the domain in the yaxis_dict dynamically
        yaxis_dict[f'yaxis{i+2}'] = dom

    
    

    ### set rows
    r_plot = 1
    r_residuals = r_plot+1
    
    ### create OMC plot for each planet
    for i, file_path in enumerate(file_path_sort):
        omc_data, omc_model, out_prob, out_flag = data_load.load_OMC_data(koi_id, file_path)
        show_outliers = True
        inds = np.arange(len(omc_data), dtype='int')

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
                fig.add_trace(omc, row=(r_plot), col=1)
                fig.add_trace(line_trace, row=(r_plot), col=1)

                ### Add scatter trace for outliers with 'x' shape markers
                scatter_outliers = px.scatter(omc_data[mask], x='TIME', y='OMC').update_traces(
                    marker=dict(symbol='x', color='orange'),
                    line=dict(width=0.7))

                fig.add_trace(scatter_outliers.data[0], row=(r_plot), col=1)
                
                ### calculate residuals
                residuals = omc_data.OMC - omc_model.OMC_MODEL[inds]
                omc_residuals = pd.DataFrame({
                    'TIME' : omc_data.TIME,
                    'RESIDUALS' : residuals
                })
                
                ### plot residuals 
                residuals_plot_omc = px.scatter(omc_residuals,x='TIME', y='RESIDUALS', color=out_prob,color_continuous_scale='viridis').data[0]
                fig.add_trace(residuals_plot_omc, row=r_residuals, col=1) 
                
                ### Add horizontal line at 0 in residual plot 
                fig.add_shape(type="line", x0=omc_data.TIME.min(), x1=omc_data.TIME.max(), y0=0, y1=0,
                            line=dict(color="Red"), row= r_residuals, col=1)
                
                ### Add scatter trace for outliers with 'x' shape markers
                scatter_outliers = px.scatter(omc_residuals[mask], x='TIME', y='RESIDUALS').update_traces(
                    marker=dict(symbol='x', color='orange'), 
                    line=dict(width=0.7))
                fig.add_trace(scatter_outliers.data[0], row=(r_residuals), col=1)

                ### Update x-axis and y-axis labels for each subplot
                fig.update_xaxes(title_text="TIME (DAYS)", row=r_residuals, col=1)
                fig.update_yaxes(title_text="O-C (MINUTES)", row=r_plot, col=1)
                fig.update_yaxes(title_text="Residuals", row=r_residuals, col=1)
                fig.update_coloraxes(colorbar_title_text='Out Probability', colorbar_len=0.2)


                ### update rows
                r_plot = r_residuals + 1
                r_residuals = r_plot+1

            else:
                mask_arr = np.array(mask)
                omc = px.scatter(omc_data[~mask_arr], x="TIME",y="OMC") 
                ### Add a line plot for OMC_MODEL
                line_trace = px.line(omc_model[~mask_arr], x="TIME", y="OMC_MODEL").data[0]
                line_trace.line.color = 'red' 
                fig.add_trace(omc, row=(r_plot), col=1)
                fig.add_trace(line_trace, row=(r_plot), col=1)
                ### update axes and colorbar
                fig.update_xaxes(title_text="TIME (DAYS)", row=r_plot, col=1)
                fig.update_yaxes(title_text="O-C (MINUTES)", row=r_plot, col=1)
                fig.update_coloraxes(colorbar_title_text='Out Probability', colorbar_len=0.2)#, row=i+1, col=1)

                ### calculate residuals
                residuals = omc_model[~mask_arr].OMC_MODEL[inds] - omc_data[~mask_arr].OMC
                ### update rows
                r_plot = r_residuals + 1
                r_residuals = r_plot+1
        
        else: 
            error_message = f'No data found for {koi_id}'
            return jsonify(error_message=error_message)
    
    ### adjust colorbar
    colorbar_spacing = [1,0.5,0.33,0.25,0.2,0.15,0.1]
    fig.update_layout(
            coloraxis=dict(
                colorbar=dict(
                    title="Out Probability", 
                    len=colorbar_spacing[npl-1],  # Adjust the length of the colorbar
                    orientation='v',  # Vertical orientation
                    x=1.05,  # Place it to the right of the plot
                    y=1,  # Start at the top
                    yanchor='top'
                )
            )
    )
    fig_height = 450 * 1.4
    fig.update_layout(height=fig_height*npl, width=1000)
    # Loop to dynamically update layout for each subplot
    for i in range(1, rows+1):
        fig.update_layout(
            **{
                f'xaxis{i}': dict(domain=[0, 1]),  # Horizontal span: 0 to 1
                f'yaxis{i}': dict(domain=yaxis_dict[f'yaxis{rows+1-i}'])  # Use dynamic yaxis from dictionary
            }
        )
        if not (i %2 ==0):
            fig.update_xaxes(showticklabels=False, row=i, col=1)
    
    # Add annotations (titles) above each main plot
    annotations = []
    for i in range(1, rows+1, 2):  # Loop through odd rows (main plots only)
        y_position = yaxis_dict[f'yaxis{rows+1-i}'][1] + 0.25*residual_space  # Get the top boundary of the yaxis and add offset
        annotations.append(dict(
            x=0.5, y=y_position,  # Centered horizontally
            xref="paper", yref="paper",
            text=subplot_titles[i-1],  # Use corresponding title
            showarrow=False,
            font=dict(size=18)
        ))

    # Update layout to include annotations
    fig.update_layout(annotations=annotations)
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder) 
    return jsonify(graphJSON)


@app.route('/generate_plot_corner/<koi_id>/<selected_columns>/<planet_num>')
def generate_plot_corner(koi_id,selected_columns, planet_num):
    """
    Function generates posterior corner plots and passes the figure to be displayed on html

    args:
        koi_id: string in the form "K00000" (KOI identification) 
        selected_columns: list of strings containing names of variables selected on app
        planet_num: integer planet number

    returns: 
        jsonify(graphJSON): contains figure able to be read and displayed in html
    """
    selected_columns = selected_columns.split(',')
    global K_id
    if K_id==False:
        star_id = koi_id.replace("K","S")
    else:
        star_id = koi_id
    file =star_id + '-results.fits'
    file_path = os.path.join(data_directory, star_id, file)

    ext = os.path.basename(data_directory) +'.csv'
    csv_file_path = os.path.join(data_directory, ext)

    data_id = data_load.get_koi_identifiers(csv_file_path, koi_id)
    data_id = data_id.sort_values(by='periods') 

    if os.path.isfile(file_path):
        data = data_load.load_posteriors(file_path,planet_num,koi_id)
        ### handle weighting
        LN_WT = data['LN_WT'].values
        weight = np.exp(LN_WT- LN_WT.max())
        w = weight/ np.sum(weight)
        data['WEIGHTS'] = w

        labels = data[selected_columns].columns.tolist()
        ### initialize figure
        fig = make_subplots(rows=len(selected_columns), cols=len(selected_columns))
        
        ### orient corner plot
        for i in range(len(selected_columns)):
            for j in range(i, len(selected_columns)):
                
                x = data[selected_columns[i]].values
                y = data[selected_columns[j]].values
                

                if i != j:
                    if labels[j]==f'DUR14_{planet_num}':
                        x = x*24
                    ### scatter plot for non-diagonal plots
                    fig.add_trace(go.Scatter(
                        x=x, 
                        y=y,
                        mode='markers', 
                        marker=dict(color='gray', size=1), 
                        showlegend=False
                        ), row=j + 1, col=i + 1)
                    
                    
                    
                else:
                    if labels[j]==f'DUR14_{planet_num}':
                        x = x*24 
                    ### plot kde for diagonal plots
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
                        fig.add_trace(go.Scatter(x=[1, 1], y=[0, np.max(y_vals)], mode='lines', line=dict(color='black', dash='dash'),showlegend=False), row=j + 1, col=i + 1)

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
                    
                ### update axes, outline plots
                fig.update_layout(plot_bgcolor='#F7FBFF') # match background with the back contour color
                fig.update_xaxes(showline=True, linewidth=1, linecolor='black', mirror=True, row=j + 1, col=i + 1, tickangle=30)
                fig.update_yaxes(showline=True, linewidth=1, linecolor='black', mirror=True, row=j + 1, col=i + 1, tickangle=0)
        
        
        fig.update_layout(height=800, width=900)
        graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder) 
        return jsonify(graphJSON)
    else:
        error_message = f'No data found for {koi_id}'
        return jsonify(error_message=error_message)


@app.route('/get_transit_file_options_corner/<koi_id>')
def planet_options_corner(koi_id):
    """
    Function gets planet numbers to auto-populate the dropdown for posterior corner plot planet options.
    
    args:
        koi_id: string in the form "K00000" (KOI identification)

    returns:
        jsonify(options): dictionary containing planet number and path value associated with it as JSON response
    """
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


if __name__ == '__main__':
    app.run(debug=True)
