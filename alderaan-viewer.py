import os
import csv
import glob
import json
import random
from   datetime import datetime

import batman
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from   plotly.subplots import make_subplots
import plotly.utils
import re
from   scipy.stats import gaussian_kde

from flask import Flask, render_template, jsonify, request, session, redirect, url_for
import data_load
import utils

LCIT = 29.4243885           # Kepler long cadence integration time + readout time [min] 
SCIT = 58.848777            # Kepler short cadence integration time + readout time [sec]

lcit = LCIT/60/24           # Kepler long cadence integration time + readout time [days]
scit = SCIT/3600/24         # Kepler short cadence integration time + readout time [days]


K_id = True
app = Flask(__name__)
app.secret_key = 'super_secret'


### Dynamically determine the directory structure
### The viewer app should be placed in "<ROOT_DIR>/alderaan-viewer"
### Pipeline outputs should be stored in "<ROOT_DIR>/alderaan/Results/<RUN_ID>"
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VIEWER_DIR = os.path.join(ROOT_DIR, 'alderaan-viewer')
RESULTS_DIR = os.path.join(ROOT_DIR, 'alderaan', 'Results')

RUN_DIR = ''
table = ''


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
    global RESULTS_DIR
    global RUN_DIR
    
    ### List all items (files and folders) in the directory 
    options = sorted(os.listdir(RESULTS_DIR))
    hidden = []

    ### Remove hidden files from list
    for opt in options:
        if opt[0] == '.':
            hidden.append(opt)

    for item in hidden:
        options.remove(item)
        
    ### Default table option is first folder in directory
    table = request.args.get('table', options[0]+'.csv')
    
    ### Update data directory
    RUN_DIR = os.path.join(RESULTS_DIR, table[:-4])
    
    ### Switch to use K versus S (simulation data)
    if 'SIMULATION' in table:
        K_id = False 
    else: 
        K_id = True
    if not table.startswith('.'):
        table_data = data_load.read_star_properties_table(table)
    else:
        table_data = 'error'
    
    ### Render content
    left_content = render_template('left.html', table_data=table_data)
    right_top_content = render_template('right_top.html')
    right_bottom_content = render_template('right_bottom.html')
    
    return render_template('index.html', 
                           left_content=left_content, 
                           right_top_content=right_top_content,
                           right_bottom_content=right_bottom_content
                          )


#@app.route('/home/get_dropdown_options_for_star_tables')
#def get_dropdown_options_for_star_tables():
@app.route('/get_dropdown_options')
def get_dropdown_options():
    '''
    Function populates options based on folders in the default directory

    Returns:
        jsonify(options): list of options in json-readable format
    '''
    global RESULTS_DIR
    
    ### List all directories in <RESULTS_DIR>
    options = sorted([item for item in os.listdir(RESULTS_DIR) 
                      if os.path.isdir(os.path.join(RESULTS_DIR, item))])
    
    options_with_tables = []
    for opt in options: 
        if os.path.isfile(os.path.join(RESULTS_DIR, opt, opt + '.csv')):
            options_with_tables.append(opt)
            
    ### Return the list of options as a JSON response
    return jsonify(options_with_tables)
    
    
#@app.route('/home/get_selected_star_table', methods=['GET'])
#def get_selected_star_table():
@app.route('/get_selected_table', methods=['GET'])
def get_selected_table():
    """
    Function passes the selected table as JSON response to display
    """
    try:
        ### Return the selected table as a JSON response
        selected_table = {'selected_table': table}
        return jsonify(selected_table)
    except Exception as exception:
        return jsonify({'error': str(exception)}), 500


#@app.route('/home/color_star_table/')
#def color_star_table():
@app.route('/table_color/')
def table_color():
    """
    Function that retrieves review status for javascript function to assign colors to table based on review status value.
    
    returns:
        jsonify(review_data): passes review status and koi_id to html
    """
    global RUN_DIR
    global table
    file_path = os.path.join(RUN_DIR, table)
    
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


#@app.route('/<koi_id>/tables/planet_properties', methods=['GET'])
#def planet_properties_table(koi_id):
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
    planet_data = data_load.read_planet_properties_table(table,koi_id)
    return jsonify(planet_data)


#@app.route('/<koi_id>/tables/period_ratios', methods=['GET'])
#def period_ratios_table(koi_id):
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
    ratio_data = data_load.read_period_ratios_table(table,koi_id)
    return jsonify(ratio_data)


#@app.route('<koi_id>/review/status/', methods=['POST'])
#def review_status(koi_id):
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
    global RUN_DIR
    global table
    ### get review status from dropdown
    data = request.json
    review_status = data['reviewStatus']
    file_path = os.path.join(RUN_DIR, table)
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


#@app.route('/<koi_id>/review/display_comment', methods=['POST'])
#def display_comment(koi_id):
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
    file_path = os.path.join(RUN_DIR, path_extension)
    if os.path.isfile(file_path):
        with open(file_path, 'r') as file:
                file_content = file.read()
    else:
        file_content = f'Comment file for {koi_id} not found.'
    return file_content


#@app.route('/<koi_id>/review/save_comment', methods=['POST'])
#def save_comment(koi_id):
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
    file_path = os.path.join(RUN_DIR, path_extension)
    comment = request.form.get('comment').strip()
    username = session.get('username')
    with open(file_path, 'a') as file:
        file.write("\n")
        file.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        file.write(f"User: {username}\n")
        file.write(f"Comment: {comment}\n")
    return display_comment_file(koi_id)


#@app.route('/<koi_id>/review/edit_comment', methods=['POST'])
#def edit_comment(koi_id):
@app.route('/star/<koi_id>/edit_file', methods=['POST'])
def save_file(koi_id):
    """
    Function saves the comment file and displays the updated file or an error message.
    A comment file will be created at <RUN_DIR>/<koi_id>/<koi_id>_comments.txt

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
    file_path = os.path.join(RUN_DIR, path_extension)
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


##########################################################################################

#@app.route('/<koi_id>/plots/detrended_lightcurve')
#def plot_detrended_lightcurve(koi_id):
@app.route('/generate_plot/<koi_id>')
def plot_detrended_lightcurve(koi_id):
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
        
        
    file_path_lc = os.path.join(RUN_DIR, star_id, f"{star_id}_lc_filtered.fits")
    file_path_sc = os.path.join(RUN_DIR, star_id, f"{star_id}_sc_filtered.fits")
    file_path_results = os.path.join(RUN_DIR, star_id, f"{star_id}-results.fits")
    
    npl = data_load.get_num_planets(file_path_results)
    
    ### Initialize figure
    fig = make_subplots(rows=1, cols=1)

    if os.path.isfile(file_path_lc):
        data_lc = data_load.load_photometry_data(file_path_lc)
        
        ### Long cadence
        plot_lc = px.scatter(data_lc, x="TIME",y="FLUX").data[0]
        plot_lc.marker.update(symbol="circle", size=4, color="blue")
        plot_lc.name = "Long Cadence"
        fig.add_trace(plot_lc, row=1, col=1)


        ### Mark quarters for Long Cadence data
        for idx, quarter in enumerate(data_lc['QUARTER'].unique()):
            times = data_lc.loc[data_lc['QUARTER'] == quarter, 'TIME']
            start = times.min()
            end = times.max()
            
            line_ypos = max(data_lc['FLUX']) + 0.0001 * max(data_lc['FLUX'])
            line_color = ['green','red','blue','orange'][quarter % 4]

            ### Add horizontal lines at the top of the plot
            fig.add_shape(
                type="line",
                x0=start,
                x1=end,
                y0=line_ypos,
                y1=line_ypos,
                line=dict(color=line_color, width=4),
                name=f"Quarter {quarter}"
            )

            ### Add an invisible scatter trace for hover information
            hover_trace = go.Scatter(
                x=[(start + end) / 2],
                y=[line_ypos],
                mode='markers',
                marker=dict(opacity=0),
                showlegend=False,
                hoverinfo='text',
                text=f"Quarter {quarter}",
                hoverlabel=dict(bgcolor=line_color, font=dict(color='white'))
            )
            fig.add_trace(hover_trace)

        ### Mark individual transits for each planet
        colors = ['orange','green','red','orange','green','red','orange','green','red']

        for n in range(npl):
            data_ttv = data_load.load_ttv_data_from_results(file_path_results,n)
            
            y_min = data_lc.FLUX.min() - 1e-5
            y_off = 0.0001
            y_pts = y_min*np.ones(len(data_ttv.ttime)) + y_off
            
            plot_tts = px.scatter(x=data_ttv.ttime, y=y_pts).data[0]
            plot_tts.marker.update(symbol="circle", size=4, color=colors[n])
            plot_tts.name = f"Planet {n}"
            fig.add_trace(plot_tts, row=1, col=1)
            
        ### Update axis labels
        fig.update_traces(showlegend=True, row=1, col=1)
        fig.update_layout(xaxis_title=f"TIME (DAYS)", yaxis_title="FLUX")
        fig.update_layout(title=star_id, title_x=0.5)
        
        ### Encode plot with JSON
        plotDetrendedLightcurveJSON= json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        return jsonify(plotDetrendedLightcurveJSON)
    
    else:
        error_message = f'No data found for {koi_id}'
        return jsonify(error_message=error_message)



















@app.route('/generate_plot_folded_light_curve/<koi_id>')
def plot_folded_lightcurve(koi_id):
    """
    Function generates folded light curves and passes the figure to be displayed on html

    args:
        koi_id: string in the form "K00000" (KOI identification) 

    returns: 
        jsonify(graphJSON): contains figure able to be read and displayed in html
    """
    global RUN_DIR
    global K_id
    if K_id==False:
        star_id = koi_id.replace("K","S")
    else:
        star_id = koi_id
    
    ### Set file paths
    file_path_lc = os.path.join(RUN_DIR, star_id, f"{star_id}_lc_filtered.fits")
    file_path_sc = os.path.join(RUN_DIR, star_id, f"{star_id}_sc_filtered.fits")
    file_path_results = os.path.join(RUN_DIR, star_id, f"{star_id}-results.fits")
    
    file_name = star_id + '_*_quick.ttvs'
    file_paths = glob.glob(os.path.join(RUN_DIR,star_id, file_name))
    

    ### Read in the data
    npl = data_load.get_num_planets(file_path_results)
    data_ttv = [None]*npl
    data_samples = [None]*npl
    
    for n in range(npl):
        data_ttv[n] = data_load.load_ttv_data_from_results(file_path_results,n)
        data_samples[n] = data_load.load_posteriors_from_results(file_path_results,n)
    
    if os.path.isfile(file_path_lc):
        data_lc = data_load.load_photometry_data(file_path_lc)
        
    if os.path.isfile(file_path_sc):
        data_sc = data_load.load_photometry_data(file_path_sc)
        
    ### Mask out overlapping transits
    overlap = [None]*npl
    for i in range(npl):
        overlap[i] = np.zeros(len(data_ttv[i].ttime),dtype='bool')    
        for j in range(npl):
            if i != j:
                dur_i = np.median(data_samples[i][f'DUR14_{i}'])
                dur_j = np.median(data_samples[j][f'DUR14_{j}'])
                for ttj in data_ttv[j].ttime:
                    overlap[i] += np.abs(data_ttv[i].ttime-ttj) / (dur_i+dur_j) < 1.5

        
    ### Get data IDs
    csv_file_path = os.path.join(RUN_DIR, os.path.basename(RUN_DIR) +'.csv')
    data_id = data_load.get_koi_identifiers(csv_file_path, koi_id)
    data_id = data_id.sort_values(by='periods') 
    koi_identifiers = data_id.koi_identifiers.values
    periods = data_id.period_title.values
    
    
    
    ### Initialize figure
    subplot_height=550
    subplot_titles = []
    spacing = [0.2,0.13,0.09,0.07,0.05,0.04,0.03] # spacing between plots for different numbers of planets
    
    for k in range(len(koi_identifiers)):
        subplot_titles.append(f'{koi_identifiers[k]}, {periods[k]}') 
        subplot_titles.append('')
        
    fig = make_subplots(rows=npl*2, cols=1,
                        row_heights=[subplot_height, subplot_height*0.4]*npl,
                        ) 
    annotations = []

    
    ### Plot the data
    for n in range(npl):
        ### Maximum likelihood parameters {P, t0, Rp/Rs, b, T14, u1, u2}
        data_samples[n] = data_samples[n].sort_values(by='LN_LIKE', ascending=False) 
        max_like_sample = data_samples[n].iloc[0]

        theta = batman.TransitParams()
        theta.per = max_like_sample[f'P']
        theta.t0 = 0.
        theta.rp = max_like_sample[f'ROR_{n}']
        theta.b = max_like_sample[f'IMPACT_{n}']
        theta.T14 = max_like_sample[f'DUR14_{n}']
        theta.u = [max_like_sample[f'LD_U1'], max_like_sample[f'LD_U2']]
        theta.limb_dark = 'quadratic'
        
        duration = np.median(data_samples[n][f'DUR14_{n}'])


        t_fold, f_fold = utils.fold_data(data_lc.TIME, 
                                         data_lc.FLUX, 
                                         data_ttv[n].model[~overlap[n]],
                                         duration
                                        )
        
        t_bin, f_bin = utils.bin_data(t_fold, f_fold, duration/11)

        ### Select 20 random draws (plus max-likelihood) from the posterior distribution
        ndraw = 20
        random_inds = np.random.randint(1,len(data_samples[n]),ndraw)
        random_draws = data_samples[n].iloc[np.hstack([random_inds, 0])]
        
        if os.path.exists(file_path_lc) and not os.path.exists(file_path_sc):
            if np.sum(~overlap[n]) > 0:
                ### Compute residuals
                m_fold = batman.TransitModel(theta, t_fold, supersample_factor=59, exp_time=lcit)
                r_fold = f_fold - m_fold.light_curve(theta)
                t_bin, r_bin = utils.bin_data(t_fold, r_fold, duration/11)
            
                ### Calculate maximum likelihood model for plotting purposes
                t_mod = np.linspace(-1.5*duration,1.5*duration,50)
                m_mod = batman.TransitModel(theta, t_mod, supersample_factor=59, exp_time=lcit)
                f_mod = m_mod.light_curve(theta)
                
                ### Indexes to thin data to 1000 points
                npts = 1000
                inds = np.arange(len(f_fold), dtype='int')
                inds = np.random.choice(inds, size=np.min([npts,len(inds)]), replace=False)
                
                ### Plot objects
                p_fold = go.Scatter(x=t_fold[inds]*24, y=f_fold[inds], mode='markers')
                p_fold.marker.update(symbol='circle', size=5, color='blue')
                p_fold.name = "Long Cadence"
                p_fold.legendgroup=f'{n}'
                fig.add_trace(p_fold, row=2*n+1, col=1)

                p_bin = go.Scatter(x=t_bin*24, y=f_bin, mode='markers')
                p_bin.marker.update(symbol='square', size=10, color='orange')
                p_bin.name = "Long Cadence"
                p_bin.legendgroup=f'{n}'
                fig.add_trace(p_bin, row=2*n+1, col=1)
               
                ### Plot random model draws
                for j, draw in random_draws.iterrows():
                    theta.per = draw[f'P']
                    theta.t0 = 0.
                    theta.rp = draw[f'ROR_{n}']
                    theta.b = draw[f'IMPACT_{n}']
                    theta.T14 = draw[f'DUR14_{n}']
                    theta.u = [draw[f'LD_U1'], draw[f'LD_U2']]
                    theta.limb_dark = 'quadratic'
                    
                    f_mod = m_mod.light_curve(theta)
                    
                    color = "rgba(255, 105, 180, 0.5)"  # pink with 50% transparency
                    p_mod = go.Scatter(x=t_mod*24, 
                                       y=f_mod, 
                                       mode="lines",
                                       showlegend=False, 
                                       name="",
                                       line=dict(color=color)
                                       )
                                       
                    fig.add_trace(p_mod, row=2*n+1, col=1)
                    fig.update_layout()
                
                ### Plot residuals
                p_resf = go.Scatter(x=t_fold[inds]*24, y=r_fold[inds], mode='markers', showlegend=False)
                p_resf.marker.update(symbol="circle", size=5, color="blue")
                fig.add_trace(p_resf, row=2*n+2, col=1)

                p_resb = go.Scatter(x=t_bin*24, y=r_bin, mode='markers', showlegend=False)
                p_resb.marker.update(symbol="square", size=10, color="orange")
                fig.add_trace(p_resb, row=2*n+2, col=1)

                ### Add horizontal line at 0 in residual plot
                fig.add_shape(type="line", x0=t_fold.min()*24, x1=t_fold.max()*24, y0=0, y1=0,
                            line=dict(color="Red"), row=2*n+2, col=1)

                ### Update x-axis and y-axis labels for each subplot
                fig.update_yaxes(title_text="FLUX", row=2*n+1, col=1)
                fig.update_xaxes(title_text="TIME (HOURS)", row=2*n+2, col=1)
                fig.update_yaxes(title_text="RESIDUALS", row=2*n+2, col=1)
                fig.update_layout(height=700, width=1000)

            else:
                ### annotation stating there are no non-overlapping transits
                text = f'No non-overlapping transit for {koi_identifiers[i]}'
                annotation = go.layout.Annotation(x=1,
                                                  y=1,
                                                  text=text, 
                                                  showarrow=False,
                                                  ont=dict(size=14, color="black"),
                                                  align='center',
                                                  xanchor='center',
                                                  yanchor='middle'
                                                 )
                                                 
                fig.add_annotation(annotation, row=2*n+1, col=1)
            
        else:
            error_message = f'No data found for {koi_id}'
            return jsonify(error_message=error_message)
        
        
    ### Update Layout
    if npl>1:
        fig.update_layout(height=npl * subplot_height,legend_tracegroupgap = 240)

    ### Dynamically set y-axis spacing based on number of planets
    space_each = 1/npl
    space_above = 0.25 * space_each
    space_between = 0.05 * space_each
    space_transit = 0.60 * space_each 
    space_residuals = 0.20 * space_each

    plot_domains = [[0,space_residuals]]
    
    yaxis_dict = {}
    yaxis_dict[f'yaxis{1}'] = [0, plot_domains[0][1]]
    
    rows = npl*2
    for i in range(0,rows-1):
        if i==rows-2:
            domain_ = plot_domains[i][1] + space_between + space_transit
            dom = [plot_domains[i][1] + space_between, 1]
        elif (i % 2 == 0):
            domain_ =  plot_domains[i][1] + space_between + space_transit
            dom = [plot_domains[i][1] + space_between, domain_]
        else:
            domain_ = space_above + plot_domains[i][1] + space_residuals
            dom = [plot_domains[i][1] + space_above, domain_]
        
        plot_domains.append(dom)
        # Store the domain in the yaxis_dict dynamically
        yaxis_dict[f'yaxis{i+2}'] = dom
        
    for i in range(1, rows+1):
        fig.update_layout(
            **{
                f'xaxis{i}': dict(domain=[0, 1]),
                f'yaxis{i}': dict(domain=yaxis_dict[f'yaxis{rows+1-i}'])
            }
        )
        if not (i %2 ==0):
            fig.update_xaxes(showticklabels=False, row=i, col=1)
    
    
    ### Add annotations (titles) above each main plot
    annotations = []
    for i in range(1, rows+1, 2):  # Loop through odd rows (main plots only)
        y_position = yaxis_dict[f'yaxis{rows+1-i}'][1] + 0.25*space_residuals
        annotations.append(dict(
            x=0.5, y=y_position,  # Centered horizontally
            xref="paper", yref="paper",
            text=subplot_titles[i-1],  # Use corresponding title
            showarrow=False,
            font=dict(size=18)
        ))

    fig.update_traces(showlegend=True, row=1, col=1)
    fig.update_layout(annotations=annotations)
    
    ### JSON encode and return
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder) 
    return jsonify(graphJSON)



















#@app.route('/<koi_id>/plots/single_transit/<koi_id>_<planet>/<int:number>')
#def plot_single_transit(koi_id, planet, line_number):
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
    ext = os.path.basename(RUN_DIR) +'.csv'
    csv_file_path = os.path.join(RUN_DIR, ext)

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
    file_path_lc = os.path.join(RUN_DIR,star_id,file_name_lc)
    
    file_name_sc = star_id + '_sc_filtered.fits'
    file_path_sc = os.path.join(RUN_DIR, star_id, file_name_sc)

    ### posteriors for most likely model
    file_results =star_id + '-results.fits'
    file_path_results = os.path.join(RUN_DIR, star_id, file_results)
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
    file_paths = glob.glob(os.path.join(RUN_DIR,star_id, file_name))
    options = []
    for i in range(len(file_paths)):
        option_value = f"_{i:02d}_quick.ttvs"
        option = {'number': f'{i:02d}', 'value': option_value}
        options.append(option)
    return jsonify(options)


    

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
    file_paths = (glob.glob(os.path.join(RUN_DIR,star_id, file_name)))   
    file_path_sort = sorted(file_paths, key=lambda  x: int(re.search(r"_(\d+)_", x).group(1))) ### sort
    
    ext = os.path.basename(RUN_DIR) +'.csv'
    csv_file_path = os.path.join(RUN_DIR, ext)
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
    space_each = 1/systems # system contains the main plot and the residual plot
    space_above = 0.25 * space_each
    space_between = 0.05 * space_each
    space_transit = 0.60 * space_each 
    space_residuals = 0.20 * space_each
    plot_domains = [[0,space_residuals]]
    yaxis_dict[f'yaxis{1}'] = [0, plot_domains[0][1]]
    for i in range(0,rows-1):
        if i==rows-2:
            domain_ = plot_domains[i][1] + space_between + space_transit
            dom = [plot_domains[i][1] + space_between, 1]
        elif (i % 2 == 0):
            domain_ =  plot_domains[i][1] + space_between + space_transit
            dom = [plot_domains[i][1] + space_between, domain_]
        else:
            domain_ = space_above + plot_domains[i][1] + space_residuals
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
        y_position = yaxis_dict[f'yaxis{rows+1-i}'][1] + 0.25*space_residuals  # Get the top boundary of the yaxis and add offset
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
    file_path = os.path.join(RUN_DIR, star_id, file)

    ext = os.path.basename(RUN_DIR) +'.csv'
    csv_file_path = os.path.join(RUN_DIR, ext)

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
    file_paths = glob.glob(os.path.join(RUN_DIR,star_id, file_name))
    options = []
    for i in range(len(file_paths)):
        option_value =  f'{i}'
        option = {'number': f'{i:02d}', 'value': option_value}
        options.append(option)
    return jsonify(options)


if __name__ == '__main__':
    app.run(debug=True)
