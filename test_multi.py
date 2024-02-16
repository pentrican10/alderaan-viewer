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

from plotly.subplots import make_subplots

data_directory = 'c:\\Users\\Paige\\Projects\\data\\alderaan_results\\2023-05-15_doubles'


def generate_plot_OMC(koi_id):
    star_id = koi_id.replace("K","S")
    file_name = star_id + '_*_quick.ttvs'
    file_paths = glob.glob(os.path.join(data_directory,star_id, file_name))
    ### number of planets from number of ttv files
    npl = len(file_paths)
    fig = make_subplots(rows=npl, cols=1,
                        subplot_titles=[f"File 0{i}" for i in range(len(file_paths))])
    # Create a list to store subplot titles
    #subplot_titles = [f"File 0{i}" for i in range(len(file_paths))]

    for i, file_path in enumerate(file_paths):
        omc_data, omc_model, out_prob, out_flag = data_load.OMC_data(koi_id, file_path)
        show_outliers = True

        if omc_data is not None:
            mask = [bool(flag) for flag in out_flag]
            if show_outliers:
                omc = px.scatter(omc_data, #[mask], 
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

                # fig.update_layout(xaxis_title=f"TIME (DAYS)", 
                #                 yaxis_title="O-C (MINUTES)",
                #                 coloraxis_colorbar=dict(title='Out Probability'))
                
                # Update x-axis and y-axis labels for each subplot
                fig.update_xaxes(title_text="TIME (DAYS)", row=i+1, col=1)
                fig.update_yaxes(title_text="O-C (MINUTES)", row=i+1, col=1)
                fig.update_coloraxes(colorbar_title_text='Out Probability', colorbar_len=0.2, row=i+1, col=1)

            else:
                mask_arr = np.array(mask)
                omc = px.scatter(omc_data[~mask_arr], x="TIME",y="OMC")
                # Add a line plot for OMC_MODEL
                line_trace = px.line(omc_model[~mask_arr], x="TIME", y="OMC_MODEL").data[0]
                line_trace.line.color = 'red'  # Set the line color to red
                fig.add_trace(omc, row=(i+1), col=1)
                fig.add_trace(line_trace, row=(i+1), col=1)
                # Update x-axis label with units
                # fig.update_layout(xaxis_title=f"TIME (DAYS)", 
                #                 yaxis_title="O-C (MINUTES)",
                #                 coloraxis_colorbar=dict(title='Out Probability'))
                # Update x-axis and y-axis labels for each subplot

                fig.update_xaxes(title_text="TIME (DAYS)", row=i+1, col=1)
                fig.update_yaxes(title_text="O-C (MINUTES)", row=i+1, col=1)
                fig.update_coloraxes(colorbar_title_text='Out Probability', colorbar_len=0.2, row=i+1, col=1)
    fig.show()

            #graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder) 
            #return jsonify(graphJSON)
        # else: 
        #     error_message = f'No data found for {koi_id}'
        #     return jsonify(error_message=error_message)
        

### IS THERE A BETTER WAY TO TELL HOW MANY PLANETS THERE ARE
def generate_plot_folded_light_curve(koi_id):
    star_id = koi_id.replace("K","S")
    file_name = star_id + '_*_quick.ttvs'
    file_paths = glob.glob(os.path.join(data_directory,star_id, file_name))
    ### number of planets from number of ttv files
    npl = len(file_paths)
    fig = make_subplots(rows=npl, cols=1,
                        subplot_titles=[f"File 0{i}" for i in range(len(file_paths))])
    
    for i, file_path in enumerate(file_paths):
        fold_data = data_load.folded_data(koi_id,file_path)

        if fold_data is not None:
            fold = px.scatter(fold_data, x="TIME",y="FLUX").data[0]
            fig.add_trace(fold, row=i+1, col=1)
            # Update x-axis label with units
            # Update x-axis and y-axis labels for each subplot
            fig.update_xaxes(title_text="TIME (HOURS)", row=i+1, col=1)
            fig.update_yaxes(title_text="FLUX", row=i+1, col=1)
    fig.show()
        #     graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder) 
        #     return jsonify(graphJSON)
        # else:
        #     error_message = f'No data found for {koi_id}'
        #     return jsonify(error_message=error_message)



if __name__ == '__main__':
    def read_fits_file(file_path):
        with fits.open(file_path) as hdul:
            # Print general information about the FITS file
            print("Number of HDUs (extensions):", len(hdul))

            # Print information about each extension
            for i, hdu in enumerate(hdul):
                print(f"\nHDU {i + 1} (Extension {i}):")
                print("Header:")
                print(repr(hdu.header))
                print("Data:")
                print(hdu.data)

    # Provide the path to your FITS file
    fits_file_path = "C://Users//Paige//Projects//data//alderaan_results//2023-05-15_doubles//S00301//S00301_lc_detrended.fits"

    # Call the function to read and display information about the FITS file
    #read_fits_file(fits_file_path)
    with fits.open(fits_file_path) as fits_file:
        lc_flux = np.array(fits_file[2].data, dtype=float)
        lc_flux_err = np.array(fits_file[3].data, dtype=float)
    print(len(lc_flux))
    print(len(lc_flux_err))
    print(lc_flux_err)