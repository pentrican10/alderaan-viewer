
@app.route('/generate_plot_single_transit/<koi_id>/<int:line_number>/<planet>')
def generate_plot_single_transit(koi_id, line_number,planet):
    planet = request.args.get('planet', '_00_quick.ttvs')
    star_id = koi_id.replace("K","S")
    
    ttv_file = star_id + planet
    #file_paths = glob.glob(os.path.join(data_directory,star_id, file_name))
    ### initialize figure
    fig = make_subplots(rows=1, cols=1)

    if (data_load.single_transit_data(koi_id, line_number,ttv_file)):
        photometry_data, transit_number, center_time = data_load.single_transit_data(koi_id, line_number,ttv_file)
        transit = px.scatter(photometry_data, x="TIME", y="FLUX").data[0]
        fig.add_trace(transit, row=1, col=1)
        # Update x-axis label with units
        fig.update_layout(xaxis_title=f"TIME (DAYS)", yaxis_title="FLUX")

        graphJSON= json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder) 
        response_data = {
            'graphJSON': graphJSON,
            'transit_number': transit_number
        }
        return jsonify(response_data)
    else:
        error_message = f'No data found for {koi_id}'
        return jsonify(error_message=error_message)

