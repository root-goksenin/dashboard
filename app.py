import dash 
import glob 
import librosa
import itertools
import os 
import subprocess


import dash_bootstrap_components as dbc
import pandas as pd 
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import soundfile as sf

from collections import Counter
from copy import deepcopy
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output, State
from itertools import count


from plotly.subplots import make_subplots


from src.app.loaders import load_json_with_houseid, load_rir_from_npy
from src.app.create_3d_plot import create_3d_visualization
from src.app.sound import process_audio, add_noise, convolve_with_rir
from src.app.utils import delete_assets, create_figure_template
from src.data_utils.query_json import (find_category_names,
                                   find_agent_height,
                                   find_agent_rotation, 
                                   find_sound_source_position,
                                   find_noise_position,
                                   get_sampled_region_with_rir,
                                   get_agent_rotation,
                                   get_source_efficency,
                                   get_noise_efficency)


def flatten(list_of_lists):
    return list(itertools.chain.from_iterable(list_of_lists))

def create_full_data_figure(house_ids):

    categories = [] 
    heights, azimuth, elevation = [], [], []
    r_ss, azimuth_ss, elevation_ss = [], [], []
    efficency_ss = []
    r_noise, azimuth_noise, elevation_noise, nr = [], [], [], []
    efficency_noise = []
    for selected_value in house_ids:
        data = load_json_with_houseid(selected_value)
        categories.append(find_category_names(data))
        heights.append(find_agent_height(data))
        azimuth_, elevation_ = find_agent_rotation(data)
        azimuth.append(azimuth_)
        elevation.append(elevation_)
        r_ss_, azimuth_ss_, elevation_ss_ = find_sound_source_position(data)
        r_ss.append(r_ss_)
        azimuth_ss.append(azimuth_ss_)
        elevation_ss.append(elevation_ss_)
        efficency_ss.append(get_source_efficency(data))
        r_noise_, elevation_noise_, azimuth_noise_, nr_ = find_noise_position(data)
        r_noise.append(r_noise_) 
        elevation_noise.append(elevation_noise_)
        azimuth_noise.append(azimuth_noise_) 
        nr.append(nr_)
        efficency_noise.append(get_noise_efficency(data))
    
    counts = Counter(flatten(categories))
    # Category Distribution Pie Chart
    pie_fig = px.pie(
        names=list(counts.keys()),
        values=list(counts.values()),
        color_discrete_sequence=px.colors.qualitative.Pastel,
        hole=0.4
    ).update_layout(create_figure_template(f"Room Category Distribution"))

    
    agent_fig = make_subplots(
        rows=1, 
        cols=3,
        subplot_titles=["Agent Height", "Azimuth", "Elevation"]
    )
    agent_fig.add_trace(go.Histogram(x=flatten(heights), name="Height", marker_color='#1f77b4', 
                                    xbins=dict(
                                            start=0.5,
                                            end=2.0,
                                            size=0.1),), 1, 1)
    agent_fig.add_trace(go.Histogram(x=flatten(azimuth), name="Azimuth", marker_color='#ff7f0e',
                                     xbins=dict(
                                            start=0,
                                            end=360,
                                            size=10),), 1, 2)
    agent_fig.add_trace(go.Histogram(x=flatten(elevation), name="Elevation", marker_color='#2ca02c',
                                      xbins=dict(
                                            start=-50,
                                            end=50,
                                            size=5)), 1, 3)
    agent_fig.update_layout(create_figure_template("Agent Position Analysis"), 
                           showlegend=False,
                           bargap=0.1)

    sound_fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=["Distance", "Azimuth", "Elevation", "RIR Ray Efficency"]
    )
    sound_fig.add_trace(go.Histogram(x=flatten(r_ss), marker_color='#17becf', xbins=dict(
                                            start=1.0,
                                            end=5.0,
                                            size=0.2)), 1, 1)
    sound_fig.add_trace(go.Histogram(x=flatten(azimuth_ss), marker_color='#bcbd22',  xbins=dict(
                                            start=0,
                                            end=360,
                                            size=10)), 1, 2)
    sound_fig.add_trace(go.Histogram(x=flatten(elevation_ss), marker_color='#7f7f7f',  xbins=dict(
                                            start=-90,
                                            end=90,
                                            size=5)), 2, 1)
    sound_fig.add_trace(go.Histogram(x=flatten(efficency_ss), marker_color='#d62728',  xbins=dict(
                                            start=0.0,
                                            end=1.0,
                                            size=0.05)), 2, 2)

    sound_fig.update_layout(create_figure_template("Sound Source Properties"), 
                           bargap=0.1, 
                           height=600,
                           showlegend = False)
    
    # Noise Properties

    noise_fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[
            "Noise Distance", "Azimuth",
            "Elevation", "RIR Ray Efficency"
        ]
    )
    noise_fig.add_trace(go.Histogram(x=flatten(r_noise), marker_color='#17becf', xbins=dict(
                                            start=1.0,
                                            end=5.0,
                                            size=0.2)), 1, 1)
    noise_fig.add_trace(go.Histogram(x=flatten(azimuth_noise), marker_color='#bcbd22',  xbins=dict(
                                            start=0,
                                            end=360,
                                            size=10)), 1, 2)
    noise_fig.add_trace(go.Histogram(x=flatten(elevation_noise), marker_color='#7f7f7f',  xbins=dict(
                                            start=-90,
                                            end=90,
                                            size=5)), 2, 1)
    noise_fig.add_trace(go.Histogram(x=flatten(efficency_noise), marker_color='#d62728',  xbins=dict(
                                            start=0.0,
                                            end=1.0,
                                            size=0.05)), 2, 2)
    noise_fig.update_layout(create_figure_template("Noise Characteristics"), 
                          height=600,
                          bargap=0.1,
                          showlegend = False)
    noise_numbers = go.Figure()
    noise_numbers.add_trace(go.Histogram(x=flatten(nr), marker_color='#17becf'))
    noise_numbers.update_layout(create_figure_template("Number of Noise Sources"), 
                        bargap=0.1, 
                        showlegend = False)
    return pie_fig, agent_fig, sound_fig, noise_fig, noise_numbers

# Custom styles
CARD_STYLE = {
    "boxShadow": "0 4px 8px 0 rgba(0,0,0,0.2)",
    "borderRadius": "15px",
    "padding": "20px",
    "marginBottom": "20px",
    "backgroundColor": "#f8f9fa"
}

GRAPH_STYLE = {
    "border": "1px solid #dee2e6",
    "borderRadius": "12px",
    "padding": "15px",
    "backgroundColor": "white"
}


house_ids = [os.path.basename(a).replace(".json", "") for a in glob.glob("data/*.json")]
app = dash.Dash(external_stylesheets=[dbc.themes.JOURNAL])
pie_fig, agent_fig, sound_fig, noise_fig, noise_numbers = create_full_data_figure(house_ids)

# Layout with improved structure and styling
app.layout = dbc.Container([
            dcc.Store(id='data'),
            dcc.Store(id="source_rirs"),
            dcc.Store(id="noise_rirs"),
            dcc.Store(id="convolved_source"),
            dcc.Store(id="convolved_summed_up_noise"),
            dbc.Row([
                dbc.Col(html.H1("Audio Scene Analysis Dashboard", 
                                className="text-center my-4 fw-bold text-primary"),
                        width=12)
            ]),
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.Label("Select House ID:", className="fw-bold mb-2"),
                            dcc.Dropdown(
                                id='data-selector',
                                options=[{'label': id, 'value': id} for id in house_ids],
                                value=house_ids[0],
                                clearable=False,
                                className="mb-3"
                            )
                        ])
                    ], style=CARD_STYLE)
                ], width=8, className="mx-auto")
            ]),
        dbc.Tabs([
            dbc.Tab(label='Instance Statistics (^)', tabClassName="flex-grow-1 text-center",
                    children = [
                        dbc.Row([
                            dbc.Col(dcc.Graph(id='house_properties'), 
                                    lg=6, className="mb-4"),
                            dbc.Col(dcc.Graph(id='agent_rotation'), 
                                    lg=6, className="mb-4")
                        ], className="g-4"),
                        
                        dbc.Row([
                            dbc.Col(dcc.Graph(id="sound_source_properties"), 
                                    lg=6, className="mb-4"),
                            dbc.Col(dcc.Graph(id="noise_properties"), 
                                    lg=6, className="mb-4")
                        ], className="g-4"), 

                        dcc.Graph(id="noise_numbers")
            ]),
            dbc.Tab(label='General Statistics', tabClassName="flex-grow-1 text-center",
                    children = [
                            dbc.Row([
                                    dbc.Col(dcc.Graph(id='house_properties_g', figure = pie_fig), 
                                            lg=6, className="mb-4"),
                                    dbc.Col(dcc.Graph(id='agent_rotation_g', figure = agent_fig), 
                                            lg=6, className="mb-4")
                                ], className="g-4"),
                                
                                dbc.Row([
                                    dbc.Col(dcc.Graph(id="sound_source_properties_g", figure = sound_fig), 
                                            lg=6, className="mb-4"),
                                    dbc.Col(dcc.Graph(id="noise_properties_g", figure = noise_fig), 
                                            lg=6, className="mb-4")
                                ], className="g-4"), 
                                dcc.Graph(id="noise_numbers_g", figure = noise_numbers)
                    ]),
            dbc.Tab(label='RIR Explorer', tabClassName="flex-grow-1 text-center", children=[
                        dash_table.DataTable(
                        id='datatable',
                        columns= None,
                        data= None,
                        sort_action="native",
                        sort_mode="multi",
                        row_selectable="single",
                        selected_rows=[],
                        page_action="native",
                        page_current= 0,
                        page_size= 10,
                        style_cell={'fontSize':14, 'font-family':'ui-sans-serif'},
                        style_table={'overflowX': 'auto'},
                    ),
                    dcc.Graph(id="audio_rir_graph"),
                    dcc.Graph(id="audio_rir_graph_of_rir"),
                    dbc.Row([
                        dbc.Col(html.Img(id = "backend_plot", style={'height':'80%','width':'80%'}), lg=6, className="mb-4"),
                        dbc.Col(dcc.Graph(id="3d_rir"), lg=6, className="mb-4")],
                        className="g-4"
                    ),
                    html.Audio(id='sample_rir_audio', controls=True, src="assets/base_audio.wav")])
            ], persistence = False)],
        
        fluid=True, className="p-5")



@app.callback(
    Output('datatable', "data"),
    Output('datatable', "columns"),
    Output("datatable", "hidden_columns"),
    Input('data', "data"),)
def update_datatable(data):

    rows = []
    house_id = data['house']['id']
    for index, data_region in enumerate(data['sampled_regions']):
        path_1 = os.path.basename(data_region['region']['scene']['source']['rir']['rir_path'])
        scene_id = data_region['region']['seed']
        radius_1 = data_region['region']['scene']['source']['radius']
        azimuth_1 = data_region['region']['scene']['source']['azimuth']
        elevation_1 = data_region['region']['scene']['source']['elevation']
        eff_1 = data_region['region']['scene']['source']['rir']['ray_efficiency']
        agent_azimuth_1 = data_region['region']["sensor_states"]["audio_sensor"]["sensor_azimuth"][0]
        agent_elevation_1 = data_region['region']["sensor_states"]["audio_sensor"]["sensor_elevation"][0]
        room_id = data_region["region"][next(iter(data_region["region"]))]["category_name"]
        source_id = f"source_{index}"
        rows.append([scene_id, room_id, source_id, path_1, eff_1, radius_1, azimuth_1, elevation_1, agent_azimuth_1, agent_elevation_1])
        for noise_index, noise_scene in enumerate(data_region['region']['scene']['noise']):
            path_2 = os.path.basename(noise_scene['rir']['rir_path'])
            eff_2 = noise_scene['rir']['ray_efficiency']
            radius_2 = noise_scene['radius']
            azimuth_2 = noise_scene['azimuth']
            elevation_2 = noise_scene['elevation']
            noise_index = f"noise_{index}_{noise_index}"
            rows.append([scene_id, room_id,noise_index, path_2, eff_2, radius_2, azimuth_2, elevation_2, agent_azimuth_1, agent_elevation_1])

    columns = ["scene_id", "room_id", "rir_id", "path", "efficency", "radius", "azimuth", "elevation", "agent_azimuth", "agent_elevation", ]
    df = pd.DataFrame(rows, columns=columns)
    max_seed = max(map(lambda x: int(x.split("_")[-1].replace(".png", "")), glob.glob(f"plots/{house_id}*.png")))
    df = df[df["scene_id"] <= max_seed]

    return df.to_dict('records'), [
            {"name": i, "id": i, "deletable": True, "selectable": True} for i in df.columns
        ], ["path"]

@app.callback(
    Output('source-selector', 'options'),
    Output("source_rirs", 'data'),
    Output("noise_rirs", 'data'),
    Input('data', 'data')
)
def update_rir_selector(data):
    options = []
    noise_rirs = {}
    for index, region in enumerate(data["sampled_regions"]):
        scene = region['region']['scene']
        source_rir = scene['source']['rir']['rir_path']
        values = {}
        values['label'] = f'source_{index}' 
        values['value'] = os.path.basename(source_rir)
        options.append(values)
        noise_rirs[f'source_{index}'] = []
        for index_, noise in enumerate(scene["noise"]):
            noise_rirs[f'source_{index}'].append({"label": f"noise_{index_}" , "value" : os.path.basename(noise['rir']['rir_path'])})

    return options, options, noise_rirs


@app.callback(
    Output('sample_rir_audio', 'src'),
    Output('audio_rir_graph', 'figure'),
    Output("audio_rir_graph_of_rir", "figure"),
    Output("backend_plot", "src"),
    Output("3d_rir", "figure"),

    Input('datatable', 'selected_rows'),
    Input('datatable', 'data'),
    Input('data', 'data'),
    State('data-selector', 'value'),


)
def show_rir(selected_rows, data, house_data, house_id, count_ = count(0)):
    path = "assets/base_audio.wav"
    base_audio, _ = librosa.load(path, sr = 16000)
    if len(selected_rows) == 0 or (selected_rows is None):
        audio_fig = go.Figure()
        threed_fig = create_3d_visualization([], [], data)
        time = np.arange(0, base_audio.shape[0]) / 16000
        audio_fig.add_trace(go.Scatter(x=time, y=base_audio,
                        mode='lines',
                        name='Mono'))
        audio_fig.update_layout(
                title=dict(
                    text='Audio Graph'
                ),
                xaxis=dict(
                    title=dict(
                        text='Time in s'
                    )
                ),
                yaxis=dict(
                    title=dict(
                        text='Amplitude'
                    )
                ),
        )
        rir_fig = go.Figure()
        rir_fig.update_layout(
            title=dict(
                text='RIR Graph'
            ),
            xaxis=dict(
                title=dict(
                    text='Time in s'
                )
            ),
            yaxis=dict(
                title=dict(
                    text='Amplitude'
                )
            ),
        )
        backend_fig = go.Figure()
        backend_fig.update_layout(
            title=dict(
                text='Top Down Map'
            ),
        )
        return path, audio_fig, rir_fig, "", threed_fig
    row = data[selected_rows[0]]
    rir_path = row["path"]
    scene_id = row["scene_id"]
    # Search window of 10 to find noise/sources with the same scene
    scene_rir_paths = [] 
    scene_rir_ids = []
    for inputs in data[max(0, selected_rows[0] - 6) : min(len(data), selected_rows[0] + 6)]:
        if inputs["scene_id"] == scene_id:
            scene_rir_paths.append(inputs["path"])
            scene_rir_ids.append(inputs["rir_id"])
    threed_fig = create_3d_visualization(scene_rir_paths, scene_rir_ids, house_data)

    loaded_rir = load_rir_from_npy(f"rirs/{rir_path}")
    processed_audio = convolve_with_rir(loaded_rir, base_audio)
    path = f"assets/convolved_rir_explorer_{deepcopy(count_)}.wav"
    next(count_)
    sf.write(path, processed_audio.T, 16000)
    time = np.arange(0, processed_audio.shape[1]) / 16000
    audio_fig = go.Figure()
    audio_fig.add_trace(go.Scatter(x=time, y=processed_audio[0, :],
                    mode='lines',
                    name='Left Channel'))
    audio_fig.add_trace(go.Scatter(x=time, y=processed_audio[1, :],
                    mode='lines',
                    name='Right Channel'))
    
    audio_fig.update_layout(
            title=dict(
                text='Audio Graph'
            ),
            xaxis=dict(
                title=dict(
                    text='Time in s'
                )
            ),
            yaxis=dict(
                title=dict(
                    text='Amplitude'
                )
            ),
    )
    rir_fig = go.Figure()
    time = np.arange(0, loaded_rir.shape[1]) / 16000

    rir_fig.add_trace(go.Scatter(x=time, y=loaded_rir[0, :],
                    mode='lines',
                    name='Left Channel'))
    rir_fig.add_trace(go.Scatter(x=time, y=loaded_rir[1, :],
                    mode='lines',
                    name='Right Channel'))
    
    rir_fig.update_layout(
            title=dict(
                text='RIR Graph'
            ),
            xaxis=dict(
                title=dict(
                    text='Time in s'
                )
            ),
            yaxis=dict(
                title=dict(
                    text='Amplitude'
                )
            ),
    )

    if not os.path.exists(f"plots/{house_id}_{scene_id}.png"):
        subprocess.check_call(["./generate_topdown.sh", str(house_id), str(scene_id)])
    
    import shutil
    if os.path.exists(f"plots/{house_id}_{scene_id}.png"):
        shutil.copy(f"plots/{house_id}_{scene_id}.png", f"assets/{house_id}_{scene_id}.png")
    return path, audio_fig, rir_fig, f"assets/{house_id}_{scene_id}.png", threed_fig



@app.callback(
    Output('house_properties', 'figure'),
    Output('agent_rotation', 'figure'),
    Output('sound_source_properties', 'figure'),
    Output('noise_properties', 'figure'),
    Output('noise_numbers', 'figure'),
    Output('data', 'data'),
    Input('data-selector', 'value')
)
def update_graph(selected_value):
    data = load_json_with_houseid(selected_value)
    categories = find_category_names(data)
    counts = Counter(categories)
    
    # Category Distribution Pie Chart
    pie_fig = px.pie(
        names=list(counts.keys()),
        values=list(counts.values()),
        color_discrete_sequence=px.colors.qualitative.Pastel,
        hole=0.4
    ).update_layout(create_figure_template(f"Room Category Distribution - {selected_value}"))
    
    # Agent Rotation Analysis
    heights = find_agent_height(data)
    azimuth, elevation = find_agent_rotation(data)
    
    agent_fig = make_subplots(
        rows=1, 
        cols=3,
        subplot_titles=["Agent Height", "Azimuth", "Elevation"]
    )
    agent_fig.add_trace(go.Histogram(x=heights, name="Height", marker_color='#1f77b4', 
                                    xbins=dict(
                                            start=0.5,
                                            end=2.0,
                                            size=0.1),), 1, 1)
    agent_fig.add_trace(go.Histogram(x=azimuth, name="Azimuth", marker_color='#ff7f0e',
                                     xbins=dict(
                                            start=0,
                                            end=360,
                                            size=10),), 1, 2)
    agent_fig.add_trace(go.Histogram(x=elevation, name="Elevation", marker_color='#2ca02c',
                                      xbins=dict(
                                            start=-50,
                                            end=50,
                                            size=5)), 1, 3)
    agent_fig.update_layout(create_figure_template("Agent Position Analysis"), 
                           showlegend=False,
                           bargap=0.1)
    
    # Sound Source Properties
    r_ss, azimuth_ss, elevation_ss = find_sound_source_position(data)
    efficency_ss = get_source_efficency(data)

    sound_fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=["Distance", "Azimuth", "Elevation", "RIR Ray Efficency"]
    )
    sound_fig.add_trace(go.Histogram(x=r_ss, marker_color='#17becf', xbins=dict(
                                            start=1.0,
                                            end=5.0,
                                            size=0.2)), 1, 1)
    sound_fig.add_trace(go.Histogram(x=azimuth_ss, marker_color='#bcbd22',  xbins=dict(
                                            start=0,
                                            end=360,
                                            size=10)), 1, 2)
    sound_fig.add_trace(go.Histogram(x=elevation_ss, marker_color='#7f7f7f',  xbins=dict(
                                            start=-90,
                                            end=90,
                                            size=5)), 2, 1)
    sound_fig.add_trace(go.Histogram(x=efficency_ss, marker_color='#d62728',  xbins=dict(
                                            start=0.0,
                                            end=1.0,
                                            size=0.05)), 2, 2)

    sound_fig.update_layout(create_figure_template("Sound Source Properties"), 
                           bargap=0.1, 
                           height=600,
                           showlegend = False)
    
    # Noise Properties
    r_noise, elevation_noise, azimuth_noise, nr = find_noise_position(data)
    efficency_noise = get_noise_efficency(data)

    noise_fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[
            "Noise Distance", "Azimuth",
            "Elevation", "RIR Ray Efficency"
        ]
    )
    noise_fig.add_trace(go.Histogram(x=r_noise, marker_color='#17becf', xbins=dict(
                                            start=1.0,
                                            end=5.0,
                                            size=0.2)), 1, 1)
    noise_fig.add_trace(go.Histogram(x=azimuth_noise, marker_color='#bcbd22',  xbins=dict(
                                            start=0,
                                            end=360,
                                            size=10)), 1, 2)
    noise_fig.add_trace(go.Histogram(x=elevation_noise, marker_color='#7f7f7f',  xbins=dict(
                                            start=-90,
                                            end=90,
                                            size=5)), 2, 1)
    noise_fig.add_trace(go.Histogram(x=efficency_noise, marker_color='#d62728',  xbins=dict(
                                            start=0.0,
                                            end=1.0,
                                            size=0.05)), 2, 2)
    noise_fig.update_layout(create_figure_template("Noise Characteristics"), 
                          height=600,
                          bargap=0.1,
                          showlegend = False)
    noise_numbers = go.Figure()
    noise_numbers.add_trace(go.Histogram(x=nr, marker_color='#17becf'))
    noise_numbers.update_layout(create_figure_template("Number of Noise Sources"), 
                        bargap=0.1, 
                        showlegend = False)
    return pie_fig, agent_fig, sound_fig, noise_fig, noise_numbers, data


if __name__ == "__main__":
    delete_assets()
    audio, _ = librosa.load("samples/CSrWzBYQ3W0.flac", sr = 16000)
    additional_noise, _ = librosa.load("samples/22ga010i_0.28655_051o020s_-0.28655.wav", sr = 16000)
    sf.write("assets/base_audio.wav", audio, 16000)
    app.run_server(port=10000, host='0.0.0.0')