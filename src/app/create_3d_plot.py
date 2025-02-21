import plotly.graph_objects as go
import numpy as np 

from src.data_utils.query_json import (get_sampled_region_with_rir,
                                             get_agent_ear_position,
                                             get_agent_rotation)
from .utils import polar_to_cartesian




noise_colors = ["#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

def create_3d_visualization(rirs,labels,data):
    # Create figure
    fig = go.Figure()
    fig.update_yaxes(autorange="reversed")
    agent_is_updated = False
    index_noise = 0
    # Won't loop if rir and label is None.
    for rir, label in zip(rirs, labels):
        region, sound_location, radius = get_sampled_region_with_rir(rir, data)
        sphere_center = get_agent_ear_position(region)
        azimuth, elevation = get_agent_rotation(region)
        dx, dy, dz = polar_to_cartesian(azimuth, elevation, r=1)
        # Create sphere surface (swap y/z coordinates for plotly's coordinate system)
        u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:20j]
        x = radius * np.cos(u)*np.sin(v) + sphere_center[0]
        y = radius * np.sin(u)*np.sin(v) + sphere_center[2]  # Matplotlib z becomes Plotly y
        z = radius * np.cos(v) + sphere_center[1]            # Matplotlib y becomes Plotly z
        color = "#1f77b4" if "source" in label else noise_colors[index_noise].upper(),
        # Add semi-transparent sphere
        fig.add_trace(go.Surface(
            x=x, y=y, z=z,
            colorscale=[[0, "sandybrown"], [1, "sandybrown"]],
            opacity=0.05,
            showscale=False,
            hoverinfo='none'
        ))
        if not agent_is_updated:
            add_agent_to_graph(fig, sphere_center, dx, dy, dz)
            agent_is_updated = True
        add_rir_to_graph(fig, label, sound_location, color)
        if "noise" in label:
            index_noise += 1

    update_fig_layout(fig)
    return fig

def update_fig_layout(fig):
    fig.update_layout(
        title='3D Sound Localization Visualization<br>Agent Head Orientation and Sound Source',
        scene=dict(
                aspectmode='cube',
                camera=dict(eye=dict(x=0, y=-2, z=2),
                            up = dict(x = 0, y = 1, z = 0),
                            ),
                yaxis= {'autorange': 'reversed'}, # reverse automatically
                ),
        margin=dict(l=0, r=0, b=0, t=40),
        legend=dict(
            x=0.8,
            y=0.9,
            bgcolor='rgba(255,255,255,0.5)'
        )
    )



def add_rir_to_graph(fig, label, rir_location, color):
    fig.add_trace(go.Scatter3d(
            x=[rir_location[0]],
            y=[rir_location[2]],
            z=[rir_location[1]],
            mode='markers',
            marker=dict(
                color=color,
                size=8,
                symbol='cross' if "source" in label else "x",
                opacity=0.8
            ),
            name=label
        ) )


def add_agent_to_graph(fig, sphere_center, dx, dy, dz):

    fig.add_trace(go.Scatter3d(
                x=[sphere_center[0]],
                y=[sphere_center[2]],
                z=[sphere_center[1]],
                mode='markers',
                marker=dict(
                    color='green',
                    size=5,
                    opacity=0.8
                ),
                name="Agent's Head Location"
            ))
    
            # Add orientation arrow
    fig.add_trace(go.Scatter3d(
        x=[sphere_center[0], sphere_center[0] + dx],
        y=[sphere_center[2], sphere_center[2] + dz],
        z=[sphere_center[1], sphere_center[1] + dy],
        mode='lines',
        line=dict(color='black', width=3),
        name='Head Orientation'
    ))

    fig.add_trace(go.Cone(
        x=[sphere_center[0] + dx],
        y=[sphere_center[2] + dz],
        z=[sphere_center[1] + dy],
        u=[dx],
        v=[dz],
        w=[dy],
        anchor='tip',
        showscale=False,
        sizemode='absolute',
        colorscale=[[0, 'black'], [1, 'black']]
    ))