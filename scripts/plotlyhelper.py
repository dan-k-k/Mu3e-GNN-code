# plotlyhelper.py
# This cell is required for 3D Plotly visualisation of frames of truth tracks (with segs10) and generated graphs ('real_tracks_df', 'fake_tracks_df', 'missing_real_tracks_df').
import numpy as np
import plotly.graph_objects as go
import plotly.offline as pyo
import itertools
import pandas as pd

# Define wireframe layer parameters based on Table 7.1
layers = [
    {
        'name': 'Layer 1',
        'radius': 23.3,      # in mm
        'length': 124.7,     # instrumented length in mm
        'color': 'red',
        'opacity': 0.2,
        'z_shift': 0         # No shift for main layers
    },
    {
        'name': 'Layer 2',
        'radius': 29.8,
        'length': 124.7,
        'color': 'green',
        'opacity': 0.2,
        'z_shift': 0
    },
    {
        'name': 'Layer 3',
        'radius': 73.9,
        'length': 351.9,
        'color': 'blue',
        'opacity': 0.2,
        'z_shift': 0
    },
    {
        'name': 'Layer 4',
        'radius': 86.3,
        'length': 372.6,
        'color': 'grey',
        'opacity': 0.2,
        'z_shift': 0
    }
]

# Define recurl layers (only outer layers, similar radii to Layers 3 and 4)
recurl_layers = [
    {
        'name': 'Recurl Layer 3 +',
        'radius': 73.9,      # same as Layer 3
        'length': 351.9,     # same as Layer 3
        'color': 'blue',
        'opacity': 0.1,
        'z_shift': 351.9      # Shift to +z end
    },
    {
        'name': 'Recurl Layer 3 -',
        'radius': 73.9,      # same as Layer 3
        'length': 351.9,     # same as Layer 3
        'color': 'blue',
        'opacity': 0.1,
        'z_shift': -351.9     # Shift to -z end
    },
    {
        'name': 'Recurl Layer 4 +',
        'radius': 86.3,      # same as Layer 4
        'length': 372.6,     # same as Layer 4
        'color': 'grey',
        'opacity': 0.1,
        'z_shift': 372.6      # Shift to +z end
    },
    {
        'name': 'Recurl Layer 4 -',
        'radius': 86.3,      # same as Layer 4
        'length': 372.6,     # same as Layer 4
        'color': 'grey',
        'opacity': 0.1,
        'z_shift': -372.6     # Shift to -z end
    }
]

# Combine all layers
all_layers = layers + recurl_layers

def create_wireframe(layer, num_theta=100, num_z=2):
    theta = np.linspace(0, 2 * np.pi, num_theta)
    z_original = np.linspace(-layer['length']/2, layer['length']/2, num_z)
    theta_grid, z_grid = np.meshgrid(theta, z_original)
    x_original = layer['radius'] * np.cos(theta_grid)  # Original Z
    y_grid = layer['radius'] * np.sin(theta_grid)      # Original Y

    # Flip axes: New X = Original Z + z_shift, New Y = Original Y, New Z = Original X
    x_grid_new = z_grid + layer.get('z_shift', 0)     # Shifted along new X-axis
    y_grid_new = y_grid
    z_grid_new = x_original

    # Initialize list to hold Scatter3d traces for wireframe
    traces = []

    # Top and Bottom circles
    for i in range(num_z):
        trace = go.Scatter3d(
            x=x_grid_new[i, :],
            y=y_grid_new[i, :],
            z=z_grid_new[i, :],
            mode='lines',
            line=dict(color=layer['color'], width=2),
            name=layer['name'] if i == 0 else None,  # Name only once
            opacity=layer['opacity'],
            showlegend=True if i == 0 else False,  # Show legend only once
            hoverinfo='skip'  # Disable hover for detector layers
        )
        traces.append(trace)

    # Vertical lines
    for i in range(num_theta):
        trace = go.Scatter3d(
            x=[x_grid_new[0, i], x_grid_new[1, i]],
            y=[y_grid_new[0, i], y_grid_new[1, i]],
            z=[z_grid_new[0, i], z_grid_new[1, i]],
            mode='lines',
            line=dict(color=layer['color'], width=1),
            name=None,
            opacity=layer['opacity'],
            showlegend=False,         # Exclude wireframes from legend
            hoverinfo='skip'  # Disable hover for detector layers
        )
        traces.append(trace)

    return traces

# Generate all wireframe traces
wireframe_traces = []
for layer in all_layers:
    wireframe_traces.extend(create_wireframe(layer, num_theta=100, num_z=2))

def visualise_frame1(frame_id, hits_df_unique, real_tracks_df, fake_tracks_df, missing_real_tracks_df, wireframe_traces):
    import plotly.graph_objects as go
    import plotly.offline as pyo
    
    # Filter hits for the given frame
    frame_hits = hits_df_unique[hits_df_unique['frameId'] == frame_id]
    
    # Filter real, fake, and missing real validated tracks for the given frame
    frame_real_tracks = real_tracks_df[real_tracks_df['frameId'] == frame_id]
    frame_fake_tracks = fake_tracks_df[fake_tracks_df['frameId'] == frame_id]
    frame_missing_real_tracks = missing_real_tracks_df[missing_real_tracks_df['frameId'] == frame_id]
    
    # Collect all hits from validated real tracks (for unique hit trace)
    validated_real_hits = []
    for _, track in frame_real_tracks.iterrows():
        validated_real_hits.extend(track['hits'])
    validated_real_hits_df = pd.DataFrame(validated_real_hits)
    
    # (Similarly, fake and missing real hits can be collected if needed.)
    
    # Create Scatter3d trace for all unique hits before validation (crosses)
    unique_hits_trace = go.Scatter3d(
        x=frame_hits['z'],  # Flipped axes: Original Z to X
        y=frame_hits['y'],  # Y remains Y
        z=frame_hits['x'],  # Flipped axes: Original X to Z
        mode='markers',
        marker=dict(
            size=4,
            symbol='cross',
            color='black',
            opacity=0.5
        ),
        name='Hits',
        hoverinfo='text',
        text=frame_hits.apply(
            lambda row: (
                f"Hit ID: {row['hit_id']}<br>"
                f"Layer: {row['layer']}<br>"
                f"X: {row['x']:.2f} mm, Y: {row['y']:.2f} mm, Z: {row['z']:.2f} mm"
            ), axis=1
        )
    )
    
    # Initialize lists for track traces
    real_track_traces = []
    fake_track_traces = []
    missing_real_track_traces = []
    
    # Define styling for real, fake, and missing real tracks
    real_track_style = {
        'mode': 'lines+markers',
        'marker': dict(
            size=5,
            symbol='circle',
            color='green',
            opacity=0.9
        ),
        'line': dict(
            color='green',
            width=4,
            dash='solid'
        ),
        'hoverinfo': 'text'
    }
    
    fake_track_style = {
        'mode': 'lines+markers',
        'marker': dict(
            size=5,
            symbol='circle',
            color='red',
            opacity=0.5
        ),
        'line': dict(
            color='red',
            width=2,
            dash='dash'
        ),
        'hoverinfo': 'text'
    }
    
    missing_real_track_style = {
        'mode': 'lines+markers',
        'marker': dict(
            size=5,
            symbol='circle',
            color='blue',
            opacity=0.7
        ),
        'line': dict(
            color='blue',
            width=3,
            dash='dot'
        ),
        'hoverinfo': 'text'
    }
    
    # Create Scatter3d traces for validated REAL tracks with a counter in the legend.
    real_counter = 0
    for idx, track in frame_real_tracks.iterrows():
        real_counter += 1
        track_hits = pd.DataFrame(track['hits'])
        
        trace = go.Scatter3d(
            x=track_hits['z'],  # Flipped axes: Original Z to X
            y=track_hits['y'],  # Y remains Y
            z=track_hits['x'],  # Flipped axes: Original X to Z
            **real_track_style,
            name=f"Real Graph {real_counter}",
            text=[
                (
                    f"Real Graph<br>"
                    f"Hit ID: {hit['hit_id']}<br>"
                    f"X: {hit['x']:.2f} mm, Y: {hit['y']:.2f} mm, Z: {hit['z']:.2f} mm"
                ) for hit in track_hits.to_dict('records')
            ]
        )
        real_track_traces.append(trace)
    
    # Create Scatter3d traces for validated FAKE tracks with a counter in the legend.
    fake_counter = 0
    for idx, track in frame_fake_tracks.iterrows():
        fake_counter += 1
        track_hits = pd.DataFrame(track['hits'])
        
        trace = go.Scatter3d(
            x=track_hits['z'],
            y=track_hits['y'],
            z=track_hits['x'],
            **fake_track_style,
            name=f"Fake Graph {fake_counter}",
            text=[
                (
                    f"Fake Graph<br>"
                    f"Hit ID: {hit['hit_id']}<br>"
                    f"X: {hit['x']:.2f} mm, Y: {hit['y']:.2f} mm, Z: {hit['z']:.2f} mm"
                ) for hit in track_hits.to_dict('records')
            ]
        )
        fake_track_traces.append(trace)
    
    # Create Scatter3d traces for validated MISSING REAL tracks with a counter.
    missing_counter = 0
    for idx, track in frame_missing_real_tracks.iterrows():
        missing_counter += 1

        # force the DataFrame to have x,y,z column names
        track_hits = pd.DataFrame(track['hits'], columns=['x','y','z'])

        trace = go.Scatter3d(
            x=track_hits['z'],  # now this exists
            y=track_hits['y'],
            z=track_hits['x'],
            **missing_real_track_style,
            name=f"Missing Real Track {missing_counter}",
            text=[
                (
                    f"Missing Real Track<br>"
                    # f"Hit ID: {hit['hit_id']}<br>"
                    f"X: {hit['x']:.2f} mm, Y: {hit['y']:.2f} mm, Z: {hit['z']:.2f} mm"
                ) for hit in track_hits.to_dict('records')
            ]
        )
        missing_real_track_traces.append(trace)
    
    # Combine all traces: wireframes first, then unique hits, then real, fake, and missing real tracks.
    all_traces = wireframe_traces + [unique_hits_trace] + real_track_traces + fake_track_traces + missing_real_track_traces
    
    layout = go.Layout(
        title=f'Visualisation of Real and Fake Graphs for Frame ID: {frame_id}',
        scene=dict(
            xaxis=dict(title='Z (mm)', autorange=True),
            yaxis=dict(title='Y (mm)', autorange=True),
            zaxis=dict(title='X (mm)', autorange=True),
            aspectmode='data'
        ),
        legend=dict(itemsizing='constant')
    )
    
    fig = go.Figure(data=all_traces, layout=layout)
    pyo.plot(fig, filename=f'frame_{frame_id}_visualization_realfake_missing.html')
    
color_cycle = itertools.cycle(['grey'])

def visualise_truth_tracks(frame_id, tracks_df, wireframe_traces):
    import plotly.graph_objects as go
    import plotly.offline as pyo
    
    # Filter the tracks for the target frame.
    frame_tracks = tracks_df[tracks_df['frameId'] == frame_id]
    
    # List to hold track traces.
    track_traces = []
    
    for idx, track in frame_tracks.iterrows():
        # Get the list of hits from the track.
        hits = track['hits']
        hits_df = pd.DataFrame(hits, columns=['x', 'y', 'z'])
        # Flip axes: New X = original Z, New Y = original Y, New Z = original X.
        x_vals = hits_df['z']
        y_vals = hits_df['y']
        z_vals = hits_df['x']
        
        # Set track_color and legend label based on mc_pid.
        if track['mc_pid'] == -11:
            track_color = 'red'
            legend_label = f"Reco Positron {track['mc_tid']}"
        elif track['mc_pid'] == 11:
            track_color = 'blue'
            legend_label = f"Reco Electron {track['mc_tid']}"
        else:
            track_color = next(color_cycle)
            legend_label = f"Noise"
        
        hover_text_list = []
        for index, hit in hits_df.iterrows():
            hover_text_list.append(
                f"X: {hit['x']:.2f} mm<br>"
                f"Y: {hit['y']:.2f} mm<br>"
                f"Z: {hit['z']:.2f} mm<br>"
                f"mc_tid: {track['mc_tid']}<br>"
                f"mc_pid: {track['mc_pid']}<br>"
                f"mc_type: {track['mc_type']}<br>"
                f"mc_p: {track['mc_p']}<br>"
                f"mc_pt: {track['mc_pt']}<br>"
                f"Hit {index+1} of {len(hits)}"
            )
        
        # Visible trace only (since you no longer want invisible traces)
        visible_trace = go.Scatter3d(
            x=x_vals,
            y=y_vals,
            z=z_vals,
            mode='lines+markers',
            line=dict(color=track_color, width=3),
            marker=dict(size=6, symbol='circle', color=track_color),
            name=legend_label,
            hoverinfo='text',
            text=hover_text_list
        )
        
        track_traces.append(visible_trace)
    
    all_traces = wireframe_traces + track_traces
    layout = go.Layout(
        title=f'Reconstructed Tracks for Frame ID: {frame_id}',
        scene=dict(
            xaxis=dict(title='Z (mm)', autorange=True),
            yaxis=dict(title='Y (mm)', autorange=True),
            zaxis=dict(title='X (mm)', autorange=True),
            aspectmode='data'
        ),
        legend=dict(itemsizing='constant')
    )
    
    fig = go.Figure(data=all_traces, layout=layout)
    pyo.plot(fig, filename=f'frame_{frame_id}_reco_tracks_visualization.html')

