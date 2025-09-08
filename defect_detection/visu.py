import pandas as pd 
import numpy as np
import plotly.graph_objects as go


def create_bounding_box_trace(xmin, xmax, ymin, ymax, zmin, zmax, color='black', width=3):
    """
    Returns a single Scatter3d trace that draws the 12 edges of a rectangular box
    spanning from (xmin, ymin, zmin) to (xmax, ymax, zmax).
    """
    edges = [
        # Bottom face (z = zmin)
        [(xmin, ymin, zmin), (xmax, ymin, zmin)],
        [(xmax, ymin, zmin), (xmax, ymax, zmin)],
        [(xmax, ymax, zmin), (xmin, ymax, zmin)],
        [(xmin, ymax, zmin), (xmin, ymin, zmin)],
        # Top face (z = zmax)
        [(xmin, ymin, zmax), (xmax, ymin, zmax)],
        [(xmax, ymin, zmax), (xmax, ymax, zmax)],
        [(xmax, ymax, zmax), (xmin, ymax, zmax)],
        [(xmin, ymax, zmax), (xmin, ymin, zmax)],
        # Vertical edges
        [(xmin, ymin, zmin), (xmin, ymin, zmax)],
        [(xmax, ymin, zmin), (xmax, ymin, zmax)],
        [(xmax, ymax, zmin), (xmax, ymax, zmax)],
        [(xmin, ymax, zmin), (xmin, ymax, zmax)]
    ]
    
    xs, ys, zs = [], [], []
    for edge in edges:
        (x0, y0, z0), (x1, y1, z1) = edge
        xs += [x0, x1, None]
        ys += [y0, y1, None]
        zs += [z0, z1, None]
    
    return go.Scatter3d(
        x=xs,
        y=ys,
        z=zs,
        mode='lines',
        line=dict(color=color, width=width),
        showlegend=False,
        hoverinfo='none'  # no hover for edges
    )

    pass

def create_xyz_axes(center, axis_length=50):
    """
    Returns a list of 3 Scatter3d traces: X (red), Y (green), Z (blue) axes.
    center: (cx, cy, cz) is where the axes intersect.
    axis_length: length of each axis line from the center.
    """
    cx, cy, cz = center
    
    # X-axis in red
    x_axis = go.Scatter3d(
        x=[cx, cx + axis_length],
        y=[cy, cy],
        z=[cz, cz],
        mode='lines',
        line=dict(color='red', width=5),
        showlegend=False,
        hoverinfo='none',
        name='X-axis'
    )
    
    # Y-axis in green
    y_axis = go.Scatter3d(
        x=[cx, cx],
        y=[cy, cy + axis_length],
        z=[cz, cz],
        mode='lines',
        line=dict(color='green', width=5),
        showlegend=False,
        hoverinfo='none',
        name='Y-axis'
    )

    # Z-axis in blue
    z_axis = go.Scatter3d(
        x=[cx, cx],
        y=[cy, cy],
        z=[cz, cz + axis_length],
        mode='lines',
        line=dict(color='blue', width=5),
        showlegend=False,
        hoverinfo='none',
        name='Z-axis'
    )
    
    return [x_axis, y_axis, z_axis]

def create_defect_visualization(df, X=15, num_dense=100, num_sparse=200, bounding_box_max=225.28, camera_eye=(1.5, 1.5, 1.5)):
    """
    Create an interactive 3D defect visualization from an XYZ file.
    
    Parameters:
        xyz_file (str): Path to the XYZ file.
        X (float): Range added to the minimum reconstruction error for the dense segment.
        num_dense (int): Number of threshold points in the dense segment.
        num_sparse (int): Number of threshold points in the sparse segment.
        bounding_box_max (float): Maximum coordinate value for the bounding box (assumes a cubic lattice from 0 to bounding_box_max).
        camera_eye (tuple): Camera eye position for the 3D scene.
        
    Returns:
        go.Figure: A Plotly figure object with the interactive defect visualization.
    """    
    # ---------------------------
    # Step 3. Define thresholds for reconstruction error.
    min_val = df['ReconError'].min()
    max_val = df['ReconError'].max()
    dense_part = np.linspace(min_val, min_val + X, num_dense, endpoint=False)
    sparse_part = np.linspace(min_val + X, max_val, num_sparse)
    thresholds = np.unique(np.concatenate([dense_part, sparse_part]))
    print("Number of thresholds =", len(thresholds))
    
    # ---------------------------
    # Step 4. Create animation frames.
    frames = []
    for t in thresholds:
        filtered = df[df['ReconError'] >= t]
        frame = go.Frame(
            data=[go.Scatter3d(
                x=filtered['x'],
                y=filtered['y'],
                z=filtered['z'],
                mode='markers',
                marker=dict(
                    size=3,
                    color=filtered['ReconError'],
                    colorscale='Viridis',
                    opacity=0.5
                ),
                text=filtered['ReconError'].round(2).astype(str)
            )],
            name=str(t)
        )
        frames.append(frame)
    
    # ---------------------------
    # Step 5. Create initial trace and additional traces.
    initial_threshold = thresholds[0]
    initial_data = df[df['ReconError'] >= initial_threshold]
    
    initial_trace = go.Scatter3d(
        x=initial_data['x'],
        y=initial_data['y'],
        z=initial_data['z'],
        mode='markers',
        marker=dict(
            size=3,
            color=initial_data['ReconError'],
            colorscale='Viridis',
            opacity=0.5
        ),
        text=initial_data['ReconError'].round(2).astype(str),
        name='Atoms'
    )
    
    box_trace = create_bounding_box_trace(
        xmin=0, xmax=bounding_box_max,
        ymin=0, ymax=bounding_box_max,
        zmin=0, zmax=bounding_box_max,
        color='black', width=3
    )
    
    box_center = (bounding_box_max / 2, bounding_box_max / 2, bounding_box_max / 2)
    xyz_axes_traces = create_xyz_axes(center=box_center, axis_length=20)
    
    # ---------------------------
    # Step 6. Build the Plotly figure.
    axis_range = [0, bounding_box_max]
    fig = go.Figure(
        data=[initial_trace, box_trace] + xyz_axes_traces,
        layout=go.Layout(
            title=dict(text="Interactive Defect Visualization", font=dict(size=24, color='black')),
            width=1300,
            height=900,
            uirevision='fixed',
            scene=dict(
                xaxis=dict(visible=False, range=axis_range, autorange=False),
                yaxis=dict(visible=False, range=axis_range, autorange=False),
                zaxis=dict(visible=False, range=axis_range, autorange=False),
                bgcolor="rgba(0,0,0,0)"
            ),
            updatemenus=[{
                "type": "buttons",
                "buttons": [{
                    "label": "Play",
                    "method": "animate",
                    "args": [None, {"frame": {"duration": 100, "redraw": True},
                                    "transition": {"duration": 0},
                                    "fromcurrent": True,
                                    "mode": "immediate"}]
                }],
                "pad": {"r": 10, "t": 10},
                "showactive": True,
                "x": 0.1,
                "xanchor": "right",
                "y": 0,
                "yanchor": "top"
            }],
            sliders=[{
                "steps": [{
                    "method": "animate",
                    "args": [[str(t)],
                             {"frame": {"duration": 100, "redraw": True},
                              "transition": {"duration": 0},
                              "mode": "immediate"}],
                    "label": f"{t:.2f}"
                } for t in thresholds],
                "active": 0,
                "currentvalue": {"prefix": "Threshold: "},
                "pad": {"b": 10, "t": 50},
                "len": 0.9,
                "x": 0.1,
                "xanchor": "left",
                "y": 0,
                "yanchor": "top"
            }],
            scene_camera=dict(eye=dict(x=camera_eye[0], y=camera_eye[1], z=camera_eye[2]))
        ),
        frames=frames
    )
    
    return fig