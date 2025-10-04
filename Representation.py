from gini import Gini
from extraction import extraction_labels
import graphviz
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np
from scipy import stats
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_tree(root_node):

    dot = graphviz.Digraph()
    dot.attr(rankdir='TB')  # Top to Bottom
    dot.attr('graph', fontsize='16', fontname='Arial')
    dot.attr('node', fontname='Arial', fontsize='12')
    dot.attr('edge', fontname='Arial', fontsize='10')
    
    node_counter = [0]

    def get_class_colors(all_labels):
        unique_labels = sorted(set(all_labels))
        n = len(unique_labels)
        cmap = cm.get_cmap('tab10', n)
        color_map = {}
        for i, lab in enumerate(unique_labels):
            rgb = cmap(i)[:3]
            hex_color = mcolors.to_hex(rgb)
            color_map[lab] = hex_color
        return color_map

    def collect_labels(node):
        labels = []
        if node is None:
            return labels
        if hasattr(node, "records"):
            labels.extend(extraction_labels(node.records))
        if hasattr(node, "left_child"):
            labels.extend(collect_labels(node.left_child))
        if hasattr(node, "right_child"):
            labels.extend(collect_labels(node.right_child))
        return labels

    all_labels = collect_labels(root_node)
    color_map = get_class_colors(all_labels)

    
    def add_node_to_graph(node, parent_id=None, is_left_child=False):
        if node is None:
            return

        current_id = f"node_{node_counter[0]}"
        node_counter[0] += 1

        if node.type == 'leaf':
            samples = len(node.records) if hasattr(node, 'records') else 0
            gini_val = Gini(extraction_labels(node.records)) if hasattr(node, 'records') and node.records else 0

            label = f"Class = {node.label}\\nSamples = {samples}\\nGini = {gini_val:.3f}"
            fill_color = color_map.get(node.label, "#D3D3D3")

            peripheries = "2" if gini_val == 0 else "1"
            penwidth = "2" if gini_val == 0 else "1"

            dot.node(current_id, label,
                     shape="ellipse",
                     style="filled",
                     fillcolor=fill_color,
                     color="black",
                     peripheries=peripheries,
                     penwidth=penwidth)
        else:
            samples = len(node.records) if hasattr(node, 'records') else 0
            gini_val = Gini(extraction_labels(node.records)) if hasattr(node, 'records') and node.records else 0
            pattern_str = ", ".join(str(x) for x in node.split_feature) if node.split_feature else "[]"

            label = f"Contains [{pattern_str}]?\\nSamples = {samples}\\nGini = {gini_val:.3f}"

            dot.node(current_id, label,
                     shape='box',
                     style='filled',
                     fillcolor='#ADD8E6',
                     color='black')

        if parent_id is not None:
            edge_label = "Yes" if is_left_child else "No"
            edge_color = 'green' if is_left_child else 'red'
            dot.edge(parent_id, current_id, label=edge_label, color=edge_color)

        if node.left_child is not None:
            add_node_to_graph(node.left_child, current_id, True)
        if node.right_child is not None:
            add_node_to_graph(node.right_child, current_id, False)

        return current_id

    add_node_to_graph(root_node)

    return dot




def plot_gaussian_with_boundaries(mean: float, std: float, boundaries: np.ndarray):
    """
    Create a Plotly visualization of Gaussian distribution with colored bins
    
    Input:
        - mean : float
        - std : float
        - boundaries : np.ndarray
    """
    # Create x values for the Gaussian curve
    x_min = mean - 4 * std
    x_max = mean + 4 * std
    x = np.linspace(x_min, x_max, 1000)
    y = stats.norm.pdf(x, loc=mean, scale=std)
    
    fig = go.Figure()
    
    colors = ['rgba(255, 99, 132, 0.5)', 'rgba(54, 162, 235, 0.5)', 
              'rgba(255, 206, 86, 0.5)', 'rgba(75, 192, 192, 0.5)',
              'rgba(153, 102, 255, 0.5)', 'rgba(255, 159, 64, 0.5)',
              'rgba(199, 199, 199, 0.5)', 'rgba(83, 102, 255, 0.5)',
              'rgba(255, 99, 255, 0.5)', 'rgba(99, 255, 132, 0.5)']
    
    # Fill areas between boundaries
    n_bins = len(boundaries) - 1
    for i in range(n_bins):
        lower = boundaries[i]
        upper = boundaries[i + 1]
        
        # Create x values for bin i
        x_bin = x[(x >= lower) & (x <= upper)]
        y_bin = stats.norm.pdf(x_bin, loc=mean, scale=std)
        
        # Add filled area
        fig.add_trace(go.Scatter(x=np.concatenate([x_bin, x_bin[::-1]]),
                                 y=np.concatenate([y_bin, np.zeros(len(y_bin))]),
                                 fill='toself',
                                 fillcolor=colors[i % len(colors)],
                                 line=dict(color='rgba(255,255,255,0)'),
                                 name=f'Bin {i+1}: [{lower:.2f}, {upper:.2f}]',
                                 hoverinfo='name'
                                ))
    
    # Add the Gaussian curve
    fig.add_trace(go.Scatter(x=x,
                             y=y,
                             mode='lines',
                             line=dict(color='black', width=2),
                             name='Gaussian curve',
                             hovertemplate='x: %{x:.2f}<br>: %{y:.2f}<extra></extra>'
                            ))
                        
                        
    # Add vertical lines at boundaries
    for i, boundary in enumerate(boundaries):
        if not np.isinf(boundary):  # Skip infinite boundaries
            fig.add_vline(x=boundary,
                          line=dict(color='red', width=1, dash='dash'),
                          annotation_text=f'{boundary:.2f}',
                          annotation_position='top'
                         )
    
    fig.update_layout(title=f'Gaussian Distribution (μ={mean:.2f}, σ={std:.2f}) with {n_bins} Equal-Area Bins',
                      xaxis_title='Value',
                      yaxis_title='Probability Density',
                      hovermode='x unified',
                      showlegend=True,
                      height=600,
                      template='plotly_white'
                     )
    
    return fig



def plot_timeseries_comparison(original: np.ndarray, translated: np.ndarray, boundaries: np.ndarray, patient_indices = None, X_RLR = None):
    """
    Plot original and translated time series side by side, with optional RLR compression
    
    Input:
        - original : np.ndarray
        - translated : np.ndarray
        - boundaries : np.ndarray
        - patient_indices : list or None
        - X_RLR : list of lists or None
    """
    
    if patient_indices is None:
        patient_indices = list(range(min(1, len(original))))
        
    n_sequences = len(patient_indices)
    
    if X_RLR is not None:
        n_cols = 3  
    else:
        n_cols = 2
    
    # Create subplot titles
    if X_RLR is not None:
        subplot_titles = []
        for i in range(n_sequences):
            subplot_titles.extend([
                f'Original Patient {patient_indices[i]}',
                f'Translated Patient {patient_indices[i]}',
                f'RLR Patient {patient_indices[i]}'
            ])
    else:
        subplot_titles = []
        for i in range(n_sequences):
            for j in range(2):
                if j == 0:
                    title = f"Original Patient {patient_indices[i]}"
                else:
                    title = f"Translated Patient {patient_indices[i]}"
                subplot_titles.append(title)
    
    
    # Create subplots
    fig = make_subplots(rows=n_sequences, 
                        cols=n_cols,
                        subplot_titles=subplot_titles,
                        vertical_spacing=0.15,
                        horizontal_spacing=0.08
                        )
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    for plot_idx, patient_idx in enumerate(patient_indices):
        orig_seq = original[patient_idx].flatten()
        trans_seq = translated[patient_idx].flatten()
        x_values_orig = list(range(len(orig_seq)))
        
        # Plot original time series
        fig.add_trace(go.Scatter(x=x_values_orig,
                                 y=orig_seq,
                                 mode='lines+markers',
                                 name=f'Patient {patient_idx} Original',
                                 line=dict(color=colors[plot_idx % len(colors)], width=2),
                                 marker=dict(size=4),
                                 showlegend=True,
                                 legendgroup=f'patient{patient_idx}',
                                 hovertemplate='Time: %{x}<br>Value: %{y:.2f}<extra></extra>'
                                ),
                      row=plot_idx+1, 
                      col=1
                    )
        
        # Horizontal lines for boundaries in original plot
        for boundary in boundaries[1:-1]:  # Skip -inf and +inf
            fig.add_hline(y=boundary,
                          line=dict(color='red', width=1, dash='dash'),
                          opacity=0.3,
                          row=plot_idx+1,
                          col=1
                        )
        
        # Plot translated time series (as step function)
        fig.add_trace(go.Scatter(x=x_values_orig,
                                y=trans_seq,
                                mode='lines+markers',
                                name=f'Patient {patient_idx} Translated',
                                line=dict(color=colors[plot_idx % len(colors)], width=2, shape='hv'),
                                marker=dict(size=8),
                                showlegend=True,
                                legendgroup=f'patient{patient_idx}',
                                hovertemplate='Time: %{x}<br>Bin: %{y}<extra></extra>'
                                ),
                      row=plot_idx+1,
                      col=2
                      )
        
        
        # Plot RLR
        if X_RLR is not None:
            RLR_seq = X_RLR[patient_idx]
            x_values_RLR = list(range(len(RLR_seq)))
            
            fig.add_trace(go.Scatter(x=x_values_RLR,
                                    y=RLR_seq,
                                    mode='lines+markers',
                                    name=f'Patient {patient_idx} RLR',
                                    line=dict(color=colors[plot_idx % len(colors)], width=2, shape='hv'),
                                    marker=dict(size=10),
                                    showlegend=True,
                                    legendgroup=f'patient{patient_idx}',
                                    hovertemplate='Time: %{x}<br>Bin: %{y}<extra></extra>'
                                ),
                            row=plot_idx+1,
                            col=3
                        )
            
            # Update y-axis for RLR plot
            fig.update_yaxes(title_text='Bin Index',
                            tickmode='linear',
                            tick0=1,
                            dtick=1,
                            row=plot_idx+1,
                            col=3
                            )
            
            # Update x-axis for RLR plot
            fig.update_xaxes(title_text='Time Step', row=plot_idx+1, col=3)
        
        
        # Update y-axis for translated plot to show integer bins
        fig.update_yaxes(title_text='Bin Index',
                        tickmode='linear',
                        tick0=1,
                        dtick=1,
                        row=plot_idx+1,
                        col=2
                        )
        
        # Update y-axis for original plot
        fig.update_yaxes(
            title_text='Value',
            row=plot_idx+1,
            col=1
        )
        
        # Update x-axes
        fig.update_xaxes(title_text='Time Step', row=plot_idx+1, col=1)
        fig.update_xaxes(title_text='Time Step', row=plot_idx+1, col=2)
    
    # Update overall layout
    if X_RLR is not None:
        title_suffix = ' vs RLR'  
    else:
        title_suffix = ''
        
    fig.update_layout(
        title_text=f'Time Series: Original Values vs Translated Bins{title_suffix}',
        height=300 * n_sequences,
        showlegend=False,
        hovermode='closest',
        template='plotly_white'
    )
    
    return fig
