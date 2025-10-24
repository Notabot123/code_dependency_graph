import os
import ast
import networkx as nx
import plotly.graph_objects as go
import streamlit as st
from concurrent.futures import ThreadPoolExecutor, as_completed

# --------------------------------------------
# CONFIGURATION
# --------------------------------------------
IGNORE_DIRS = {
    '.git', '__pycache__', 'venv', 'env', 'build', 'dist',
    '.mypy_cache', '.pytest_cache', '.idea', '.vscode'
}

# --------------------------------------------
# DEPENDENCY PARSER
# --------------------------------------------
def get_imports_from_file(filepath):
    """Extract imported modules from a Python file, safely handling encodings."""
    imports = []
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
    except UnicodeDecodeError:
        with open(filepath, "r", encoding="latin-1", errors="ignore") as f:
            content = f.read()

    try:
        node = ast.parse(content, filename=filepath)
    except SyntaxError:
        return imports

    for n in ast.walk(node):
        if isinstance(n, ast.Import):
            for alias in n.names:
                imports.append(alias.name.split('.')[0])
        elif isinstance(n, ast.ImportFrom):
            if n.module:
                imports.append(n.module.split('.')[0])
    return imports


def discover_py_files(base_path):
    """Recursively discover Python files, skipping ignored directories."""
    py_files = []
    for dp, dn, filenames in os.walk(base_path):
        if any(ignored in dp for ignored in IGNORE_DIRS):
            continue
        for f in filenames:
            if f.endswith('.py'):
                py_files.append(os.path.join(dp, f))
    return py_files


def build_dependency_graph(base_path, include_external=False, max_workers=8, progress_callback=None):
    """Build a dependency graph for all Python files in a directory."""
    G = nx.DiGraph()
    py_files = discover_py_files(base_path)
    module_names = {os.path.splitext(os.path.basename(f))[0]: f for f in py_files}

    def process_file(f):
        src_module = os.path.splitext(os.path.basename(f))[0]
        imports = get_imports_from_file(f)
        edges = []
        for imp in imports:
            if imp in module_names:
                edges.append((src_module, imp))
            elif include_external:
                edges.append((src_module, imp))
        return edges

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_file, f): f for f in py_files}
        total = len(futures)
        for i, future in enumerate(as_completed(futures), 1):
            edges = future.result()
            G.add_edges_from(edges)
            if progress_callback and i % 25 == 0:
                progress_callback(i / total)

    return G

# --------------------------------------------
# VISUALIZATION
# --------------------------------------------
def plot_sankey(G, dark_mode=False):
    nodes = list(G.nodes)
    node_idx = {n: i for i, n in enumerate(nodes)}
    sources = [node_idx[u] for u, v in G.edges]
    targets = [node_idx[v] for u, v in G.edges]
    values = [1] * len(sources)

    node_color = "skyblue" if not dark_mode else "lightgray"

    fig = go.Figure(go.Sankey(
        node=dict(label=nodes, pad=20, thickness=20, color=node_color),
        link=dict(source=sources, target=targets, value=values)
    ))
    fig.update_layout(
        title_text="Python Module Dependency Sankey",
        font_size=12,
        paper_bgcolor='black' if dark_mode else 'white',
        plot_bgcolor='black' if dark_mode else 'white',
        font=dict(color='white' if dark_mode else 'black')
    )
    return fig


def plot_network(G, centrality_metric="degree", dark_mode=True):
    """Plot network graph with node size based on centrality."""
    if centrality_metric == "degree":
        centrality = nx.degree_centrality(G)
    elif centrality_metric == "betweenness":
        centrality = nx.betweenness_centrality(G)
    else:
        centrality = {n: 1 for n in G.nodes()}

    pos = nx.spring_layout(G, k=0.8, iterations=50, seed=42)
    edge_x, edge_y = [], []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    node_x, node_y, sizes, texts = [], [], [], []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        # Scale size by centrality
        sizes.append(10 + 60 * centrality[node])
        texts.append(f"{node}<br>Centrality: {centrality[node]:.3f}")

    node_color = 'skyblue' if not dark_mode else 'lightgray'
    edge_color = '#666' if dark_mode else '#888'

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color=edge_color),
        hoverinfo='none', mode='lines'
    ))
    fig.add_trace(go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        text=list(G.nodes()),
        textposition="top center",
        marker=dict(size=sizes, color=node_color, line=dict(width=1, color='white' if dark_mode else 'black')),
        hovertext=texts,
        hoverinfo='text'
    ))

    fig.update_layout(
    title=f"Python Module Dependency Network ({centrality_metric.capitalize()} centrality)",
    showlegend=False,
    paper_bgcolor='black' if dark_mode else 'white',
    plot_bgcolor='black' if dark_mode else 'white',
    font=dict(color='white' if dark_mode else 'black'),
    margin=dict(l=0, r=0, t=40, b=0)
    )

    # Hide axes for a cleaner look
    fig.update_xaxes(showgrid=False, zeroline=False, visible=False)
    fig.update_yaxes(showgrid=False, zeroline=False, visible=False)

    return fig


# --------------------------------------------
# STREAMLIT APP
# --------------------------------------------
st.set_page_config(page_title="Python Dependency Visualizer", layout="wide")
st.title("Python Dependency Visualizer")

st.write("Analyze inter-file import dependencies in your Python project, with centrality-based sizing and dark mode!")

base_path = st.text_input("📁 Enter your project path:", "./")
include_external = st.checkbox("Include external dependencies", value=False)
view_type = st.selectbox("View type", ["Sankey Diagram", "Network Graph"])
centrality_metric = st.selectbox("Node size metric", ["degree", "betweenness", "none"])
dark_mode = st.checkbox("Dark mode (black background)", value=True)
max_workers = st.slider("Max parallel threads", 2, 32, value=min(8, os.cpu_count() or 4))

# Cache the graph so re-renders are instant
@st.cache_data(show_spinner=False)
def cached_graph(base_path, include_external, max_workers):
    progress = st.progress(0)
    return build_dependency_graph(base_path, include_external, max_workers,
                                  progress_callback=lambda p: progress.progress(p))

if st.button("Generate Diagram"):
    if not os.path.isdir(base_path):
        st.error("Invalid directory path.")
    else:
        with st.spinner("Building dependency graph..."):
            G = cached_graph(base_path, include_external, max_workers)

        if len(G.nodes) == 0:
            st.warning("No dependencies found.")
        else:
            st.success(f"✅ Graph built with {len(G.nodes)} modules and {len(G.edges)} dependencies.")
            fig = plot_sankey(G, dark_mode) if view_type == "Sankey Diagram" else plot_network(G, centrality_metric, dark_mode)
            st.plotly_chart(fig, use_container_width=True)
