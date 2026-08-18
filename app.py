import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.colors import hex_to_rgb
from streamlit_plotly_events import plotly_events


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="K-Means Explorer",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
    <style>

    .block-container {
        padding-top: 0.6rem;
        padding-bottom: 0.2rem;
        max-width: 1500px;
    }

    .main-title {
        text-align: center;
        font-size: 34px;
        font-weight: 800;
        margin: 0;
    }

    .subtitle {
        text-align: center;
        font-size: 15px;
        color: #666;
        margin-bottom: 8px;
    }

    .status-box {
        padding: 9px 14px;
        border-radius: 12px;
        text-align: center;
        font-size: 15px;
        font-weight: 600;
        margin-bottom: 8px;
    }

    .iteration-box {
        padding: 8px;
        border-radius: 12px;
        text-align: center;
        background: #f3f3f3;
        font-size: 19px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .info-card {
        padding: 10px;
        border-radius: 12px;
        background: #f5f5f5;
        margin-bottom: 7px;
        text-align: center;
    }

    .info-title {
        font-size: 12px;
        color: #666;
    }

    .info-value {
        font-size: 22px;
        font-weight: 700;
    }

    div.stButton > button {
        border-radius: 10px;
        font-weight: 600;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# TITLE
# ============================================================

st.markdown(
    '<div class="main-title">🎨 K-Means Explorer</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Watch points discover their own groups — or choose where the groups begin'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# CONSTANTS
# ============================================================

X_MIN = 0
X_MAX = 100

Y_MIN = 0
Y_MAX = 100

MAX_ITERATIONS = 20

COLORS = [
    "#FF6B6B",
    "#4D96FF",
    "#6BCB77",
    "#B983FF",
    "#FFB84C",
    "#00B8A9"
]

CLUSTER_NAMES = [
    "Group 1",
    "Group 2",
    "Group 3",
    "Group 4",
    "Group 5",
    "Group 6"
]


# ============================================================
# SESSION STATE
# ============================================================

defaults = {
    "points": None,
    "n_points": 100,
    "n_clusters": 3,
    "iteration": 0,
    "centroids": None,
    "initial_centroids": None,
    "centroid_history": None,
    "label_history": None,
    "mode": "Random centroids",
    "selecting_centroids": False,
    "selected_centroids": [],
    "experiment_ready": False,
    "click_key": 0
}

for key, value in defaults.items():

    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# DATA GENERATION
# ============================================================

def generate_points(n):

    return np.random.uniform(
        5,
        95,
        size=(n, 2)
    )


# ============================================================
# RANDOM INITIAL CENTROIDS
# ============================================================

def random_centroids(points, n_clusters):

    indices = np.random.choice(
        len(points),
        size=n_clusters,
        replace=False
    )

    return points[indices].copy()


# ============================================================
# K-MEANS ASSIGNMENT
# ============================================================

def assign_clusters(
    points,
    centroids
):

    distances = np.sqrt(
        (
            points[:, None, :]
            - centroids[None, :, :]
        ) ** 2
    ).sum(axis=2)

    return np.argmin(
        distances,
        axis=1
    )


# ============================================================
# UPDATE CENTROIDS
# ============================================================

def move_centroids(
    points,
    labels,
    centroids
):

    new_centroids = centroids.copy()

    for cluster_id in range(
        len(centroids)
    ):

        members = points[
            labels == cluster_id
        ]

        if len(members) > 0:

            new_centroids[
                cluster_id
            ] = members.mean(axis=0)

    return new_centroids


# ============================================================
# RUN K-MEANS
# ============================================================

def run_kmeans(
    points,
    initial_centroids
):

    centroids = initial_centroids.copy()

    centroid_history = [
        centroids.copy()
    ]

    label_history = []

    for _ in range(
        MAX_ITERATIONS
    ):

        labels = assign_clusters(
            points,
            centroids
        )

        label_history.append(
            labels.copy()
        )

        new_centroids = move_centroids(
            points,
            labels,
            centroids
        )

        centroid_history.append(
            new_centroids.copy()
        )

        centroids = new_centroids

    return (
        centroid_history,
        label_history
    )


# ============================================================
# START RANDOM EXPERIMENT
# ============================================================

def start_random_experiment():

    points = generate_points(
        st.session_state.n_points
    )

    centroids = random_centroids(
        points,
        st.session_state.n_clusters
    )

    history, labels = run_kmeans(
        points,
        centroids
    )

    st.session_state.points = points

    st.session_state.initial_centroids = centroids

    st.session_state.centroid_history = history

    st.session_state.label_history = labels

    st.session_state.centroids = centroids

    st.session_state.iteration = 0

    st.session_state.selected_centroids = []

    st.session_state.selecting_centroids = False

    st.session_state.experiment_ready = True

    st.session_state.click_key += 1


# ============================================================
# START MANUAL CENTROID SELECTION
# ============================================================

def start_manual_selection():

    st.session_state.points = generate_points(
        st.session_state.n_points
    )

    st.session_state.selected_centroids = []

    st.session_state.selecting_centroids = True

    st.session_state.experiment_ready = False

    st.session_state.iteration = 0

    st.session_state.click_key += 1


# ============================================================
# FINISH MANUAL EXPERIMENT
# ============================================================

def finish_manual_experiment():

    centroids = np.array(
        st.session_state.selected_centroids
    )

    history, labels = run_kmeans(
        st.session_state.points,
        centroids
    )

    st.session_state.initial_centroids = centroids

    st.session_state.centroid_history = history

    st.session_state.label_history = labels

    st.session_state.centroids = centroids

    st.session_state.iteration = 0

    st.session_state.selecting_centroids = False

    st.session_state.experiment_ready = True

    st.session_state.click_key += 1


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("## 🎮 Experiment")

    st.markdown("### 1. Number of points")

    point_slider = st.slider(
        "Points",
        20,
        300,
        st.session_state.n_points,
        10
    )

    st.session_state.n_points = point_slider

    st.markdown("### 2. Number of clusters")

    cluster_slider = st.slider(
        "Clusters",
        2,
        6,
        st.session_state.n_clusters,
        1
    )

    st.session_state.n_clusters = cluster_slider

    st.divider()

    st.markdown("### 3. Choose centroid initialization")

    mode = st.radio(
        "How should the starting centroids be selected?",
        [
            "🎲 Random centroids",
            "👆 Place them myself"
        ]
    )

    st.divider()

    if mode == "🎲 Random centroids":

        if st.button(
            "✨ Generate & Start",
            use_container_width=True
        ):

            start_random_experiment()

    else:

        if st.button(
            "👆 Place Centroids",
            use_container_width=True
        ):

            start_manual_selection()

    st.divider()

    st.markdown("### 💡 What to observe")

    st.caption(
        """
        Watch the coloured regions change as the
        centroids move.

        Try choosing very different starting
        positions and compare the final result.
        """
    )


# ============================================================
# MANUAL CENTROID SELECTION MESSAGE
# ============================================================

if st.session_state.selecting_centroids:

    required = st.session_state.n_clusters

    selected = len(
        st.session_state.selected_centroids
    )

    st.markdown(
        f"""
        <div class="status-box"
        style="background:#fff4d6;">
        👆 Click on the plot to place centroid
        <b>{selected + 1}</b> of <b>{required}</b>
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# CREATE EMPTY DATASET IF NECESSARY
# ============================================================

if st.session_state.points is None:

    st.session_state.points = generate_points(
        st.session_state.n_points
    )


# ============================================================
# CURRENT POINTS
# ============================================================

points = st.session_state.points


# ============================================================
# CURRENT ITERATION
# ============================================================

iteration = st.session_state.iteration


# ============================================================
# CURRENT CENTROIDS AND LABELS
# ============================================================

if st.session_state.experiment_ready:

    if iteration == 0:

        current_centroids = (
            st.session_state
            .centroid_history[0]
        )

        current_labels = None

    else:

        current_centroids = (
            st.session_state
            .centroid_history[
                iteration
            ]
        )

        current_labels = (
            st.session_state
            .label_history[
                iteration - 1
            ]

        )

else:

    current_centroids = np.array(
        st.session_state.selected_centroids
    )

    current_labels = None


# ============================================================
# CREATE COLOUR BACKGROUND
# ============================================================

def create_background(
    centroids,
    n_clusters
):

    resolution = 80

    x = np.linspace(
        X_MIN,
        X_MAX,
        resolution
    )

    y = np.linspace(
        Y_MIN,
        Y_MAX,
        resolution
    )

    xx, yy = np.meshgrid(
        x,
        y
    )

    grid = np.column_stack(
        [
            xx.ravel(),
            yy.ravel()
        ]
    )

    labels = assign_clusters(
        grid,
        centroids
    )

    labels = labels.reshape(
        xx.shape
    )

    return xx, yy, labels


# ============================================================
# PLOT FUNCTION
# ============================================================

def create_plot():

    fig = go.Figure()

    # --------------------------------------------------------
    # COLOUR BACKGROUND
    # --------------------------------------------------------

    if (
        st.session_state.experiment_ready
        and len(current_centroids) > 0
    ):

        xx, yy, region_labels = (
            create_background(
                current_centroids,
                st.session_state.n_clusters
            )
        )

        # Add each cluster's soft background
        for cluster_id in range(
            st.session_state.n_clusters
        ):

            mask = (
                region_labels
                == cluster_id
            )

            rgb = hex_to_rgb(
                COLORS[cluster_id]
            )

            # Very light version of cluster colour
            rgba = (
                f"rgba({rgb[0]},"
                f"{rgb[1]},"
                f"{rgb[2]},"
                f"0.12)"
            )

            z = np.where(
                mask,
                1,
                np.nan
            )

            fig.add_trace(
                go.Heatmap(
                    x=xx[0],
                    y=yy[:, 0],
                    z=z,
                    colorscale=[
                        [0, rgba],
                        [1, rgba]
                    ],
                    showscale=False,
                    hoverinfo="skip"
                )
            )

    # --------------------------------------------------------
    # DATA POINTS
    # --------------------------------------------------------

    if current_labels is None:

        fig.add_trace(
            go.Scatter(
                x=points[:, 0],
                y=points[:, 1],
                mode="markers",
                marker=dict(
                    size=9,
                    color="#777777",
                    opacity=0.8,
                    line=dict(
                        width=1,
                        color="white"
                    )
                ),
                name="Random points",
                hovertemplate=(
                    "X: %{x:.1f}"
                    "<br>Y: %{y:.1f}"
                    "<extra></extra>"
                )
            )
        )

    else:

        for cluster_id in range(
            st.session_state.n_clusters
        ):

            mask = (
                current_labels
                == cluster_id
            )

            if np.any(mask):

                fig.add_trace(
                    go.Scatter(
                        x=points[mask, 0],
                        y=points[mask, 1],
                        mode="markers",
                        marker=dict(
                            size=9,
                            color=COLORS[
                                cluster_id
                            ],
                            opacity=0.9,
                            line=dict(
                                width=1,
                                color="white"
                            )
                        ),
                        name=CLUSTER_NAMES[
                            cluster_id
                        ],
                        hovertemplate=(
                            "X: %{x:.1f}"
                            "<br>Y: %{y:.1f}"
                            "<br>"
                            + CLUSTER_NAMES[
                                cluster_id
                            ]
                            + "<extra></extra>"
                        )
                    )
                )

    # --------------------------------------------------------
    # CENTROIDS
    # --------------------------------------------------------

    if len(current_centroids) > 0:

        for cluster_id, centroid in enumerate(
            current_centroids
        ):

            fig.add_trace(
                go.Scatter(
                    x=[centroid[0]],
                    y=[centroid[1]],
                    mode="markers+text",
                    marker=dict(
                        size=22,
                        symbol="x",
                        color=COLORS[
                            cluster_id
                        ],
                        line=dict(
                            width=3,
                            color="#222222"
                        )
                    ),
                    text=[
                        f"C{cluster_id + 1}"
                    ],
                    textposition="top center",
                    textfont=dict(
                        size=12,
                        color="#222222"
                    ),
                    name=f"Centroid {cluster_id + 1}",
                    hovertemplate=(
                        f"Centroid {cluster_id + 1}"
                        "<br>X: %{x:.1f}"
                        "<br>Y: %{y:.1f}"
                        "<extra></extra>"
                    )
                )
            )

    # --------------------------------------------------------
    # LAYOUT
    # --------------------------------------------------------

    title = (
        "Choose your centroids"
        if st.session_state.selecting_centroids
        else
        f"K-Means — Iteration {iteration} / "
        f"{MAX_ITERATIONS}"
    )

    fig.update_layout(
        title=dict(
            text=title,
            x=0.5,
            font=dict(
                size=18
            )
        ),
        xaxis=dict(
            title="X",
            range=[X_MIN, X_MAX],
            fixedrange=True,
            showgrid=True,
            gridcolor="rgba(0,0,0,0.08)"
        ),
        yaxis=dict(
            title="Y",
            range=[Y_MIN, Y_MAX],
            fixedrange=True,
            showgrid=True,
            gridcolor="rgba(0,0,0,0.08)",
            scaleanchor="x",
            scaleratio=1
        ),
        height=570,
        margin=dict(
            l=35,
            r=20,
            t=50,
            b=35
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.01,
            xanchor="center",
            x=0.5
        ),
        hovermode="closest"
    )

    return fig


# ============================================================
# MAIN SCREEN
# ============================================================

plot_col, info_col = st.columns(
    [3.4, 1]
)


# ============================================================
# PLOT
# ============================================================

with plot_col:

    fig = create_plot()

    # --------------------------------------------------------
    # ENABLE CLICKING ONLY DURING MANUAL SELECTION
    # --------------------------------------------------------

    if st.session_state.selecting_centroids:

        clicked = plotly_events(
            fig,
            click_event=True,
            hover_event=False,
            select_event=False,
            override_height=570,
            key=f"centroid_click_{st.session_state.click_key}"
        )

        if clicked:

            click = clicked[-1]

            x = click.get("x")
            y = click.get("y")

            if (
                x is not None
                and y is not None
            ):

                already = (
                    len(
                        st.session_state
                        .selected_centroids
                    )
                    >= st.session_state.n_clusters
                )

                if not already:

                    st.session_state.selected_centroids.append(
                        [float(x), float(y)]
                    )

                    st.session_state.click_key += 1

                    # If enough centroids selected
                    if (
                        len(
                            st.session_state
                            .selected_centroids
                        )
                        ==
                        st.session_state.n_clusters
                    ):

                        finish_manual_experiment()

                    st.rerun()

    else:

        st.plotly_chart(
            fig,
            use_container_width=True,
            config={
                "displayModeBar": False
            }
        )


# ============================================================
# INFORMATION PANEL
# ============================================================

with info_col:

    # --------------------------------------------------------
    # MANUAL MODE
    # --------------------------------------------------------

    if st.session_state.selecting_centroids:

        st.markdown("### 👆 Place centroids")

        selected = len(
            st.session_state.selected_centroids
        )

        required = st.session_state.n_clusters

        st.markdown(
            f"""
            <div class="info-card">
                <div class="info-title">
                    Centroids selected
                </div>

                <div class="info-value">
                    {selected} / {required}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.write(
            "Click directly on the plot to place "
            "each centroid."
        )

        if selected > 0:

            st.write("Your centroids:")

            for i, c in enumerate(
                st.session_state.selected_centroids
            ):

                st.markdown(
                    f"""
                    <div style="
                        padding:5px;
                        margin:3px;
                        border-radius:7px;
                        background:{COLORS[i]};
                        color:white;
                        text-align:center;
                        font-size:12px;
                        font-weight:600;
                    ">
                    C{i + 1}: ({c[0]:.1f}, {c[1]:.1f})
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        if selected == required:

            st.success(
                "All centroids selected! "
                "The experiment will begin."
            )

    # --------------------------------------------------------
    # NORMAL MODE
    # --------------------------------------------------------

    else:

        st.markdown("### 📌 Experiment")

        st.markdown(
            f"""
            <div class="info-card">
                <div class="info-title">
                    Data points
                </div>
                <div class="info-value">
                    {st.session_state.n_points}
                </div>
            </div>

            <div class="info-card">
                <div class="info-title">
                    Clusters
                </div>
                <div class="info-value">
                    {st.session_state.n_clusters}
                </div>
            </div>

            <div class="info-card">
                <div class="info-title">
                    Iteration
                </div>
                <div class="info-value">
                    {iteration} / {MAX_ITERATIONS}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.divider()

        # ----------------------------------------------------
        # CURRENT STATUS
        # ----------------------------------------------------

        if iteration == 0:

            st.info(
                "👀 Random points are ready. "
                "The centroids are in their starting positions."
            )

        elif iteration < 5:

            st.info(
                "🌱 The first groups are emerging. "
                "Watch the centroids move."
            )

        elif iteration < 12:

            st.info(
                "🧩 The groups are becoming clearer. "
                "Notice how the coloured regions change."
            )

        elif iteration < 20:

            st.info(
                "🎨 The clusters are getting stable."
            )

        else:

            st.success(
                "✨ The centroids have converged."
            )


# ============================================================
# ITERATION SLIDER
# ============================================================

if st.session_state.experiment_ready:

    st.divider()

    st.markdown(
        f"""
        <div class="iteration-box">
        🔄 Iteration {iteration} / {MAX_ITERATIONS}
        </div>
        """,
        unsafe_allow_html=True
    )

    c1, c2, c3 = st.columns(
        [1, 7, 1]
    )

    with c1:

        if st.button(
            "◀",
            use_container_width=True
        ):

            st.session_state.iteration = max(
                0,
                iteration - 1
            )

            st.rerun()

    with c2:

        new_iteration = st.slider(
            "Move through the iterations",
            0,
            MAX_ITERATIONS,
            iteration,
            1,
            label_visibility="collapsed"
        )

        if new_iteration != iteration:

            st.session_state.iteration = (
                new_iteration
            )

            st.rerun()

    with c3:

        if st.button(
            "▶",
            use_container_width=True
        ):

            st.session_state.iteration = min(
                MAX_ITERATIONS,
                iteration + 1
            )

            st.rerun()


# ============================================================
# BOTTOM EXPLANATION
# ============================================================

st.divider()

if iteration == 0:

    message = (
        "**Start:** The points are completely random. "
        "The X marks show the initial centroid positions."
    )

elif iteration == 1:

    message = (
        "**Iteration 1:** Each point is assigned to "
        "the centroid it is closest to. The colours "
        "show those assignments."
    )

elif iteration < MAX_ITERATIONS:

    message = (
        f"**Iteration {iteration}:** Points are assigned "
        "to their nearest centroid, then each centroid "
        "moves toward the centre of its assigned points."
    )

else:

    message = (
        "**Converged:** The centroids have settled into "
        "positions where the cluster assignments are stable."
    )

st.markdown(
    f"""
    <div class="status-box"
    style="background:#f5f5f5;">
    {message}
    </div>
    """,
    unsafe_allow_html=True
)
