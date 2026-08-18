import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.colors import hex_to_rgb


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="K-Means Explorer",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .block-container {
        padding-top: 1rem;
        padding-bottom: 0.5rem;
        max-width: 1500px;
    }

    .title {
        text-align: center;
        font-size: 34px;
        font-weight: 800;
        margin-bottom: 0px;
    }

    .subtitle {
        text-align: center;
        font-size: 15px;
        color: #666;
        margin-bottom: 10px;
    }

    .iteration-box {
        text-align: center;
        font-size: 20px;
        font-weight: 700;
        padding: 8px;
        border-radius: 12px;
        background-color: #f4f4f4;
    }

    div.stButton > button {
        border-radius: 10px;
        font-weight: 600;
    }

    section[data-testid="stSidebar"] {
        width: 270px !important;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# TITLE
# ============================================================

st.markdown(
    '<div class="title">🎨 K-Means Explorer</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Watch random points discover their own groups'
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

CLUSTER_COLORS = [
    "#FF6B6B",
    "#4D96FF",
    "#6BCB77",
    "#B983FF",
    "#FFB84C",
    "#00B8A9",
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
    "centroid_history": None,
    "label_history": None,
    "generated": False,
    "centroid_mode": "🎲 Random centroids",
    "manual_centroids": [],
    "manual_selection_active": False,
    "plot_key": 0,
}

for key, value in defaults.items():

    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# RANDOM DATA GENERATION
# ============================================================

def generate_points(n_points):

    """
    Generate completely random points.

    At iteration 0, all points are identical
    in appearance and have no class information.
    """

    return np.random.uniform(
        5,
        95,
        size=(n_points, 2)
    )


# ============================================================
# INITIAL CENTROIDS
# ============================================================

def initialize_centroids(
    points,
    n_clusters
):

    """
    Select random points from the dataset
    as initial centroids.
    """

    indices = np.random.choice(
        len(points),
        size=n_clusters,
        replace=False
    )

    return points[indices].copy()


# ============================================================
# ASSIGN POINTS TO CENTROIDS
# ============================================================

def assign_clusters(
    points,
    centroids
):

    """
    Assign every point to its nearest centroid.

    Euclidean distance is used.
    """

    distances = np.sqrt(
        (
            points[:, np.newaxis, :]
            - centroids[np.newaxis, :, :]
        ) ** 2
    ).sum(axis=2)

    return np.argmin(
        distances,
        axis=1
    )


# ============================================================
# UPDATE CENTROIDS
# ============================================================

def update_centroids(
    points,
    labels,
    centroids
):

    """
    Move each centroid to the mean
    of the points assigned to it.
    """

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
            ] = members.mean(
                axis=0
            )

    return new_centroids


# ============================================================
# RUN K-MEANS
# ============================================================

def run_kmeans(
    points,
    n_clusters,
    max_iterations=20,
    initial_centroids=None
):

    """
    Run K-means and save every iteration.

    centroid_history:
        centroid locations at every step

    label_history:
        cluster assignment at every step
    """

    # --------------------------------------------------------
    # INITIAL CENTROIDS
    # --------------------------------------------------------

    if initial_centroids is None:

        centroids = initialize_centroids(
            points,
            n_clusters
        )

    else:

        centroids = np.array(
            initial_centroids,
            dtype=float
        ).copy()

    centroid_history = [
        centroids.copy()
    ]

    label_history = []

    # --------------------------------------------------------
    # ITERATIONS
    # --------------------------------------------------------

    for _ in range(
        max_iterations
    ):

        # Assign points
        labels = assign_clusters(
            points,
            centroids
        )

        label_history.append(
            labels.copy()
        )

        # Move centroids
        new_centroids = update_centroids(
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
# CREATE EXPERIMENT
# ============================================================

def create_experiment(
    n_points,
    n_clusters,
    initial_centroids=None
):

    points = generate_points(
        n_points
    )

    (
        centroid_history,
        label_history
    ) = run_kmeans(
        points,
        n_clusters,
        MAX_ITERATIONS,
        initial_centroids
    )

    st.session_state.points = points

    st.session_state.n_points = n_points

    st.session_state.n_clusters = n_clusters

    st.session_state.centroid_history = (
        centroid_history
    )

    st.session_state.label_history = (
        label_history
    )

    st.session_state.iteration = 0

    st.session_state.generated = True

    st.session_state.manual_selection_active = False

    st.session_state.plot_key += 1


# ============================================================
# COLOR HELPER
# ============================================================

def rgba(
    hex_color,
    alpha
):

    rgb = hex_to_rgb(
        hex_color
    )

    return (
        f"rgba("
        f"{rgb[0]},"
        f"{rgb[1]},"
        f"{rgb[2]},"
        f"{alpha}"
        f")"
    )


# ============================================================
# CREATE CLUSTER REGION
# ============================================================

def create_cluster_region(
    centroids,
    cluster_id,
    resolution=100
):

    """
    Create a visual Voronoi-like region.

    Each grid location is assigned to the
    nearest centroid.

    The region is used only for visualization.
    """

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
        (
            xx.ravel(),
            yy.ravel()
        )
    )

    labels = assign_clusters(
        grid,
        centroids
    )

    labels = labels.reshape(
        xx.shape
    )

    mask = (
        labels == cluster_id
    )

    z = np.where(
        mask,
        1,
        np.nan
    )

    return (
        x,
        y,
        z
    )


# ============================================================
# CREATE MAIN PLOT
# ============================================================

def create_plot(
    points,
    n_clusters,
    iteration,
    centroid_history,
    label_history
):

    fig = go.Figure()

    # ========================================================
    # ITERATION 0
    # ========================================================

    if iteration == 0:

        # ----------------------------------------------------
        # Random points
        # ----------------------------------------------------

        fig.add_trace(
            go.Scatter(
                x=points[:, 0],
                y=points[:, 1],
                mode="markers",
                marker=dict(
                    size=9,
                    color="#777777",
                    opacity=0.75,
                    line=dict(
                        color="white",
                        width=1
                    )
                ),
                name="Random points",
                hovertemplate=(
                    "X: %{x:.1f}<br>"
                    "Y: %{y:.1f}"
                    "<extra></extra>"
                )
            )
        )

        # ----------------------------------------------------
        # Initial centroids
        # ----------------------------------------------------

        initial_centroids = (
            centroid_history[0]
        )

        fig.add_trace(
            go.Scatter(
                x=initial_centroids[:, 0],
                y=initial_centroids[:, 1],
                mode="markers+text",
                marker=dict(
                    size=20,
                    symbol="x",
                    color="#222222",
                    line=dict(
                        color="white",
                        width=2
                    )
                ),
                text=[
                    f"C{i + 1}"
                    for i in range(
                        n_clusters
                    )
                ],
                textposition="top right",
                textfont=dict(
                    size=12,
                    color="#222222"
                ),
                name="Initial centroids",
                hovertemplate=(
                    "Centroid %{text}"
                    "<br>X: %{x:.1f}"
                    "<br>Y: %{y:.1f}"
                    "<extra></extra>"
                )
            )
        )

    # ========================================================
    # K-MEANS ITERATIONS
    # ========================================================

    else:

        current_labels = (
            label_history[
                iteration - 1
            ]
        )

        current_centroids = (
            centroid_history[
                iteration
            ]
        )

        # ----------------------------------------------------
        # LIGHT CLUSTER REGIONS
        # ----------------------------------------------------

        for cluster_id in range(
            n_clusters
        ):

            x, y, z = (
                create_cluster_region(
                    current_centroids,
                    cluster_id
                )
            )

            fig.add_trace(
                go.Heatmap(
                    x=x,
                    y=y,
                    z=z,
                    colorscale=[
                        [
                            0,
                            rgba(
                                CLUSTER_COLORS[
                                    cluster_id
                                ],
                                0
                            )
                        ],
                        [
                            1,
                            rgba(
                                CLUSTER_COLORS[
                                    cluster_id
                                ],
                                0.10
                            )
                        ]
                    ],
                    showscale=False,
                    hoverinfo="skip",
                    zsmooth=False
                )
            )

        # ----------------------------------------------------
        # CLUSTERED POINTS
        # ----------------------------------------------------

        for cluster_id in range(
            n_clusters
        ):

            mask = (
                current_labels
                == cluster_id
            )

            fig.add_trace(
                go.Scatter(
                    x=points[
                        mask,
                        0
                    ],
                    y=points[
                        mask,
                        1
                    ],
                    mode="markers",
                    marker=dict(
                        size=9,
                        color=CLUSTER_COLORS[
                            cluster_id
                        ],
                        opacity=0.82,
                        line=dict(
                            color="white",
                            width=1
                        )
                    ),
                    name=CLUSTER_NAMES[
                        cluster_id
                    ],
                    hovertemplate=(
                        "X: %{x:.1f}<br>"
                        "Y: %{y:.1f}"
                        "<extra>"
                        f"{CLUSTER_NAMES[cluster_id]}"
                        "</extra>"
                    )
                )
            )

        # ----------------------------------------------------
        # CENTROID TRAILS
        # ----------------------------------------------------

        if iteration > 1:

            for cluster_id in range(
                n_clusters
            ):

                trajectory = np.array(
                    [
                        centroid_history[
                            i
                        ][cluster_id]
                        for i in range(
                            0,
                            iteration + 1
                        )
                    ]
                )

                fig.add_trace(
                    go.Scatter(
                        x=trajectory[:, 0],
                        y=trajectory[:, 1],
                        mode="lines",
                        line=dict(
                            color=CLUSTER_COLORS[
                                cluster_id
                            ],
                            width=2,
                            dash="dash"
                        ),
                        opacity=0.45,
                        showlegend=False,
                        hoverinfo="skip"
                    )
                )

        # ----------------------------------------------------
        # CURRENT CENTROIDS
        # ----------------------------------------------------

        fig.add_trace(
            go.Scatter(
                x=current_centroids[:, 0],
                y=current_centroids[:, 1],
                mode="markers+text",
                marker=dict(
                    size=23,
                    symbol="x",
                    color=[
                        CLUSTER_COLORS[i]
                        for i in range(
                            n_clusters
                        )
                    ],
                    line=dict(
                        color="black",
                        width=2
                    )
                ),
                text=[
                    f"C{i + 1}"
                    for i in range(
                        n_clusters
                    )
                ],
                textposition="top right",
                textfont=dict(
                    size=12,
                    color="black"
                ),
                name="Centroids",
                hovertemplate=(
                    "Centroid %{text}"
                    "<br>X: %{x:.1f}"
                    "<br>Y: %{y:.1f}"
                    "<extra></extra>"
                )
            )
        )

    # ========================================================
    # LAYOUT
    # ========================================================

    if iteration == 0:

        title = (
            "Random points + initial centroids"
        )

    else:

        title = (
            f"Clusters forming — "
            f"Iteration {iteration}"
        )

    fig.update_layout(

        title=dict(
            text=title,
            x=0.5,
            xanchor="center",
            font=dict(
                size=18
            )
        ),

        xaxis=dict(
            title="X",
            range=[
                X_MIN,
                X_MAX
            ],
            fixedrange=True,
            showgrid=True,
            gridcolor="rgba(0,0,0,0.10)",
            zeroline=False
        ),

        yaxis=dict(
            title="Y",
            range=[
                Y_MIN,
                Y_MAX
            ],
            fixedrange=True,
            showgrid=True,
            gridcolor="rgba(0,0,0,0.10)",
            zeroline=False,
            scaleanchor="x",
            scaleratio=1
        ),

        height=570,

        margin=dict(
            l=45,
            r=25,
            t=60,
            b=45
        ),

        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.01,
            xanchor="right",
            x=1
        ),

        plot_bgcolor="white",

        paper_bgcolor="white",

        hovermode="closest"
    )

    return fig


# ============================================================
# CREATE MANUAL-CENTROID PLOT
# ============================================================

def create_manual_selection_plot():

    """
    Interactive Plotly plot used only when the
    student is placing centroids manually.
    """

    points = (
        st.session_state.points
    )

    fig = go.Figure()

    # --------------------------------------------------------
    # Random points
    # --------------------------------------------------------

    fig.add_trace(
        go.Scatter(
            x=points[:, 0],
            y=points[:, 1],
            mode="markers",
            marker=dict(
                size=9,
                color="#777777",
                opacity=0.75,
                line=dict(
                    color="white",
                    width=1
                )
            ),
            name="Random points",
            hovertemplate=(
                "X: %{x:.1f}<br>"
                "Y: %{y:.1f}"
                "<extra></extra>"
            )
        )
    )

    # --------------------------------------------------------
    # Already selected centroids
    # --------------------------------------------------------

    selected = (
        st.session_state.manual_centroids
    )

    if len(selected) > 0:

        selected = np.array(
            selected
        )

        fig.add_trace(
            go.Scatter(
                x=selected[:, 0],
                y=selected[:, 1],
                mode="markers+text",
                marker=dict(
                    size=24,
                    symbol="x",
                    color=[
                        CLUSTER_COLORS[i]
                        for i in range(
                            len(selected)
                        )
                    ],
                    line=dict(
                        color="black",
                        width=2
                    )
                ),
                text=[
                    f"C{i + 1}"
                    for i in range(
                        len(selected)
                    )
                ],
                textposition="top right",
                textfont=dict(
                    size=12,
                    color="black"
                ),
                name="Selected centroids",
                hovertemplate=(
                    "Centroid %{text}"
                    "<br>X: %{x:.1f}"
                    "<br>Y: %{y:.1f}"
                    "<extra></extra>"
                )
            )
        )

    # --------------------------------------------------------
    # Layout
    # --------------------------------------------------------

    remaining = (
        st.session_state.n_clusters
        - len(selected)
    )

    if remaining > 0:

        title = (
            f"👆 Click to place "
            f"centroid "
            f"{len(selected) + 1} "
            f"of "
            f"{st.session_state.n_clusters}"
        )

    else:

        title = (
            "All starting centroids selected"
        )

    fig.update_layout(

        title=dict(
            text=title,
            x=0.5,
            xanchor="center",
            font=dict(
                size=18
            )
        ),

        xaxis=dict(
            title="X",
            range=[
                X_MIN,
                X_MAX
            ],
            fixedrange=True,
            showgrid=True,
            gridcolor="rgba(0,0,0,0.10)",
            zeroline=False
        ),

        yaxis=dict(
            title="Y",
            range=[
                Y_MIN,
                Y_MAX
            ],
            fixedrange=True,
            showgrid=True,
            gridcolor="rgba(0,0,0,0.10)",
            zeroline=False,
            scaleanchor="x",
            scaleratio=1
        ),

        height=570,

        margin=dict(
            l=45,
            r=25,
            t=60,
            b=45
        ),

        plot_bgcolor="white",

        paper_bgcolor="white",

        hovermode="closest"
    )

    return fig


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("## 🎮 Experiment")

    st.caption(
        "Create your own K-means experiment."
    )

    st.divider()

    # --------------------------------------------------------
    # NUMBER OF POINTS
    # --------------------------------------------------------

    st.markdown(
        "### 🔵 Data points"
    )

    n_points_input = st.number_input(
        "Enter number of points",
        min_value=20,
        max_value=300,
        value=st.session_state.n_points,
        step=10
    )

    n_points_slider = st.slider(
        "Or use the slider",
        min_value=20,
        max_value=300,
        value=int(
            n_points_input
        ),
        step=10
    )

    n_points = n_points_slider

    # --------------------------------------------------------
    # NUMBER OF CLUSTERS
    # --------------------------------------------------------

    st.markdown(
        "### 🎨 Number of clusters"
    )

    n_clusters_input = st.number_input(
        "Enter number of clusters",
        min_value=2,
        max_value=6,
        value=st.session_state.n_clusters,
        step=1
    )

    n_clusters_slider = st.slider(
        "Or use the slider",
        min_value=2,
        max_value=6,
        value=int(
            n_clusters_input
        ),
        step=1
    )

    n_clusters = n_clusters_slider

    st.divider()

    # --------------------------------------------------------
    # CENTROID INITIALIZATION
    # --------------------------------------------------------

    st.markdown(
        "### 📍 Starting centroids"
    )

    centroid_mode = st.radio(
        "How should centroids be placed?",
        [
            "🎲 Random centroids",
            "👆 Place centroids myself"
        ],
        index=(
            0
            if st.session_state.centroid_mode
            == "🎲 Random centroids"
            else 1
        )
    )

    st.session_state.centroid_mode = (
        centroid_mode
    )

    st.divider()

    # --------------------------------------------------------
    # GENERATE
    # --------------------------------------------------------

    if st.button(
        "✨ Generate Points & Start",
        use_container_width=True
    ):

        if (
            centroid_mode
            == "🎲 Random centroids"
        ):

            create_experiment(
                n_points,
                n_clusters
            )

        else:

            # Generate points first
            st.session_state.points = (
                generate_points(
                    n_points
                )
            )

            st.session_state.n_points = (
                n_points
            )

            st.session_state.n_clusters = (
                n_clusters
            )

            st.session_state.manual_centroids = []

            st.session_state.manual_selection_active = True

            st.session_state.generated = False

            st.session_state.plot_key += 1

        st.rerun()

    # --------------------------------------------------------
    # NEW EXPERIMENT
    # --------------------------------------------------------

    if st.button(
        "🔄 New Experiment",
        use_container_width=True
    ):

        if (
            centroid_mode
            == "🎲 Random centroids"
        ):

            create_experiment(
                n_points,
                n_clusters
            )

        else:

            st.session_state.points = (
                generate_points(
                    n_points
                )
            )

            st.session_state.n_points = (
                n_points
            )

            st.session_state.n_clusters = (
                n_clusters
            )

            st.session_state.manual_centroids = []

            st.session_state.manual_selection_active = True

            st.session_state.generated = False

            st.session_state.plot_key += 1

        st.rerun()

    st.divider()

    st.markdown(
        "### 💡 Try this"
    )

    if (
        centroid_mode
        == "🎲 Random centroids"
    ):

        st.caption(
            """
            Change the number of points
            and clusters.

            Watch how the centroids move
            and how the groups form.
            """
        )

    else:

        st.caption(
            """
            Choose where you think the
            starting centroids should be.

            If you select 3 clusters,
            click exactly 3 locations
            on the plot.
            """
        )


# ============================================================
# FIRST RUN
# ============================================================

if (
    not st.session_state.generated
    and
    not st.session_state.manual_selection_active
):

    create_experiment(
        st.session_state.n_points,
        st.session_state.n_clusters
    )


# ============================================================
# MANUAL CENTROID PLACEMENT
# ============================================================

if (
    st.session_state.manual_selection_active
):

    selected_count = len(
        st.session_state.manual_centroids
    )

    required_count = (
        st.session_state.n_clusters
    )

    # --------------------------------------------------------
    # STATUS
    # --------------------------------------------------------

    st.markdown(
        '<div class="iteration-box">'
        f'📍 Select starting centroids: '
        f'{selected_count} / '
        f'{required_count}'
        '</div>',
        unsafe_allow_html=True
    )

    if selected_count < required_count:

        st.info(
            f"""
            **Click anywhere on the plot to place
            centroid C{selected_count + 1}.**

            You need to place
            **{required_count - selected_count}**
            more centroid(s).
            """
        )

    # --------------------------------------------------------
    # PLOT
    # --------------------------------------------------------

    manual_fig = (
        create_manual_selection_plot()
    )

    event = st.plotly_chart(
        manual_fig,
        use_container_width=True,
        key=(
            f"manual_plot_"
            f"{st.session_state.plot_key}"
        ),
        on_select="rerun",
        selection_mode="points"
    )

    # --------------------------------------------------------
    # READ PLOTLY CLICK
    # --------------------------------------------------------

    if event is not None:

        selection = getattr(
            event,
            "selection",
            None
        )

        if selection is not None:

            points_selected = getattr(
                selection,
                "points",
                []
            )

            if (
                points_selected
                and
                selected_count
                < required_count
            ):

                clicked = (
                    points_selected[-1]
                )

                # ------------------------------------------------
                # Plotly event normally gives x/y
                # for a clicked data point.
                #
                # Because the random points are themselves
                # clickable, we use the nearest clicked
                # coordinate as the centroid position.
                # ------------------------------------------------

                clicked_x = clicked.get(
                    "x"
                )

                clicked_y = clicked.get(
                    "y"
                )

                if (
                    clicked_x is not None
                    and
                    clicked_y is not None
                ):

                    # ------------------------------------------------
                    # IMPORTANT:
                    # The student can click anywhere on the plot.
                    #
                    # Plotly selection is based on the nearest
                    # visible point. To make the centroid truly
                    # correspond to the selected location, use
                    # the clicked point coordinates.
                    # ------------------------------------------------

                    new_centroid = [
                        float(clicked_x),
                        float(clicked_y)
                    ]

                    # Avoid exact duplicate centroid locations
                    duplicate = False

                    for old_centroid in (
                        st.session_state.manual_centroids
                    ):

                        distance = np.linalg.norm(
                            np.array(
                                new_centroid
                            )
                            -
                            np.array(
                                old_centroid
                            )
                        )

                        if distance < 1:

                            duplicate = True

                            break

                    if not duplicate:

                        st.session_state.manual_centroids.append(
                            new_centroid
                        )

                        # ------------------------------------------------
                        # ALL CENTROIDS SELECTED
                        # ------------------------------------------------

                        if (
                            len(
                                st.session_state.manual_centroids
                            )
                            ==
                            required_count
                        ):

                            create_experiment(
                                n_points,
                                n_clusters,
                                st.session_state.manual_centroids
                            )

                            st.session_state.manual_selection_active = (
                                False
                            )

                            st.rerun()

                        else:

                            st.session_state.plot_key += 1

                            st.rerun()

    # --------------------------------------------------------
    # SELECTED CENTROIDS
    # --------------------------------------------------------

    if selected_count > 0:

        st.markdown(
            "### 📍 Selected centroids"
        )

        cols = st.columns(
            selected_count
        )

        for i, centroid in enumerate(
            st.session_state.manual_centroids
        ):

            with cols[i]:

                st.markdown(
                    f"""
                    <div style="
                        padding:7px;
                        border-radius:8px;
                        background:{CLUSTER_COLORS[i]};
                        color:white;
                        text-align:center;
                        font-weight:600;
                        font-size:12px;
                    ">
                    C{i + 1}
                    <br>
                    ({centroid[0]:.1f},
                     {centroid[1]:.1f})
                    </div>
                    """,
                    unsafe_allow_html=True
                )

    st.stop()


# ============================================================
# LOAD CURRENT EXPERIMENT
# ============================================================

points = st.session_state.points

n_points = st.session_state.n_points

n_clusters = st.session_state.n_clusters

iteration = st.session_state.iteration

centroid_history = (
    st.session_state.centroid_history
)

label_history = (
    st.session_state.label_history
)


# ============================================================
# ITERATION CONTROL
# ============================================================

st.markdown(
    '<div class="iteration-box">'
    f'ITERATION {iteration} / '
    f'{MAX_ITERATIONS}'
    '</div>',
    unsafe_allow_html=True
)

control1, control2, control3 = st.columns(
    [1, 6, 1]
)


# ============================================================
# PREVIOUS
# ============================================================

with control1:

    if st.button(
        "◀",
        use_container_width=True
    ):

        st.session_state.iteration = max(
            0,
            iteration - 1
        )

        st.rerun()


# ============================================================
# ITERATION SLIDER
# ============================================================

with control2:

    selected_iteration = st.slider(
        "Iteration",
        min_value=0,
        max_value=MAX_ITERATIONS,
        value=iteration,
        step=1,
        label_visibility="collapsed"
    )

    if (
        selected_iteration
        != iteration
    ):

        st.session_state.iteration = (
            selected_iteration
        )

        st.rerun()


# ============================================================
# NEXT
# ============================================================

with control3:

    if st.button(
        "▶",
        use_container_width=True
    ):

        st.session_state.iteration = min(
            MAX_ITERATIONS,
            iteration + 1
        )

        st.rerun()


iteration = (
    st.session_state.iteration
)


# ============================================================
# CURRENT DATA
# ============================================================

if iteration == 0:

    current_labels = None

    current_centroids = (
        centroid_history[0]
    )

else:

    current_labels = (
        label_history[
            iteration - 1
        ]
    )

    current_centroids = (
        centroid_history[
            iteration
        ]
    )


# ============================================================
# MAIN CONTENT
# ============================================================

plot_col, info_col = st.columns(
    [3.5, 1]
)


# ============================================================
# MAIN VISUALIZATION
# ============================================================

with plot_col:

    fig = create_plot(
        points,
        n_clusters,
        iteration,
        centroid_history,
        label_history
    )

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

    st.markdown(
        "### 📌 What's happening?"
    )

    if iteration == 0:

        if (
            st.session_state.centroid_mode
            == "👆 Place centroids myself"
        ):

            st.markdown(
                """
                **Step 0 — Starting positions**

                The points are still random.

                The **X markers** show the
                starting centroid positions
                you selected.

                No cluster colours have been
                assigned yet.
                """
            )

        else:

            st.markdown(
                """
                **Step 0 — Random**

                All points are random.

                No clusters have been formed yet.

                The **X markers** are the initial
                centroid positions.
                """
            )

    elif iteration == 1:

        st.markdown(
            """
            **Step 1 — Assign**

            Each point looks for its
            nearest centroid.

            Points receive their first
            colours.

            The light background shows the
            region belonging to each centroid.
            """
        )

    elif iteration < MAX_ITERATIONS:

        st.markdown(
            f"""
            **Step {iteration}**

            The points have been assigned
            to the nearest centroid.

            The centroids then move toward
            the middle of their groups.

            Watch the **X markers** and
            coloured regions move!
            """
        )

    else:

        st.markdown(
            """
            **✨ Final iteration**

            The groups have stabilised.

            Each centroid now sits close to
            the centre of its cluster.
            """
        )

    st.divider()

    # --------------------------------------------------------
    # DATA SUMMARY
    # --------------------------------------------------------

    st.markdown(
        "### 📊 Experiment"
    )

    st.markdown(
        f"""
        **Points:** {n_points}

        **Clusters:** {n_clusters}

        **Iteration:** {iteration} / 20
        """
    )

    st.divider()

    # --------------------------------------------------------
    # COLOUR LEGEND
    # --------------------------------------------------------

    st.markdown(
        "### 🎨 Groups"
    )

    for i in range(
        n_clusters
    ):

        st.markdown(
            f"""
            <div style="
                padding:4px;
                margin:2px;
                border-radius:6px;
                background:{CLUSTER_COLORS[i]};
                color:white;
                text-align:center;
                font-weight:600;
                font-size:12px;
            ">
            {CLUSTER_NAMES[i]}
            </div>
            """,
            unsafe_allow_html=True
        )


# ============================================================
# BOTTOM EXPLANATION
# ============================================================

st.divider()

step1, step2, step3 = st.columns(
    3
)

with step1:

    st.markdown(
        """
        ### ① Pick starting points

        K-means first places a number of
        **centroids** on the canvas.

        You can either let the computer
        choose them randomly or place
        them yourself.
        """
    )

with step2:

    st.markdown(
        """
        ### ② Find the nearest centroid

        Every point asks:

        **"Which centroid am I closest to?"**

        Points belonging to the same centroid
        receive the same colour.
        """
    )

with step3:

    st.markdown(
        """
        ### ③ Move & repeat

        Each centroid moves toward the centre
        of its assigned points.

        The coloured regions also change as
        the centroids move.
        """
    )
