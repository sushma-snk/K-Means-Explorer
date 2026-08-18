import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

from streamlit_drawable_canvas import st_canvas
from PIL import Image
import io


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

    /* Remove excessive Streamlit spacing */
    .block-container {
        padding-top: 1rem;
        padding-bottom: 0.5rem;
        max-width: 1500px;
    }

    /* Main title */
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

    /* Cards */
    .card {
        padding: 12px;
        border-radius: 14px;
        background-color: #f6f6f6;
        text-align: center;
        margin-bottom: 8px;
    }

    .card-title {
        font-size: 13px;
        color: #666;
        margin-bottom: 2px;
    }

    .card-value {
        font-size: 24px;
        font-weight: 700;
    }

    /* Iteration display */
    .iteration-box {
        text-align: center;
        font-size: 20px;
        font-weight: 700;
        padding: 8px;
        border-radius: 12px;
        background-color: #f4f4f4;
    }

    /* Explanation */
    .explanation {
        font-size: 14px;
        line-height: 1.4;
    }

    /* Buttons */
    div.stButton > button {
        border-radius: 10px;
        font-weight: 600;
    }

    /* Sidebar */
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


# Visually distinct colours
CLUSTER_COLORS = [
    "#FF6B6B",  # red/coral
    "#4D96FF",  # blue
    "#6BCB77",  # green
    "#B983FF",  # purple
    "#FFB84C",  # orange
    "#00B8A9",  # teal
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

if "points" not in st.session_state:
    st.session_state.points = None

if "n_points" not in st.session_state:
    st.session_state.n_points = 100

if "n_clusters" not in st.session_state:
    st.session_state.n_clusters = 3

if "iteration" not in st.session_state:
    st.session_state.iteration = 0

if "centroid_history" not in st.session_state:
    st.session_state.centroid_history = None

if "label_history" not in st.session_state:
    st.session_state.label_history = None

if "generated" not in st.session_state:
    st.session_state.generated = False

# ------------------------------------------------------------
# NEW: centroid initialization mode
# ------------------------------------------------------------

if "centroid_mode" not in st.session_state:
    st.session_state.centroid_mode = "🎲 Random centroids"

# ------------------------------------------------------------
# NEW: manually selected centroids
# ------------------------------------------------------------

if "manual_centroids" not in st.session_state:
    st.session_state.manual_centroids = []

# ------------------------------------------------------------
# NEW: whether user is currently selecting centroids
# ------------------------------------------------------------

if "manual_selection_active" not in st.session_state:
    st.session_state.manual_selection_active = False

# ------------------------------------------------------------
# NEW: number of clicks already processed
# ------------------------------------------------------------

if "processed_clicks" not in st.session_state:
    st.session_state.processed_clicks = 0


# ============================================================
# RANDOM DATA GENERATION
# ============================================================

def generate_points(n_points):

    """
    Generate completely random points.

    Important:
    At iteration 0, ALL points look identical.
    No cluster information is shown.
    """

    return np.random.uniform(
        5,
        95,
        size=(n_points, 2)
    )


# ============================================================
# INITIAL CENTROIDS
# ============================================================

def initialize_centroids(points, n_clusters):

    """
    Select initial centroids randomly from the dataset.
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

def assign_clusters(points, centroids):

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
# RUN K-MEANS AND SAVE EVERY ITERATION
# ============================================================

def run_kmeans(
    points,
    n_clusters,
    max_iterations=20,
    initial_centroids=None
):

    """
    Run K-means once and store:

        centroid position
        cluster assignment

    for every iteration.

    If initial_centroids is supplied,
    those are used instead of randomly
    selecting the initial centroids.
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

        # Step 1:
        # Assign points to nearest centroid
        labels = assign_clusters(
            points,
            centroids
        )

        label_history.append(
            labels.copy()
        )

        # Step 2:
        # Move centroid to mean of its points
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


# ============================================================
# CREATE BACKGROUND IMAGE FOR MANUAL CLICKING
# ============================================================

def create_selection_plot():

    """
    Create the same visual style as the normal
    Matplotlib plot, but save it as an image so
    the drawable canvas can receive clicks.
    """

    points = st.session_state.points

    fig, ax = plt.subplots(
        figsize=(8.5, 6.0),
        dpi=100
    )

    # --------------------------------------------------------
    # RANDOM POINTS
    # --------------------------------------------------------

    ax.scatter(
        points[:, 0],
        points[:, 1],
        s=65,
        color="#777777",
        alpha=0.75,
        edgecolors="white",
        linewidths=0.7
    )

    # --------------------------------------------------------
    # ALREADY SELECTED CENTROIDS
    # --------------------------------------------------------

    selected = (
        st.session_state.manual_centroids
    )

    for i, centroid in enumerate(selected):

        ax.scatter(
            centroid[0],
            centroid[1],
            s=280,
            marker="X",
            color=CLUSTER_COLORS[i],
            edgecolors="black",
            linewidths=1.5,
            zorder=10
        )

        ax.annotate(
            f"C{i + 1}",
            (
                centroid[0],
                centroid[1]
            ),
            xytext=(7, 7),
            textcoords="offset points",
            fontsize=10,
            fontweight="bold"
        )

    # --------------------------------------------------------
    # AXES
    # --------------------------------------------------------

    ax.set_xlim(
        X_MIN,
        X_MAX
    )

    ax.set_ylim(
        Y_MIN,
        Y_MAX
    )

    ax.set_xlabel(
        "X",
        fontsize=11
    )

    ax.set_ylabel(
        "Y",
        fontsize=11
    )

    ax.grid(
        alpha=0.15
    )

    ax.set_title(
        f"👆 Click to place centroid "
        f"{len(selected) + 1} of "
        f"{st.session_state.n_clusters}",
        fontsize=16,
        fontweight="bold"
    )

    fig.tight_layout()

    # --------------------------------------------------------
    # SAVE FIGURE TO MEMORY
    # --------------------------------------------------------

    buffer = io.BytesIO()

    fig.savefig(
        buffer,
        format="png",
        dpi=100,
        bbox_inches="tight"
    )

    plt.close(fig)

    buffer.seek(0)

    return Image.open(buffer)


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

    st.markdown("### 🔵 Data points")

    n_points = st.number_input(
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
        value=int(n_points),
        step=10
    )

    # Use slider value
    n_points = n_points_slider

    # --------------------------------------------------------
    # NUMBER OF CLUSTERS
    # --------------------------------------------------------

    st.markdown("### 🎨 Number of clusters")

    n_clusters = st.number_input(
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
        value=int(n_clusters),
        step=1
    )

    n_clusters = n_clusters_slider

    st.divider()

    # ========================================================
    # NEW: CENTROID INITIALIZATION
    # ========================================================

    st.markdown("### 📍 Starting centroids")

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

        # ----------------------------------------------------
        # RANDOM CENTROIDS
        # ----------------------------------------------------

        if centroid_mode == "🎲 Random centroids":

            create_experiment(
                n_points,
                n_clusters
            )

            st.session_state.manual_centroids = []

            st.session_state.processed_clicks = 0

            st.rerun()

        # ----------------------------------------------------
        # MANUAL CENTROIDS
        # ----------------------------------------------------

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

            # Clear old selections
            st.session_state.manual_centroids = []

            # Activate manual mode
            st.session_state.manual_selection_active = True

            st.session_state.generated = False

            # Reset click tracking
            st.session_state.processed_clicks = 0

            st.rerun()

    # --------------------------------------------------------
    # RESET
    # --------------------------------------------------------

    if st.button(
        "🔄 New Experiment",
        use_container_width=True
    ):

        if centroid_mode == "🎲 Random centroids":

            create_experiment(
                n_points,
                n_clusters
            )

            st.session_state.manual_centroids = []

            st.session_state.processed_clicks = 0

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

            st.session_state.processed_clicks = 0

        st.rerun()

    st.divider()

    st.markdown("### 💡 Try this")

    if centroid_mode == "🎲 Random centroids":

        st.caption(
            """
            Change the number of points and clusters,
            then watch how the centroids move.

            Can you guess where the final clusters
            will form before reaching iteration 20?
            """
        )

    else:

        st.caption(
            """
            Choose where you think the starting
            centroids should be.

            If you select 3 clusters, click
            exactly 3 locations on the plot.
            """
        )


# ============================================================
# FIRST RUN
# ============================================================

if not st.session_state.generated:

    # Only automatically create a random experiment
    # if manual selection is NOT active.

    if not st.session_state.manual_selection_active:

        create_experiment(
            st.session_state.n_points,
            st.session_state.n_clusters
        )


# ============================================================
# MANUAL CENTROID SELECTION
# ============================================================

if st.session_state.manual_selection_active:

    # --------------------------------------------------------
    # INFORMATION MESSAGE
    # --------------------------------------------------------

    selected_count = len(
        st.session_state.manual_centroids
    )

    required_count = (
        st.session_state.n_clusters
    )

    st.markdown(
        f"""
        <div class="iteration-box">
        👆 Place centroid
        {selected_count + 1}
        of
        {required_count}
        </div>
        """,
        unsafe_allow_html=True
    )

    st.info(
        f"""
        Click on the plot to place the starting
        centroids.

        You have selected **{selected_count}**
        of **{required_count}** centroids.
        """
    )

    # --------------------------------------------------------
    # CREATE IMAGE
    # --------------------------------------------------------

    background_image = (
        create_selection_plot()
    )

    # --------------------------------------------------------
    # CANVAS SIZE
    # --------------------------------------------------------

    CANVAS_WIDTH = 850
    CANVAS_HEIGHT = 600

    # Resize background to canvas size
    background_image = background_image.resize(
        (
            CANVAS_WIDTH,
            CANVAS_HEIGHT
        )
    )

    # --------------------------------------------------------
    # CLICKABLE CANVAS
    # --------------------------------------------------------

    canvas_result = st_canvas(
        fill_color="rgba(255,255,255,0)",
        stroke_width=1,
        stroke_color="rgba(255,255,255,0)",
        background_image=background_image,
        update_streamlit=True,
        height=CANVAS_HEIGHT,
        width=CANVAS_WIDTH,
        drawing_mode="point",
        point_display_radius=8,
        display_toolbar=False,
        key="centroid_canvas"
    )

    # --------------------------------------------------------
    # READ CLICKED POINTS
    # --------------------------------------------------------

    if (
        canvas_result.json_data is not None
    ):

        objects = (
            canvas_result
            .json_data
            .get(
                "objects",
                []
            )
        )

        # Process only NEW clicks
        if len(objects) > (
            st.session_state.processed_clicks
        ):

            latest_object = objects[-1]

            canvas_x = (
                latest_object.get(
                    "left",
                    0
                )
            )

            canvas_y = (
                latest_object.get(
                    "top",
                    0
                )
            )

            # ------------------------------------------------
            # Convert canvas coordinates
            # to X/Y coordinates 0-100
            # ------------------------------------------------

            # Canvas is 850 x 600
            #
            # X:
            # 0 -> 0
            # 850 -> 100
            #
            # Y:
            # 0 -> 100
            # 600 -> 0

            plot_x = (
                canvas_x
                / CANVAS_WIDTH
                * (X_MAX - X_MIN)
                + X_MIN
            )

            plot_y = (
                1
                - (
                    canvas_y
                    / CANVAS_HEIGHT
                )
            ) * (
                Y_MAX - Y_MIN
            ) + Y_MIN

            # Keep inside plotting area
            plot_x = np.clip(
                plot_x,
                X_MIN,
                X_MAX
            )

            plot_y = np.clip(
                plot_y,
                Y_MIN,
                Y_MAX
            )

            # ------------------------------------------------
            # Add centroid
            # ------------------------------------------------

            if len(
                st.session_state.manual_centroids
            ) < required_count:

                st.session_state.manual_centroids.append(
                    [
                        float(plot_x),
                        float(plot_y)
                    ]
                )

            st.session_state.processed_clicks = (
                len(objects)
            )

            # ------------------------------------------------
            # ALL CENTROIDS SELECTED
            # ------------------------------------------------

            if len(
                st.session_state.manual_centroids
            ) == required_count:

                create_experiment(
                    n_points,
                    n_clusters,
                    st.session_state.manual_centroids
                )

                st.session_state.manual_selection_active = (
                    False
                )

                st.session_state.processed_clicks = 0

            st.rerun()

    # --------------------------------------------------------
    # SHOW SELECTED CENTROIDS
    # --------------------------------------------------------

    if selected_count > 0:

        st.markdown(
            "### 📍 Selected starting centroids"
        )

        centroid_cols = st.columns(
            min(
                selected_count,
                6
            )
        )

        for i, centroid in enumerate(
            st.session_state.manual_centroids
        ):

            with centroid_cols[
                i % len(centroid_cols)
            ]:

                st.markdown(
                    f"""
                    <div style="
                        padding:7px;
                        margin:2px;
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

    # Stop here so the normal
    # iteration visualization doesn't appear
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
    f'ITERATION {iteration} / {MAX_ITERATIONS}'
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
# SLIDER
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

    if selected_iteration != iteration:

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


iteration = st.session_state.iteration


# ============================================================
# CURRENT DATA
# ============================================================

if iteration == 0:

    # No classes yet
    current_labels = None

    current_centroids = centroid_history[0]

else:

    current_labels = label_history[
        iteration - 1
    ]

    current_centroids = centroid_history[
        iteration
    ]


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

    fig, ax = plt.subplots(
        figsize=(8.5, 6.0)
    )

    # ========================================================
    # RANDOM POINT STAGE
    # ========================================================

    if iteration == 0:

        ax.scatter(
            points[:, 0],
            points[:, 1],
            s=65,
            color="#777777",
            alpha=0.75,
            edgecolors="white",
            linewidths=0.7
        )

        # Initial centroids
        ax.scatter(
            current_centroids[:, 0],
            current_centroids[:, 1],
            s=230,
            marker="X",
            color="#222222",
            edgecolors="white",
            linewidths=1.5,
            zorder=5
        )

        ax.set_title(
            "Random points + initial centroids",
            fontsize=16,
            fontweight="bold"
        )

    # ========================================================
    # K-MEANS ITERATIONS
    # ========================================================

    else:

        # ----------------------------------------------------
        # NEW:
        # CREATE COLOUR REGIONS
        # ----------------------------------------------------

        # Create a fine grid over the fixed X-Y space
        grid_x = np.linspace(
            X_MIN,
            X_MAX,
            180
        )

        grid_y = np.linspace(
            Y_MIN,
            Y_MAX,
            180
        )

        xx, yy = np.meshgrid(
            grid_x,
            grid_y
        )

        grid_points = np.column_stack(
            (
                xx.ravel(),
                yy.ravel()
            )
        )

        # Determine which centroid owns
        # each region of the space
        grid_labels = assign_clusters(
            grid_points,
            current_centroids
        )

        grid_labels = grid_labels.reshape(
            xx.shape
        )

        # ----------------------------------------------------
        # Draw each cluster region
        # ----------------------------------------------------

        for cluster_id in range(
            n_clusters
        ):

            region = (
                grid_labels
                == cluster_id
            )

            # Very light version of the
            # actual cluster colour
            ax.contourf(
                xx,
                yy,
                region.astype(float),
                levels=[
                    0.5,
                    1.5
                ],
                colors=[
                    CLUSTER_COLORS[
                        cluster_id
                    ]
                ],
                alpha=0.10
            )

        # ----------------------------------------------------
        # COLOURED DATA POINTS
        # ----------------------------------------------------

        for cluster_id in range(
            n_clusters
        ):

            mask = (
                current_labels
                == cluster_id
            )

            ax.scatter(
                points[mask, 0],
                points[mask, 1],
                s=65,
                color=CLUSTER_COLORS[
                    cluster_id
                ],
                alpha=0.78,
                edgecolors="white",
                linewidths=0.7,
                label=CLUSTER_NAMES[
                    cluster_id
                ]
            )

        # ----------------------------------------------------
        # SHOW CENTROID MOVEMENT TRAIL
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

                ax.plot(
                    trajectory[:, 0],
                    trajectory[:, 1],
                    linestyle="--",
                    linewidth=1.5,
                    color=CLUSTER_COLORS[
                        cluster_id
                    ],
                    alpha=0.45
                )

        # ----------------------------------------------------
        # CURRENT CENTROIDS
        # ----------------------------------------------------

        for cluster_id in range(
            n_clusters
        ):

            centroid = current_centroids[
                cluster_id
            ]

            ax.scatter(
                centroid[0],
                centroid[1],
                s=280,
                marker="X",
                color=CLUSTER_COLORS[
                    cluster_id
                ],
                edgecolors="black",
                linewidths=1.5,
                zorder=10
            )

            ax.annotate(
                f"C{cluster_id + 1}",
                (
                    centroid[0],
                    centroid[1]
                ),
                xytext=(7, 7),
                textcoords="offset points",
                fontsize=10,
                fontweight="bold"
            )

        ax.set_title(
            f"Clusters forming — Iteration {iteration}",
            fontsize=16,
            fontweight="bold"
        )

        ax.legend(
            loc="upper right",
            fontsize=9
        )

    # ========================================================
    # AXES
    # ========================================================

    ax.set_xlim(
        X_MIN,
        X_MAX
    )

    ax.set_ylim(
        Y_MIN,
        Y_MAX
    )

    ax.set_xlabel(
        "X",
        fontsize=11
    )

    ax.set_ylabel(
        "Y",
        fontsize=11
    )

    ax.grid(
        alpha=0.15
    )

    st.pyplot(
        fig,
        use_container_width=True
    )

    plt.close(fig)


# ============================================================
# INFORMATION PANEL
# ============================================================

with info_col:

    st.markdown(
        "### 📌 What's happening?"
    )

    if iteration == 0:

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

            Watch the **X markers** move!
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

        The number of centroids depends on
        the number of clusters you choose.
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

        This happens again and again until
        the clusters become stable.
        """
    )
