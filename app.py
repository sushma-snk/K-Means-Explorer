import streamlit as st
import numpy as np
import matplotlib.pyplot as plt


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
    max_iterations=20
):

    """
    Run K-means once and store:

        centroid position
        cluster assignment

    for every iteration.

    This allows the slider to move backward
    and forward without recalculating.
    """

    centroids = initialize_centroids(
        points,
        n_clusters
    )

    centroid_history = [
        centroids.copy()
    ]

    label_history = []

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
    n_clusters
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
        MAX_ITERATIONS
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

    # --------------------------------------------------------
    # GENERATE
    # --------------------------------------------------------

    if st.button(
        "✨ Generate Points & Start",
        use_container_width=True
    ):

        create_experiment(
            n_points,
            n_clusters
        )

    # --------------------------------------------------------
    # RESET
    # --------------------------------------------------------

    if st.button(
        "🔄 New Experiment",
        use_container_width=True
    ):

        create_experiment(
            n_points,
            n_clusters
        )

    st.divider()

    st.markdown("### 💡 Try this")

    st.caption(
        """
        Change the number of points and clusters,
        then watch how the centroids move.

        Can you guess where the final clusters
        will form before reaching iteration 20?
        """
    )


# ============================================================
# FIRST RUN
# ============================================================

if not st.session_state.generated:

    create_experiment(
        st.session_state.n_points,
        st.session_state.n_clusters
    )


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


# Previous
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


# Slider
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


# Next
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

    # --------------------------------------------------------
    # RANDOM POINT STAGE
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # K-MEANS ITERATIONS
    # --------------------------------------------------------

    else:

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
        # Show centroid movement trail
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
        # Current centroids
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

    st.pyplot(
        fig,
        use_container_width=True
    )

    plt.close(fig)


# ============================================================
# INFORMATION PANEL
# ============================================================

with info_col:

    st.markdown("### 📌 What's happening?")

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

    st.markdown("### 📊 Experiment")

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

    st.markdown("### 🎨 Groups")

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

step1, step2, step3 = st.columns(3)

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
