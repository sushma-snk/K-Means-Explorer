import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from sklearn.neighbors import KNeighborsClassifier


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="The KNN Detective",
    page_icon="🔎",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

    .main-title {
        text-align: center;
        font-size: 42px;
        font-weight: 800;
        margin-bottom: 0px;
    }

    .subtitle {
        text-align: center;
        font-size: 18px;
        color: #666;
        margin-bottom: 25px;
    }

    .step-card {
        padding: 20px;
        border-radius: 18px;
        background-color: #f7f7f7;
        text-align: center;
        margin-bottom: 15px;
    }

    .big-number {
        font-size: 35px;
        font-weight: 800;
    }

    .prediction {
        font-size: 28px;
        font-weight: 800;
        text-align: center;
        padding: 20px;
        border-radius: 18px;
        background-color: #f2f2f2;
    }

    .instruction {
        font-size: 17px;
        line-height: 1.6;
    }

    div.stButton > button {
        border-radius: 12px;
        font-weight: 600;
    }

</style>
""", unsafe_allow_html=True)


# ============================================================
# TITLE
# ============================================================

st.markdown(
    '<div class="main-title">🔎 The KNN Detective</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Can you figure out where the new point belongs?'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# SESSION STATE
# ============================================================

if "points" not in st.session_state:
    st.session_state.points = None

if "labels" not in st.session_state:
    st.session_state.labels = None

if "query" not in st.session_state:
    st.session_state.query = None

if "iteration" not in st.session_state:
    st.session_state.iteration = 0

if "started" not in st.session_state:
    st.session_state.started = False

if "classified" not in st.session_state:
    st.session_state.classified = False


# ============================================================
# CONSTANTS
# ============================================================

X_MIN = 0
X_MAX = 100

Y_MIN = 0
Y_MAX = 100

TOTAL_POINTS = 90

# Fixed internally.
# Students don't need to control this.
K = 5

# Number of stages before classification
TOTAL_ITERATIONS = 10


# Soft colours suitable for design students
COLORS = [
    "#FF6B6B",   # coral
    "#4D96FF",   # blue
    "#6BCB77",   # green
    "#B983FF"    # purple
]

CLASS_NAMES = [
    "Coral Group",
    "Blue Group",
    "Green Group",
    "Purple Group"
]


# ============================================================
# GENERATE INITIAL RANDOM POINTS
# ============================================================

def create_random_points():

    rng = np.random.default_rng()

    points = rng.uniform(
        8,
        92,
        size=(TOTAL_POINTS, 2)
    )

    return points


# ============================================================
# CLUSTER FORMATION
# ============================================================

def create_clustered_positions(points, iteration):

    """
    Slowly transform random points into four clusters.

    iteration = 0
        completely random

    iteration = TOTAL_ITERATIONS
        strongly clustered
    """

    rng = np.random.default_rng(1234)

    centres = np.array([
        [25, 70],
        [75, 70],
        [28, 28],
        [72, 28]
    ])

    n = len(points)

    # Assign each point to one of four eventual groups
    assignments = np.arange(n) % 4

    rng.shuffle(assignments)

    # Progress from 0 to 1
    progress = iteration / TOTAL_ITERATIONS

    # Smooth transition
    progress = progress ** 0.8

    clustered = points.copy()

    for i in range(n):

        target = centres[assignments[i]]

        clustered[i] = (
            (1 - progress) * points[i]
            + progress * target
        )

    return clustered, assignments


# ============================================================
# CREATE NEW QUERY POINT
# ============================================================

def create_query_point():

    rng = np.random.default_rng()

    return np.array([
        rng.uniform(15, 85),
        rng.uniform(15, 85)
    ])


# ============================================================
# INITIALIZE EXPERIMENT
# ============================================================

def initialize_experiment():

    st.session_state.points = create_random_points()

    st.session_state.query = create_query_point()

    st.session_state.iteration = 0

    st.session_state.started = False

    st.session_state.classified = False


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("🎮 Experiment")

    st.write(
        "Use the controls below to explore how KNN "
        "makes a decision."
    )

    st.divider()

    if st.button(
        "✨ Create Random Points",
        use_container_width=True
    ):
        initialize_experiment()

    if st.button(
        "🔄 Start Again",
        use_container_width=True
    ):
        initialize_experiment()

    st.divider()

    st.subheader("How to play")

    st.write(
        """
        **1.** Look at the random points.

        **2.** Start the experiment.

        **3.** Move through the iterations.

        **4.** Watch groups emerge.

        **5.** Meet the new point.

        **6.** See how KNN decides where it belongs.
        """
    )


# ============================================================
# INITIAL DATA
# ============================================================

if st.session_state.points is None:

    initialize_experiment()


# ============================================================
# CURRENT DATA
# ============================================================

iteration = st.session_state.iteration

points = st.session_state.points

query = st.session_state.query


# ============================================================
# CONTROL BAR
# ============================================================

st.markdown("### 🎬 Watch the points evolve")

control1, control2, control3 = st.columns(
    [1, 2, 1]
)

with control1:

    if st.button(
        "⬅ Previous",
        use_container_width=True
    ):

        st.session_state.iteration = max(
            0,
            st.session_state.iteration - 1
        )

with control2:

    new_iteration = st.slider(
        "Iteration",
        min_value=0,
        max_value=TOTAL_ITERATIONS,
        value=iteration,
        step=1,
        label_visibility="collapsed"
    )

    st.session_state.iteration = new_iteration

with control3:

    if st.button(
        "Next ➡",
        use_container_width=True
    ):

        st.session_state.iteration = min(
            TOTAL_ITERATIONS,
            st.session_state.iteration + 1
        )


iteration = st.session_state.iteration


# ============================================================
# GENERATE CURRENT POSITIONS
# ============================================================

current_points, hidden_labels = create_clustered_positions(
    points,
    iteration
)


# ============================================================
# DETERMINE WHETHER QUERY SHOULD APPEAR
# ============================================================

show_query = (
    iteration >= TOTAL_ITERATIONS
)


# ============================================================
# KNN CLASSIFICATION
# ============================================================

prediction = None
nearest_indices = []

if show_query:

    model = KNeighborsClassifier(
        n_neighbors=K
    )

    model.fit(
        current_points,
        hidden_labels
    )

    prediction = int(
        model.predict(
            query.reshape(1, -1)
        )[0]
    )

    distances = np.sqrt(
        np.sum(
            (current_points - query) ** 2,
            axis=1
        )
    )

    nearest_indices = np.argsort(
        distances
    )[:K]


# ============================================================
# PLOT
# ============================================================

fig, ax = plt.subplots(
    figsize=(5, 3)
)

# ------------------------------------------------------------
# Draw points
# ------------------------------------------------------------

if iteration == 0:

    # ALL RANDOM POINTS LOOK THE SAME
    ax.scatter(
        current_points[:, 0],
        current_points[:, 1],
        s=90,
        color="#777777",
        alpha=0.85,
        edgecolors="white",
        linewidths=1
    )

else:

    # Once clustering starts, reveal colours
    for class_id in range(4):

        mask = hidden_labels == class_id

        ax.scatter(
            current_points[mask, 0],
            current_points[mask, 1],
            s=90,
            color=COLORS[class_id],
            alpha=0.85,
            edgecolors="white",
            linewidths=1,
            label=CLASS_NAMES[class_id]
        )


# ------------------------------------------------------------
# Query point
# ------------------------------------------------------------

if show_query:

    # Draw lines to nearest neighbours
    for idx in nearest_indices:

        ax.plot(
            [
                query[0],
                current_points[idx, 0]
            ],
            [
                query[1],
                current_points[idx, 1]
            ],
            color="#333333",
            linestyle="--",
            linewidth=1.5,
            alpha=0.55
        )

        # Highlight neighbour
        ax.scatter(
            current_points[idx, 0],
            current_points[idx, 1],
            s=230,
            facecolors="none",
            edgecolors="#222222",
            linewidths=2.5
        )

    # Query point
    ax.scatter(
        query[0],
        query[1],
        s=500,
        marker="*",
        color="#111111",
        edgecolors="white",
        linewidths=2,
        zorder=10
    )

    ax.annotate(
        "NEW POINT",
        xy=(query[0], query[1]),
        xytext=(10, 15),
        textcoords="offset points",
        fontsize=12,
        fontweight="bold"
    )


# ------------------------------------------------------------
# Axis
# ------------------------------------------------------------

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
    fontsize=13
)

ax.set_ylabel(
    "Y",
    fontsize=13
)

ax.grid(
    alpha=0.15
)

ax.set_title(
    f"Iteration {iteration} / {TOTAL_ITERATIONS}",
    fontsize=18,
    fontweight="bold"
)

if iteration > 0:

    ax.legend(
        loc="upper right",
        frameon=True
    )

st.pyplot(
    fig,
    use_container_width=True
)

plt.close(fig)


# ============================================================
# STATUS MESSAGE
# ============================================================

if iteration == 0:

    st.info(
        "👀 Look carefully! These points are completely random. "
        "There are no groups yet."
    )

elif iteration < 3:

    st.info(
        "🌱 Something is beginning to happen... "
        "The points are starting to move towards groups."
    )

elif iteration < 6:

    st.info(
        "🧩 The groups are becoming easier to recognize."
    )

elif iteration < TOTAL_ITERATIONS:

    st.info(
        "🎨 The clusters are becoming clearer. "
        "Can you predict where the next point might belong?"
    )

else:

    st.success(
        "✨ The groups have formed! Now let's introduce a new point."
    )


# ============================================================
# CLASSIFICATION SECTION
# ============================================================

if show_query:

    st.divider()

    st.markdown(
        "## ⭐ Meet the new point"
    )

    st.write(
        """
        A completely new point has appeared.

        It doesn't have a class yet.

        **Can KNN figure out where it belongs?**
        """
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Nearby points considered",
            K
        )

    with col2:

        st.metric(
            "Question",
            "Where does ★ belong?"
        )

    with col3:

        st.metric(
            "KNN's answer",
            CLASS_NAMES[prediction]
        )

    st.markdown(
        f"""
        <div class="prediction">
        🔎 KNN thinks the new point belongs to<br><br>
        {CLASS_NAMES[prediction]}
        </div>
        """,
        unsafe_allow_html=True
    )

    st.divider()

    st.markdown(
        "### 🗳️ Why?"

    )

    votes = hidden_labels[
        nearest_indices
    ]

    vote_counts = np.bincount(
        votes,
        minlength=4
    )

    vote_cols = st.columns(4)

    for i in range(4):

        with vote_cols[i]:

            st.markdown(
                f"""
                <div class="step-card">

                <div style="
                    font-size:22px;
                    color:{COLORS[i]};
                    font-weight:700;
                ">
                {CLASS_NAMES[i]}
                </div>

                <div class="big-number">
                {vote_counts[i]}
                </div>

                votes

                </div>
                """,
                unsafe_allow_html=True
            )

    st.write(
        f"""
        KNN looks at the **{K} closest points**.

        The group with the most nearby points wins.

        In this case:

        **{CLASS_NAMES[prediction]} wins! 🎉**
        """
    )


# ============================================================
# TEACHING SECTION
# ============================================================

st.divider()

st.markdown(
    "## 💡 What just happened?"
)

step1, step2, step3, step4 = st.columns(4)

with step1:

    st.markdown(
        """
        ### ① Random

        At the beginning, the points were completely random.

        No groups.
        No classes.
        """
    )

with step2:

    st.markdown(
        """
        ### ② Formation

        As the iterations progressed, nearby points began forming groups.
        """
    )

with step3:

    st.markdown(
        """
        ### ③ New point

        A new point appeared without a class.
        """
    )

with step4:

    st.markdown(
        """
        ### ④ Voting

        KNN looked at nearby points and used majority voting to decide.
        """
    )


# ============================================================
# FUN QUESTION
# ============================================================

if show_query:

    st.divider()

    st.markdown(
        "### 🤔 Your turn to think"
    )

    st.write(
        """
        Before clicking **Start Again**, look at the clusters.

        If another point appeared near the boundary between two groups,
        would you be confident about its class?

        **That's where KNN becomes interesting!**
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.markdown(
    """
    <div style="
        text-align:center;
        color:#888;
        padding:15px;
    ">
        🔎 The KNN Detective · An interactive machine-learning experiment
    </div>
    """,
    unsafe_allow_html=True
)
