import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from matplotlib.colors import ListedColormap


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Interactive KNN Classification",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 700;
        text-align: center;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        font-size: 18px;
        color: #666666;
        margin-bottom: 30px;
    }

    .prediction-box {
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        font-size: 25px;
        font-weight: bold;
        border: 2px solid #444444;
        margin-top: 10px;
    }

    .info-box {
        padding: 15px;
        border-radius: 10px;
        background-color: #f5f5f5;
        margin-bottom: 10px;
    }

    .small-text {
        font-size: 14px;
        color: #666666;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# TITLE
# ============================================================

st.markdown(
    '<div class="main-title">🎯 Interactive KNN Classification Lab</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Explore how K-Nearest Neighbours classifies a new point '
    'iteration by iteration'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# CLASS COLOURS
# ============================================================

CLASS_COLORS = {
    0: "#e74c3c",
    1: "#3498db",
    2: "#2ecc71",
    3: "#9b59b6"
}

CLASS_NAMES = {
    0: "Class A",
    1: "Class B",
    2: "Class C",
    3: "Class D"
}


# ============================================================
# SESSION STATE INITIALIZATION
# ============================================================

if "dataset" not in st.session_state:
    st.session_state.dataset = None

if "query_point" not in st.session_state:
    st.session_state.query_point = None

if "iteration" not in st.session_state:
    st.session_state.iteration = 1

if "seed" not in st.session_state:
    st.session_state.seed = 42

if "generated" not in st.session_state:
    st.session_state.generated = False


# ============================================================
# DATA GENERATION FUNCTION
# ============================================================

def generate_dataset(
    n_points,
    n_classes,
    spread,
    seed
):
    """
    Generate 2D clustered data.
    """

    rng = np.random.default_rng(seed)

    # Fixed cluster centres
    centres = np.array([
        [25, 25],
        [75, 25],
        [50, 75],
        [80, 75]
    ])

    points_per_class = n_points // n_classes
    remainder = n_points % n_classes

    X_list = []
    y_list = []

    for class_id in range(n_classes):

        count = points_per_class

        if class_id < remainder:
            count += 1

        centre = centres[class_id]

        points = rng.normal(
            loc=centre,
            scale=spread,
            size=(count, 2)
        )

        # Keep points inside fixed axis range
        points[:, 0] = np.clip(points[:, 0], 2, 98)
        points[:, 1] = np.clip(points[:, 1], 2, 98)

        X_list.append(points)
        y_list.extend([class_id] * count)

    X = np.vstack(X_list)
    y = np.array(y_list)

    # Shuffle the observations
    indices = rng.permutation(len(X))

    X = X[indices]
    y = y[indices]

    return X, y


# ============================================================
# QUERY POINT GENERATION
# ============================================================

def generate_query_point(seed=None):

    rng = np.random.default_rng(seed)

    x = rng.uniform(5, 95)
    y = rng.uniform(5, 95)

    return np.array([x, y])


# ============================================================
# KNN INFORMATION
# ============================================================

def get_knn_information(X, y, query, k):

    if len(X) == 0:
        return None

    k = min(k, len(X))

    distances = np.sqrt(
        np.sum(
            (X - query) ** 2,
            axis=1
        )
    )

    nearest_indices = np.argsort(distances)[:k]

    neighbour_classes = y[nearest_indices]

    unique_classes, counts = np.unique(
        neighbour_classes,
        return_counts=True
    )

    winning_class = unique_classes[
        np.argmax(counts)
    ]

    return {
        "distances": distances,
        "nearest_indices": nearest_indices,
        "neighbour_classes": neighbour_classes,
        "winning_class": int(winning_class),
        "unique_classes": unique_classes,
        "counts": counts
    }


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("⚙️ Experiment Controls")

st.sidebar.subheader("Dataset")

n_points = st.sidebar.slider(
    "Number of points",
    min_value=20,
    max_value=300,
    value=100,
    step=10
)

n_classes = st.sidebar.slider(
    "Number of classes",
    min_value=2,
    max_value=4,
    value=3,
    step=1
)

spread = st.sidebar.slider(
    "Cluster spread",
    min_value=2.0,
    max_value=20.0,
    value=8.0,
    step=0.5
)

seed = st.sidebar.number_input(
    "Random seed",
    min_value=0,
    max_value=9999,
    value=42,
    step=1
)

st.sidebar.subheader("KNN")

k_value = st.sidebar.slider(
    "K — Number of neighbours",
    min_value=1,
    max_value=20,
    value=5,
    step=1
)


# ============================================================
# GENERATE BUTTONS
# ============================================================

if st.sidebar.button(
    "🔄 Generate New Dataset",
    use_container_width=True
):

    X, y = generate_dataset(
        n_points=n_points,
        n_classes=n_classes,
        spread=spread,
        seed=seed
    )

    st.session_state.dataset = (X, y)

    st.session_state.query_point = generate_query_point(
        seed=seed + 100
    )

    st.session_state.iteration = 1
    st.session_state.generated = True


if st.sidebar.button(
    "🎯 Generate New Query Point",
    use_container_width=True
):

    if st.session_state.dataset is not None:

        st.session_state.query_point = generate_query_point()

        st.session_state.iteration = 1


if st.sidebar.button(
    "↩️ Reset",
    use_container_width=True
):

    st.session_state.dataset = None
    st.session_state.query_point = None
    st.session_state.iteration = 1
    st.session_state.generated = False


# ============================================================
# INITIAL DATASET
# ============================================================

if st.session_state.dataset is None:

    X, y = generate_dataset(
        n_points=n_points,
        n_classes=n_classes,
        spread=spread,
        seed=seed
    )

    st.session_state.dataset = (X, y)

    st.session_state.query_point = generate_query_point(
        seed=seed + 100
    )

    st.session_state.generated = True


X, y = st.session_state.dataset

query_point = st.session_state.query_point


# ============================================================
# ITERATION CONTROLS
# ============================================================

st.subheader("⏱️ Iteration Control")

max_iteration = len(X)

col1, col2, col3 = st.columns([1, 5, 1])

with col1:

    if st.button("⬅️ Previous"):

        st.session_state.iteration = max(
            1,
            st.session_state.iteration - 1
        )

with col2:

    iteration = st.slider(
        "Current iteration",
        min_value=1,
        max_value=max_iteration,
        value=st.session_state.iteration,
        step=1
    )

    st.session_state.iteration = iteration

with col3:

    if st.button("Next ➡️"):

        st.session_state.iteration = min(
            max_iteration,
            st.session_state.iteration + 1
        )


# ============================================================
# DATA FOR CURRENT ITERATION
# ============================================================

current_iteration = st.session_state.iteration

X_current = X[:current_iteration]
y_current = y[:current_iteration]


# ============================================================
# KNN CALCULATION
# ============================================================

knn_info = get_knn_information(
    X_current,
    y_current,
    query_point,
    k_value
)

nearest_indices = knn_info["nearest_indices"]

winning_class = knn_info["winning_class"]

nearest_distances = knn_info["distances"][
    nearest_indices
]


# ============================================================
# MAIN LAYOUT
# ============================================================

plot_col, info_col = st.columns(
    [2.2, 1]
)


# ============================================================
# PLOT
# ============================================================

with plot_col:

    st.subheader(
        f"📊 Dataset — Iteration {current_iteration}"
    )

    fig, ax = plt.subplots(
        figsize=(9, 7)
    )

    # --------------------------------------------------------
    # Decision regions
    # --------------------------------------------------------

    if current_iteration >= max(3, k_value):

        xx, yy = np.meshgrid(
            np.linspace(0, 100, 150),
            np.linspace(0, 100, 150)
        )

        grid_points = np.c_[
            xx.ravel(),
            yy.ravel()
        ]

        model = KNeighborsClassifier(
            n_neighbors=min(
                k_value,
                len(X_current)
            )
        )

        model.fit(
            X_current,
            y_current
        )

        predictions = model.predict(
            grid_points
        )

        predictions = predictions.reshape(
            xx.shape
        )

        cmap = ListedColormap(
            [
                "#fde2e2",
                "#dceeff",
                "#ddf7e4",
                "#eee0f7"
            ][:n_classes]
        )

        ax.contourf(
            xx,
            yy,
            predictions,
            alpha=0.35,
            cmap=cmap
        )

    # --------------------------------------------------------
    # Plot all current points
    # --------------------------------------------------------

    for class_id in range(n_classes):

        mask = y_current == class_id

        if np.any(mask):

            ax.scatter(
                X_current[mask, 0],
                X_current[mask, 1],
                s=65,
                alpha=0.85,
                color=CLASS_COLORS[class_id],
                label=CLASS_NAMES[class_id],
                edgecolors="white",
                linewidths=0.8
            )

    # --------------------------------------------------------
    # Highlight nearest neighbours
    # --------------------------------------------------------

    for idx in nearest_indices:

        neighbour = X_current[idx]

        ax.plot(
            [
                query_point[0],
                neighbour[0]
            ],
            [
                query_point[1],
                neighbour[1]
            ],
            linestyle="--",
            linewidth=1.2,
            color="black",
            alpha=0.6
        )

        ax.scatter(
            neighbour[0],
            neighbour[1],
            s=180,
            facecolors="none",
            edgecolors="black",
            linewidths=2
        )

    # --------------------------------------------------------
    # Query point
    # --------------------------------------------------------

    ax.scatter(
        query_point[0],
        query_point[1],
        s=300,
        marker="*",
        color="black",
        edgecolors="white",
        linewidths=1.5,
        zorder=10,
        label="New / Query Point"
    )

    # --------------------------------------------------------
    # Axes
    # --------------------------------------------------------

    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)

    ax.set_xlabel(
        "X axis",
        fontsize=12
    )

    ax.set_ylabel(
        "Y axis",
        fontsize=12
    )

    ax.set_title(
        f"KNN Classification | K = {k_value}",
        fontsize=16,
        fontweight="bold"
    )

    ax.grid(
        alpha=0.2
    )

    ax.legend(
        loc="upper right"
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

    st.subheader("🔎 Current Result")

    st.metric(
        "Current iteration",
        f"{current_iteration} / {max_iteration}"
    )

    st.metric(
        "Training points",
        current_iteration
    )

    st.metric(
        "K",
        k_value
    )

    st.write("### 🎯 Query Point")

    st.write(
        f"X = **{query_point[0]:.2f}**"
    )

    st.write(
        f"Y = **{query_point[1]:.2f}**"
    )

    st.divider()

    st.write("### 🗳️ Neighbour Voting")

    vote_data = pd.DataFrame({
        "Class": [
            CLASS_NAMES[int(c)]
            for c in knn_info["unique_classes"]
        ],
        "Votes": knn_info["counts"]
    })

    st.dataframe(
        vote_data,
        hide_index=True,
        use_container_width=True
    )

    st.divider()

    st.write("### 🏆 Prediction")

    st.markdown(
        f"""
        <div class="prediction-box">
            {CLASS_NAMES[winning_class]}
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# NEIGHBOUR DETAILS
# ============================================================

st.divider()

st.subheader("📍 K Nearest Neighbours")

neighbour_rows = []

for rank, idx in enumerate(
    nearest_indices,
    start=1
):

    neighbour_rows.append({
        "Rank": rank,
        "Class": CLASS_NAMES[int(y_current[idx])],
        "X": round(float(X_current[idx, 0]), 2),
        "Y": round(float(X_current[idx, 1]), 2),
        "Distance": round(
            float(nearest_distances[rank - 1]),
            3
        )
    })

neighbour_df = pd.DataFrame(
    neighbour_rows
)

st.dataframe(
    neighbour_df,
    hide_index=True,
    use_container_width=True
)


# ============================================================
# EXPLANATION
# ============================================================

st.divider()

st.subheader("🧠 How did KNN make this decision?")

st.markdown(
    f"""
    **Step 1 — Find the neighbours**

    The algorithm calculates the distance between the query point
    and every training point.

    **Step 2 — Select K neighbours**

    The closest **{k_value}** points are selected.

    **Step 3 — Voting**

    Each neighbour votes for its class.

    **Step 4 — Majority rule**

    The class receiving the largest number of votes becomes the
    predicted class.

    ### Current prediction

    **{CLASS_NAMES[winning_class]}**

    This means that among the selected nearest neighbours,
    **{CLASS_NAMES[winning_class]}** received the highest number
    of votes.
    """
)


# ============================================================
# EDUCATIONAL NOTE
# ============================================================

with st.expander(
    "📚 What should you observe while changing the iteration?"
):

    st.markdown(
        """
        ### Try this experiment

        1. Keep **K = 3**.
        2. Start from **Iteration 1**.
        3. Slowly move the iteration slider.
        4. Observe how the points gradually appear.
        5. Watch the KNN decision regions change.
        6. Observe which neighbours influence the query point.
        7. Increase K to **5, 7, 11**.
        8. Generate another query point.
        9. Compare the predictions.

        ### Important observation

        A small value of **K** makes the classifier more sensitive
        to nearby individual points.

        A larger value of **K** considers a larger neighbourhood
        and generally produces smoother decision regions.

        This demonstrates the fundamental **bias–variance trade-off**
        in KNN.
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.markdown(
    """
    <div style="text-align:center; color:#777;">
        🎯 Interactive KNN Classification Lab |
        Built with Streamlit & Python
    </div>
    """,
    unsafe_allow_html=True
)
