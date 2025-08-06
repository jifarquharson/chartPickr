import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import random

# --- Chart suggestion logic
def get_options_for_level(selected_option):
    dist_dict = {
        "Comparison": ["1.1 Between things", "1.2 Over time"],
        "Distribution": ["2.1 One variable", "2.2 Two variables", "2.3 Three variables"],
        "Composition": ["3.1 Dynamic over time", "3.2 Static"],
        "Relationship": ["4.1 Two variables", "4.2 Three variables"],

        "1.1 Between things": ["1.1.1 Two variables per item", "1.1.2 One variable per item"],
        "1.2 Over time": ["1.2.1 Many periods", "1.2.2 Few periods"],
        "1.2.1 Many periods": ["1.2.1.1 Cyclic", "1.2.1.2 Acyclic"],
        "1.2.1.1 Cyclic": ["Circular area chart"],
        "1.2.1.2 Acyclic": ["Line chart"],
        "1.2.2 Few periods": ["1.2.2.1 Few categories", "1.2.2.2 Many categories"],
        "1.2.2.1 Few categories": ["Column chart"],
        "1.2.2.2 Many categories": ["Multi-line chart"],
        "1.1.1 Two variables per item": ["Variable width column chart"],
        "1.1.2 One variable per item": ["1.1.2.1 Many categories", "1.1.2.2 Few categories"],
        "1.1.2.1 Many categories": ["Table or matrix"],
        "1.1.2.2 Few categories": ["1.1.2.2.1 Many items", "1.1.2.2.2 Few items"],
        "1.1.2.2.1 Many items": ["Bar chart"],
        "1.1.2.2.2 Few items": ["Column chart"],

        "2.1 One variable": ["2.1.1 Few data points", "2.1.2 Many data points"],
        "2.1.1 Few data points": ["Column histogram"],
        "2.1.2 Many data points": ["Line histogram"],
        "2.2 Two variables": ["Scatter (xy) plot"],
        "2.3 Three variables": ["3D area chart"],

        "3.1 Dynamic over time": ["3.1.1 Few periods", "3.1.2 Many periods"],
        "3.1.1 Few periods": ["3.1.1.1 Relative difference", "3.1.1.2 Absolute difference"],
        "3.1.1.1 Relative difference": ["Normalised stacked column chart"],
        "3.1.1.2 Absolute difference": ["Stacked column chart"],
        "3.1.2 Many periods": ["3.1.2.1 Relative difference", "3.1.2.2 Absolute difference"],
        "3.1.2.1 Relative difference": ["Normalised stacked area chart"],
        "3.1.2.2 Absolute difference": ["Stacked area chart"],
        "3.2 Static": ["3.2.1 Proportion or percentage", "3.2.2 Accumulation or subtraction", "3.2.3 Nested components"],
        "3.2.1 Proportion or percentage": ["Pie chart"],
        "3.2.2 Accumulation or subtraction": ["Waterfall chart"],
        "3.2.3 Nested components": ["Normalised column chart with subcomponents"],

        "4.1 Two variables": ["Scatter (xy) plot"],
        "4.2 Three variables": ["Bubble chart"]
    }
    return dist_dict.get(selected_option, [])

# --- Plot generator using seed
def plot_example(chart):
    st.markdown(f"### Example: {chart}")
    fig, ax = plt.subplots()

    if chart == "Line chart":
        y = np.cumsum(np.random.randn(10))
        ax.plot(y)
        ax.set_title("Line chart")

    elif chart == "Bar chart":
        labels = ["A", "B", "C", "D"]
        values = np.random.randint(5, 20, size=len(labels))
        ax.bar(labels, values)
        ax.set_title("Bar chart")

    elif chart == "Pie chart":
        sizes = np.random.randint(1, 10, size=3)
        labels = ["X", "Y", "Z"]
        ax.pie(sizes, labels=labels)
        ax.set_title("Pie chart")

    elif chart == "Scatter (xy) plot":
        x = np.random.randn(50)
        y = x + np.random.randn(50) * 0.5
        ax.scatter(x, y)
        ax.set_title("Scatter plot")

    else:
        ax.text(0.5, 0.5, f"[{chart}]", ha='center', va='center', fontsize=16)
        ax.axis('off')

    st.pyplot(fig)

# --- App start
st.set_page_config(page_title="Chart Picker", layout="centered")
st.title("Chart Picker")
st.markdown("Follow the prompts to discover a chart type suited to your data.")

# --- Seed input
user_seed = st.text_input("Enter a seed (any number or word)", value="default-seed")
random.seed(user_seed)
np.random.seed(abs(hash(user_seed)) % (2**32))

# --- Reset button
if st.button("🔄 Reset selection"):
    st.session_state.clear()
    st.experimental_rerun()

# --- Chart selection logic
level = 0
current_key = "level_0"
selected = st.selectbox("Select chart category", options=["Comparison", "Distribution", "Relationship", "Composition"], key=current_key)

while selected:
    options = get_options_for_level(selected)
    if not options:
        plot_example(selected)
        break
    level += 1
    next_key = f"level_{level}"
    selected = st.selectbox(f"Select {selected}", options=options, key=next_key)
