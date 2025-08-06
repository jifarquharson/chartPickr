import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import random

# --- Chart-generating functions ---

def column_chart(seed):
    np.random.seed(seed)
    categories = ['A', 'B', 'C', 'D', 'E']
    values1 = [np.random.randint(0, 10) for i in range(5)]
    values2 = [np.random.randint(0, 10) for i in range(5)]
    bar_width = 0.35
    
    fig = plt.figure(1, figsize = (1,1), dpi = 200)
    ax = fig.add_subplot(111)
    ax.bar(np.arange(len(categories)) - bar_width/2, values1,
           bar_width, label='Variable 1', alpha=0.85, color = "k")
    ax.bar(np.arange(len(categories)) + bar_width/2, values2,
           bar_width, label='Variable 2', alpha=0.85, edgecolor = "k", color = "r")
    ax.set_xlabel('Categories', fontsize = "xx-small")
    ax.set_ylabel('Values', fontsize = "xx-small")
    ax.yaxis.set_ticks([])
    ax.xaxis.set_ticks([])
    ax.set_title('Column Chart', fontsize = "xx-small")
    plt.show()

def bar_chart(seed):
    np.random.seed(seed)
    categories = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
    values1 = [np.random.randint(0, 10) for i in range(10)]
    values2 = [np.random.randint(0, 10) for i in range(10)]
    bar_height = 0.9
    fig = plt.figure(1, figsize = (1,1), dpi = 200)
    ax = fig.add_subplot(111)
    ax.barh(np.arange(len(categories)), values1,
            bar_height, label='Variable 1', color='k', alpha=1.)
    ax.barh(np.arange(len(categories)), values2,
            bar_height, left=12, label='Variable 2', color='r', alpha=1.)

    ax.set_ylabel('Categories', fontsize = "xx-small")
    ax.set_xlabel('Values', fontsize = "xx-small")
    ax.yaxis.set_ticks([])
    ax.xaxis.set_ticks([])
    ax.set_title('Bar Chart', fontsize = "xx-small")
    plt.show()

def xy_chart(seed):
    np.random.seed(seed)
    values1 = [np.random.randint(0, 10)+i for i in range(10)]
    values2 = [np.random.random()+i for i in range(10)]
    
    fig = plt.figure(1, figsize = (1,1), dpi = 200)
    ax = fig.add_subplot(111)
    ax.scatter(values1, values2,
            color='k',marker = ".", alpha=1.)
  
    ax.set_ylabel('Y', fontsize = "xx-small")
    ax.set_xlabel('X', fontsize = "xx-small")
    ax.yaxis.set_ticks([])
    ax.xaxis.set_ticks([])
    ax.set_title('Scatter Plot', fontsize = "xx-small")
    ax.set_ylim(ymin=0)
    ax.set_xlim(xmin=0)
    plt.show()


def bar_hist_chart(seed):
    np.random.seed(seed)
    values1 = [np.random.randint(0, 10) for i in range(30)]

    fig = plt.figure(1, figsize = (1,1), dpi = 200)
    ax = fig.add_subplot(111)
    ax.hist(values1, bins = 6,
            color='k',alpha=1.)

    ax.yaxis.set_ticks([])
    ax.xaxis.set_ticks([])
    ax.set_title('Histogram Plot (Bar)', fontsize = "xx-small")
    plt.show()


def line_hist_chart(seed):
    np.random.seed(seed)
    
    fig = plt.figure(1, figsize = (1,1), dpi = 200)
    ax = fig.add_subplot(111)
    
    values = np.random.normal(loc=0, scale=1, size=1000)
    
    x = np.linspace(-5,5, 100)
    p1 = norm.pdf(x, np.mean(values), np.std(values))
    ax.hist(values, bins=30, density=True, alpha=1, label='Distribution 1',
            histtype="step", color = "k")
    ax.plot(x, p1, 'r', linewidth=1, alpha = 0.5)
    ax.yaxis.set_ticks([])
    ax.xaxis.set_ticks([])
    ax.set_title('Histogram Plot (Line)', fontsize = "xx-small")
    plt.show()

def line_chart(seed):
    np.random.seed(seed)
    
    fig = plt.figure(1, figsize = (1,1), dpi = 200)
    ax = fig.add_subplot(111)
    
    values1 = np.cumsum(np.random.randn(30))
    
    
    ax.plot(values1,alpha=1,
           color = "k", lw=0.85)
    ax.yaxis.set_ticks([])
    ax.xaxis.set_ticks([])
    ax.set_ylim(ymin=min(values1)-1, ymax=max(values1)+1)
    ax.set_title('Line Chart', fontsize = "xx-small")
    plt.show()

def mult_line_chart(seed):
    np.random.seed(seed)
    
    fig = plt.figure(1, figsize = (1,1), dpi = 200)
    ax = fig.add_subplot(111)
    
    values1 = np.cumsum(np.random.randn(30))
    values2 = np.cumsum(np.random.randn(30))
    values3 = np.cumsum(np.random.randn(30))

    
    ax.plot(values1,alpha=1,color = "k", lw=0.85)
    ax.plot(values2,alpha=1,color = "grey",ls="-.", lw=0.85)
    ax.plot(values3,alpha=1,color = "darkgrey",ls="--", lw=0.85)
    ax.yaxis.set_ticks([])
    ax.xaxis.set_ticks([])
    ax.set_title('Line Chart', fontsize = "xx-small")
    plt.show()
    
def threeD_chart(seed):
    np.random.seed(seed)
    
    fig = plt.figure(1, figsize = (1,1), dpi = 200)
    ax = fig.add_subplot(111, projection='3d')
    
    x = np.linspace(-5, 5, 50)
    y = np.linspace(-5, 5, 50)
    X, Y = np.meshgrid(x, y)
    
    Zs = [np.sin(np.sqrt(X**2 + Y**2)),
         np.tan(np.sqrt(X**2 + Y**2)),
         np.cos(np.sqrt(X**2 + Y**2))]
    
    ax.plot_surface(X, Y,Zs[np.random.randint(0,3)],
                                       cmap='Greys', alpha=0.7, label='Z')

    ax.xaxis.set_ticks([])
    ax.yaxis.set_ticks([])
    ax.zaxis.set_ticks([])
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.edgecolor = 'black'
    ax.yaxis.pane.edgecolor = 'black'
    ax.zaxis.pane.edgecolor = 'black'

    ax.set_title('3D Area Chart', fontsize = "xx-small")
    plt.show()

def bubble_chart(seed):
    np.random.seed(seed)
    values1 = [np.random.randint(0, 10)+i for i in range(10)]
    values2 = [np.random.random()+i for i in range(10)]
    values3 = [np.random.randint(20, 120)*2 for i in range(10)]
    
    fig = plt.figure(1, figsize = (1,1), dpi = 200)
    ax = fig.add_subplot(111)
    ax.scatter(values1, values2,s=values3,
            color='k',marker = ".", alpha=1.)
  
    ax.set_ylabel('Y', fontsize = "xx-small")
    ax.set_xlabel('X', fontsize = "xx-small")
    ax.yaxis.set_ticks([])
    ax.xaxis.set_ticks([])
    ax.set_title('Bubble Plot', fontsize = "xx-small")
    ax.set_ylim(ymin=0)
    ax.set_xlim(xmin=0)
    plt.show()

def waterfall_chart(seed):
    np.random.seed(seed)
    cats = np.arange(1,13,1)
    change = [np.random.randint(-100, 100) for i in range(11)]
    change.insert(0,200)
    start = np.cumsum(change)-change
    df = pd.DataFrame({'cat':cats, "change": change, "start":start})
    
    fig = plt.figure(1, figsize = (1,1), dpi = 200)
    ax = fig.add_subplot(111)
    ax.bar(np.arange(1,13,1),
        change,
        bottom=start,
        color =df["change"].apply(lambda x: 'k' if x > 0 else 'lightgrey'))
    ax.yaxis.set_ticks([])
    ax.xaxis.set_ticks([])
    ax.set_title('Waterfall Chart', fontsize = "xx-small")

    plt.show()

def stack_col_chart(seed):
    np.random.seed(seed)
    categories = [1,2,3]
    values = np.array([[10, 20, 30], [15, 25, 35], [25, 35, 45]])
    
    fig = plt.figure(1, figsize = (1,1), dpi = 200)
    ax = fig.add_subplot(111)
    ax.bar(categories, values[0], label='Component 1', color = "grey")
    ax.bar(categories, values[1], bottom=values[0], label='Component 2', color= "lightgrey")
    ax.bar(categories, values[2], bottom=np.sum(values[:2], axis=0),
            label='Component 3', color= "darkgrey")
    ax.set_xlabel('Categories', fontsize = "xx-small")
    ax.set_ylabel('Values', fontsize = "xx-small")
    ax.yaxis.set_ticks([])
    ax.xaxis.set_ticks([])
    ax.set_title('Stacked Column Chart', fontsize = "xx-small")

    plt.show()


def stack_area_chart(seed):
    np.random.seed(seed)
    categories = [1,2,3]
    values = np.array([[10, 20, 30], [15, 25, 35], [25, 35, 45]])
    
    fig = plt.figure(1, figsize = (1,1), dpi = 200)
    ax = fig.add_subplot(111)
    
    ax.stackplot(categories, values[0], values[1], values[2],
                 labels=['Component 1', 'Component 2', 'Component 3'], alpha=1.0,
                colors= ["lightgrey", "darkgrey", "grey"])
    ax.set_xlabel('X', fontsize = "xx-small")
    ax.set_ylabel('Values', fontsize = "xx-small")
    ax.yaxis.set_ticks([])
    ax.xaxis.set_ticks([])
    ax.set_title('Stacked Area Chart', fontsize = "xx-small")

    plt.show()


def norm_stack_area_chart(seed):
    np.random.seed(seed)
    categories = [1,2,3]
    values = np.array([[10, 20, 30], [15, 25, 35], [25, 35, 45]])
    norm_values = values/np.sum(values, axis = 0)*100

    fig = plt.figure(1, figsize = (1,1), dpi = 200)
    ax = fig.add_subplot(111)
    
    ax.stackplot(categories, norm_values[0], norm_values[1], norm_values[2],
                 labels=['Component 1', 'Component 2', 'Component 3'], alpha=1.0,
                colors= ["lightgrey", "darkgrey", "grey"])
    ax.set_xlabel('X', fontsize = "xx-small")
    ax.set_ylabel('%', fontsize = "xx-small")
    ax.yaxis.set_ticks([])
    ax.xaxis.set_ticks([])
    ax.set_title('Normalised Stacked Area Chart', fontsize = "xx-small")

    plt.show()

def plot_matrix():
    
    iris = load_iris()
    iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
    iris_df['target'] = iris.target
    target_names = iris.target_names
    fig = scatter_matrix(iris_df.iloc[:, :-1],
                   alpha=0.8,
                   c=iris_df['target'],  figsize=(2, 2), cmap="Greys", diagonal='hist',
                        hist_kwds={'color':'lightgrey','edgecolor':'k', 'histtype':'stepfilled'})

    for ax in fig.flatten():
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_ylabel("")
        ax.set_xlabel("")
    plt.show()

def circ_area_chart(seed):
    np.random.seed(seed)
    theta = np.linspace(0, 2*np.pi, 12, endpoint=True)
    r = [3, 3, 3, 4, 2, 5, 3, 3, 3, 3, 3, 3]
    r = np.roll(r,np.random.randint(0,11))
    fig = plt.figure(1, figsize = (1,1), dpi = 200)
    ax = fig.add_subplot(111, projection='polar')
    ax.fill_between(theta, 0, r, color ="k", alpha=1.)
    ax.yaxis.set_ticks([])
    ax.xaxis.set_ticks([])
    ax.set_title('Circular Area Chart', fontsize = "xx-small")
    plt.show()

def pie_chart(seed):
    np.random.seed(seed)
    values = [np.random.random() for i in range(6)]
    
    fig = plt.figure(1, figsize = (1,1), dpi = 200)
    ax = fig.add_subplot(111)#, projection='polar')
    ax.pie(values, startangle=30, colors="k", explode = [0.05, 0.05, 0.4,  0.05, 0.05, 0.05])
    ax.yaxis.set_ticks([])
    ax.xaxis.set_ticks([])
    ax.set_title('Pie Chart', fontsize = "xx-small")
    plt.show()

def norm_component_chart(seed):
    np.random.seed(seed)
    categories = [1,2,3]
    values = np.array([[10, 20, 30], [15, 25, 35], [25, 35, 45]])
    norm_values = values/np.sum(values, axis = 0)*100

    fig = plt.figure(1, figsize = (1,1), dpi = 200)
    ax = fig.add_subplot(111)
    ax.bar(categories, norm_values[0], label='Component 1', color = "k", width=0.5)
    ax.bar(categories, norm_values[1], bottom=norm_values[0], label='Component 2', color= "lightgrey", width=0.5)
    ax.bar(categories, norm_values[2], bottom=np.sum(norm_values[:2], axis=0),
            label='Component 3', color= "darkgrey", width=0.5)
    ax.set_xlabel('Categories', fontsize = "xx-small")
    ax.set_ylabel('%', fontsize = "xx-small")
    ax.yaxis.set_ticks([])
    ax.xaxis.set_ticks([])
    ax.set_title('Normalised Component Chart', fontsize = "xx-small")

    plt.plot([categories[0]+.25, categories[1]-.25],
             [norm_values[0][0], 100], "k", ls="-.", lw=.5)

    plt.plot([categories[1]+.25, categories[2]-.25],
             [norm_values[0][1], 100], "K", ls="-.", lw=.5)
    
    plt.show()
    
def var_width_chart(seed):
    np.random.seed(seed)
    categories = ['A', 'B', 'C', 'D', 'E']
    values = [np.random.randint(0, 10) for i in range(5)]
    widths = [np.random.random() for i in range(5)]
    fig = plt.figure(1, figsize = (1,1), dpi = 200)
    ax = fig.add_subplot(111)
    ax.bar(categories, values, width=widths, align='center', alpha=1, color = "k")
    ax.set_xlabel('Categories', fontsize = "xx-small")
    ax.set_ylabel('Values', fontsize = "xx-small")
    ax.yaxis.set_ticks([])
    ax.xaxis.set_ticks([])
    ax.set_title('Variable Width Column Chart', fontsize = "xx-small")
    
def norm_stack_col_chart(seed):
    np.random.seed(seed)
    categories = [1,2,3]
    values = np.array([[10, 20, 30], [15, 25, 35], [25, 35, 45]])
    norm_values = values/np.sum(values, axis = 0)*100
    
    fig = plt.figure(1, figsize = (1,1), dpi = 200)
    ax = fig.add_subplot(111)
    ax.bar(categories, norm_values[0], label='Component 1', color = "lightgrey")
    ax.bar(categories, norm_values[1], bottom=norm_values[0], label='Component 2', color= "darkgrey")
    ax.bar(categories, norm_values[2], bottom=np.sum(norm_values[:2], axis=0),
            label='Component 3', color= "grey")
    ax.set_xlabel('Categories', fontsize = "xx-small")
    ax.set_ylabel('%', fontsize = "xx-small")
    ax.yaxis.set_ticks([])
    ax.xaxis.set_ticks([])
    ax.set_title('Normalised stacked Column Chart', fontsize = "xx-small")

    plt.show()
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
