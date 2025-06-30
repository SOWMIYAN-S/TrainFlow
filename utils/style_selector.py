
import seaborn as sns
import matplotlib.pyplot as plt

def apply_style(style_name: str):
    if style_name in ["darkgrid", "whitegrid", "white", "ticks", "dark"]:
        sns.set_style(style_name)
        plt.style.use("default")
    else:
        sns.set_style("whitegrid")
