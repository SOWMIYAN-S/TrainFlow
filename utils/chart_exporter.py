

from io import BytesIO
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pandas as pd

def export_chart_as_png(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches='tight')
    buf.seek(0)
    return buf

def export_chart_as_gif(df: pd.DataFrame, x_col: str, y_col: str, color: str):
    fig, ax = plt.subplots()
    line, = ax.plot([], [], color=color)

    ax.set_xlim(df[x_col].min(), df[x_col].max())
    ax.set_ylim(df[y_col].min(), df[y_col].max())
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title("Animated Line Chart")

    def init():
        line.set_data([], [])
        return line,

    def update(frame):
        line.set_data(df[x_col][:frame], df[y_col][:frame])
        return line,

    anim = FuncAnimation(fig, update, init_func=init, frames=len(df), interval=100, blit=True)

    buf = BytesIO()
    anim.save(buf, writer='pillow', format='gif')
    buf.seek(0)
    return buf
