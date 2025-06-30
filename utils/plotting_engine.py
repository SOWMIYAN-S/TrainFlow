
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from mpl_toolkits.mplot3d import Axes3D

def plot_chart(df, plot_type, x_col, y_col, hue_col, multi_cols, style, alpha, color, rotation, grid, title, x_label, y_label, backend):
    fig = None

    if plot_type == "pairplot":
        fig = sns.pairplot(df[multi_cols], hue=hue_col if hue_col in multi_cols else None)
        return fig

    elif plot_type == "heatmap":
        corr_data = df[multi_cols].corr(numeric_only=True)
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(corr_data, annot=True, cmap="coolwarm", ax=ax)

    elif plot_type == "3d_scatter":
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(df[x_col], df[y_col], df[hue_col], c=color, alpha=alpha)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_zlabel(hue_col)
        ax.set_title(title)

    elif plot_type == "animated_line":
        fig = plt.figure()
        
        return fig

    elif backend == "Seaborn":
        fig, ax = plt.subplots(figsize=(10, 6))
        common_args = dict(data=df, x=x_col, y=y_col, hue=hue_col, alpha=alpha)
        if plot_type == "scatter":
            sns.scatterplot(**common_args, color=color, ax=ax)
        elif plot_type == "line":
            sns.lineplot(**common_args, color=color, ax=ax)
        elif plot_type == "bar":
            sns.barplot(**common_args, color=color, ax=ax)
        elif plot_type == "hist":
            sns.histplot(data=df, x=x_col, hue=hue_col, color=color, alpha=alpha, ax=ax)
        elif plot_type == "box":
            sns.boxplot(**common_args, ax=ax)
        elif plot_type == "violin":
            sns.violinplot(**common_args, ax=ax)
        elif plot_type == "kde":
            sns.kdeplot(data=df, x=x_col, hue=hue_col, fill=True, alpha=alpha, ax=ax)
        elif plot_type == "count":
            sns.countplot(data=df, x=x_col, hue=hue_col, color=color, ax=ax)
        elif plot_type == "lm":
            sns.lmplot(data=df, x=x_col, y=y_col, hue=hue_col)
            return plt.gcf()
        elif plot_type == "strip":
            sns.stripplot(**common_args, ax=ax)
        elif plot_type == "swarm":
            sns.swarmplot(**common_args, ax=ax)

    elif backend == "Matplotlib":
        fig, ax = plt.subplots(figsize=(10, 6))
        if plot_type == "scatter":
            ax.scatter(df[x_col], df[y_col], alpha=alpha, color=color)
        elif plot_type == "line":
            ax.plot(df[x_col], df[y_col], alpha=alpha, color=color)
        elif plot_type == "bar":
            ax.bar(df[x_col], df[y_col], alpha=alpha, color=color)
        elif plot_type == "area":
            df.set_index(x_col)[y_col].plot.area(alpha=alpha, color=color, ax=ax)
        elif plot_type == "pie":
            pie_data = df[y_col].value_counts()
            ax.pie(pie_data, labels=pie_data.index, autopct='%1.1f%%', startangle=90)
            ax.axis('equal')
        else:
            ax.text(0.5, 0.5, "Unsupported for Matplotlib backend", ha='center')

    elif backend == "Plotly":
        if plot_type == "scatter":
            fig = px.scatter(df, x=x_col, y=y_col, color=hue_col, opacity=alpha)
        elif plot_type == "line":
            fig = px.line(df, x=x_col, y=y_col, color=hue_col)
        elif plot_type == "bar":
            fig = px.bar(df, x=x_col, y=y_col, color=hue_col)
        elif plot_type == "pie":
            fig = px.pie(df, names=y_col)
        elif plot_type == "3d_scatter":
            fig = px.scatter_3d(df, x=x_col, y=y_col, z=hue_col, color=hue_col)

        if fig:
            fig.update_layout(title=title)

    
    if backend != "Plotly" and plot_type not in ["lm", "pairplot", "heatmap", "pie"]:
        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.tick_params(axis='x', rotation=rotation)
        if grid:
            ax.grid(True)

    return fig
