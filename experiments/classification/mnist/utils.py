import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D  # <-- add this at the top
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter
from scipy.stats import mannwhitneyu, pearsonr, shapiro, spearmanr, ttest_ind


def parse_config_from_path(path):
    # Works for .../ce_adam_lr=0.001/all/shifted_metrics.csv (with or without leading slash)
    match = re.search(r"([a-zA-Z0-9]+)_([a-zA-Z0-9]+)_lr=([0-9.]+)/all/", path)
    if match:
        loss, optimizer, lr = match.groups()
        return loss, optimizer, lr
    print(f"Warning: Could not parse config from path: {path}")
    return None, None, None


def parse_model_seed(name):
    # Extract model and seed from Name column like "model/seed_42"
    match = re.match(r"([^/]+)/seed_(\d+)", name)
    if match:
        model, seed = match.groups()
        return model, seed
    # If not matching, fallback to name and None
    return name, None


def load_and_combine_metrics(sources, metrics_csv):
    records = []
    for source in sources:
        csv_path = os.path.join(source, "all", metrics_csv)
        print(f"Found CSV: {csv_path}")
        loss, optimizer, lr = parse_config_from_path(csv_path)
        print(f"Parsed config: loss={loss}, optimizer={optimizer}, lr={lr}")
        try:
            df = pd.read_csv(csv_path)
            for _, row in df.iterrows():
                name = row.get("Name", "")
                model, seed = parse_model_seed(name)
                records.append(
                    {
                        "loss": loss,
                        "optimizer": optimizer,
                        "lr": lr,
                        "model": model,
                        "seed": seed,
                        **row.to_dict(),
                    }
                )
        except Exception as e:
            print(f"Failed to read {csv_path}: {e}")

    if not records:
        raise ValueError("No records found. Check your directory structure and parsing.")
    df = pd.DataFrame(records)
    df.drop(columns=["Name"], inplace=True, errors="ignore")
    df.set_index(["loss", "optimizer", "lr", "model", "seed"], inplace=True)
    return df


def plot_boxplots(
    df, metric, groupby, title, save_path, x_label=None, y_labels=None, format_xtick=None
):
    """Plots one horizontal boxplot per group (e.g., loss), stacked vertically with a shared x-axis.

    Args:
        df (pd.DataFrame): DataFrame with metrics.
        metric (str): The metric column to plot (e.g., 'ECE').
        groupby (str): The column to group by (e.g., 'loss').
        title (str): Title of the plot.
        save_path (str): Path to save the figure.
        x_label (str, optional): Label for the x-axis.
        y_labels (list, optional): List of y-axis labels for each group.
        format_xtick (callable, optional): Function to format x-tick labels.
    """
    data = df.reset_index()
    groups = data[groupby].unique()
    n_groups = len(groups)
    fig, axes = plt.subplots(n_groups, 1, figsize=(8, 3 * n_groups), sharex=True)
    tilt_x_labels = n_groups > 3

    if n_groups == 1:
        axes = [axes]

    for ax, group in zip(axes, groups, strict=False):
        group_data = data[data[groupby] == group]
        sns.boxplot(data=group_data, x=metric, orient="h", ax=ax)
        if y_labels is not None:
            ax.set_ylabel(y_labels[groups.tolist().index(group)])
        else:
            ax.set_ylabel(group)
        ax.set_title(None)
        ax.grid(True, axis="x", linestyle="--", alpha=0.5)
        if format_xtick is not None:
            ax.set_xticklabels([format_xtick(x) for x in ax.get_xticks()])

    if x_label is not None:
        axes[-1].set_xlabel(x_label)
    else:
        axes[-1].set_xlabel(metric)
    # Tilt x-labels if there are more than 3 groups for readability
    if tilt_x_labels:
        plt.setp(axes[-1].get_xticklabels(), rotation=45, ha="right")
    fig.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)
    print(f"Saved boxplot figure to {save_path}")


def plot_boxplots_multi(
    dfs,
    target,
    groupby,
    df_label_mapping,
    colors=None,
    title=None,
    group_label_mapping=None,
    target_label=None,
    save_path=None,
    format_ticks=None,
    use_legend=False,
    title_size=18,
    label_size=14,
    sharey=True,
    statistic=None,
    signif_level=0.05,
    plot_type="box",
    filter_outliers=False,
    outlier_k=3.0,
    subgroupby=None,
    subgroup_colors=None,
    subgroup_label_mapping=None,
    central_tendency=None,
    marker_size=6,
    subgroup_palette_mapping=None,  # <--- NEW ARGUMENT
):
    """Plots vertical boxplots, violinplots, or swarmplots for each dataframe, grouped and color-coded by subgroup.
    All plots share the same y-axis (`target`). Uses a legend for subgroup colors or x-labels.

    Args:
        dfs (dict): Dictionary of DataFrames to plot.
        target (str): Column name of the target variable to construct plots for.
        groupby (str): Column to group the plots by (x-axis).
        df_label_mapping (dict): Dictionary mapping DataFrame keys in `dfs` to their labels.
        colors (dict, optional): Color mapping for each group value.
        title (str, optional): Figure title.
        group_label_mapping (dict, optional): Mapping of group values to display labels (order matters).
        target_label (str, optional): Y-axis label for the target variable.
        save_path (str, optional): Where to save the figure.
        format_ticks (callable, optional): Functions to format y-tick labels.
        use_legend (bool, optional): Whether to show a legend for group colors or use x-labels.
        plot_type (str): "box", "violin", or "swarm".
        subgroupby (str, optional): Column for subgroups (hue, e.g. 'loss') for swarm.
        subgroup_colors (dict, optional): Color mapping for subgroups.
        subgroup_label_mapping (dict, optional): Label mapping for subgroups.
        central_tendency (str or None): "mean", "median", or None (for swarm only).
    """
    n_sets = len(df_label_mapping)
    central_marker_size = marker_size * 1.5

    # Main groups (x-axis)
    if group_label_mapping is not None:
        groups = list(group_label_mapping.keys())
        group_labels = list(group_label_mapping.values())
    else:
        groups = sorted(set().union(*[df.reset_index()[groupby].unique() for df in dfs.values()]))
        group_labels = groups

    n_groups = len(groups)
    tilt_x_labels = n_groups > 3

    # Subgroups (hue)
    if plot_type == "swarm" and subgroupby is not None:
        all_subgroups = set()
        for df in dfs.values():
            all_subgroups.update(df.reset_index()[subgroupby].unique())
        if subgroup_label_mapping is not None:
            subgroups = list(subgroup_label_mapping.keys())
            subgroup_labels = list(subgroup_label_mapping.values())
        else:
            subgroups = sorted(all_subgroups)
            subgroup_labels = subgroups
        n_subgroups = len(subgroups)
        # --- NEW: Build subgroup_colors from palette mapping ---
        if subgroup_palette_mapping is not None:
            subgroup_colors = {}
            for subgroup in subgroups:
                palette_name = subgroup_palette_mapping.get(subgroup, "Set1")
                palette = sns.color_palette(palette_name, len(groups))
                for i, group in enumerate(groups):
                    subgroup_colors[(group, subgroup)] = palette[i]
        elif subgroup_colors is None:
            palette = sns.color_palette("Set1", n_subgroups)
            subgroup_colors = {g: palette[i] for i, g in enumerate(subgroups)}

        use_legend = use_legend or (
            subgroupby is not None and subgroups is not None and len(subgroups) > 1
        )
    else:
        subgroups = None
        subgroup_labels = None
        subgroup_colors = None

    if colors is None:
        palette = sns.color_palette("Blues", n_groups)
        colors = {g: palette[i] for i, g in enumerate(groups)}

    fig, axes = plt.subplots(1, n_sets, figsize=(5 * n_sets, 5), sharey=sharey)
    if n_sets == 1:
        axes = [axes]

    for i, (ax, (df_key, df_label)) in enumerate(zip(axes, df_label_mapping.items(), strict=False)):
        data = dfs[df_key].reset_index()
        if filter_outliers:
            filtered = []
            if plot_type == "swarm" and subgroupby is not None:
                for group in groups:
                    for subgroup in subgroups:
                        group_data = data[(data[groupby] == group) & (data[subgroupby] == subgroup)]
                        if not group_data.empty:
                            q1 = group_data[target].quantile(0.25)
                            q3 = group_data[target].quantile(0.75)
                            iqr = q3 - q1
                            lower = q1 - outlier_k * iqr
                            upper = q3 + outlier_k * iqr
                            filtered.append(
                                group_data[
                                    (group_data[target] >= lower) & (group_data[target] <= upper)
                                ]
                            )
                data = pd.concat(filtered, ignore_index=True) if filtered else data
            else:
                for group in groups:
                    group_data = data[data[groupby] == group]
                    if not group_data.empty:
                        q1 = group_data[target].quantile(0.25)
                        q3 = group_data[target].quantile(0.75)
                        iqr = q3 - q1
                        lower = q1 - outlier_k * iqr
                        upper = q3 + outlier_k * iqr
                        filtered.append(
                            group_data[
                                (group_data[target] >= lower) & (group_data[target] <= upper)
                            ]
                        )
                data = pd.concat(filtered, ignore_index=True) if filtered else data

        plot_kwargs = dict(data=data, x=groupby, y=target, order=groups, ax=ax, legend=False)

        if plot_type == "box":
            plot_kwargs["hue"] = groupby
            plot_kwargs["palette"] = colors
            plot_kwargs["legend"] = False
            sns.boxplot(**plot_kwargs)
        elif plot_type == "violin":
            plot_kwargs["hue"] = groupby
            plot_kwargs["palette"] = colors
            plot_kwargs["legend"] = False
            sns.violinplot(**plot_kwargs, inner="box")
        elif plot_type == "swarm":
            for k, subgroup in enumerate(subgroups):
                subgroup_data = data[data[subgroupby] == subgroup]
                # Determine the palette for this subgroup
                if subgroup_palette_mapping is not None:
                    palette_name = subgroup_palette_mapping.get(subgroup, "Set1")
                    palette = sns.color_palette(palette_name, len(groups))
                elif subgroup_colors is not None:
                    # Use the single color for all groups in this subgroup
                    single_color = subgroup_colors[subgroup]
                    palette = [single_color for _ in groups]
                else:
                    # Default: use Set1 palette for subgroups
                    default_palette = sns.color_palette("Set1", len(subgroups))
                    single_color = default_palette[k]
                    palette = [single_color for _ in groups]
                group_palette = {g: palette[i] for i, g in enumerate(groups)}
                sns.stripplot(
                    data=subgroup_data,
                    x=groupby,
                    y=target,
                    order=groups,
                    palette=group_palette,
                    ax=ax,
                    dodge=True,
                    size=marker_size,
                    alpha=0.8,
                    zorder=1,
                    legend=False,
                    jitter=1.0,
                )
                # --- Plot central tendency marker for each group in this subgroup ---
                if central_tendency:
                    for j, group in enumerate(groups):
                        group_vals = subgroup_data[subgroup_data[groupby] == group][target].dropna()
                        if len(group_vals) == 0:
                            continue
                        if central_tendency == "mean":
                            ct = group_vals.mean()
                        elif central_tendency == "median":
                            ct = group_vals.median()
                        else:
                            raise ValueError("central_tendency must be 'mean' or 'median'")
                        n_sub = len(subgroups)
                        width = 0.6
                        offset = (k - (n_sub - 1) / 2) * (width / n_sub)
                        x_pos = j + offset
                        ax.scatter(
                            x_pos,
                            ct,
                            color=group_palette[group],
                            edgecolor="black",
                            s=central_marker_size**2,
                            marker="D",
                            zorder=3,
                            label=None,
                        )
        else:
            raise ValueError("plot_type must be 'box', 'violin', or 'swarm'")

        ax.set_title(df_label, fontsize=label_size)
        if use_legend and plot_type != "swarm":
            ax.set_xticks([])
            ax.set_xticklabels([])
        else:
            ax.set_xticks(range(len(groups)))
            ax.set_xticklabels(group_labels, fontsize=label_size)
        ax.set_xlabel(None)

        if tilt_x_labels:
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

        # Statistical test (not shown for swarm)
        stat_str = ""
        if plot_type in ("box", "violin") and statistic is not None and n_groups == 2:
            group1 = data[data[groupby] == groups[0]][target].dropna()
            group2 = data[data[groupby] == groups[1]][target].dropna()
            chosen_stat = statistic
            if statistic == "auto":
                chosen_stat = auto_statistic(group1, group2, signif_level)
            if chosen_stat == "t-test":
                stat, p = ttest_ind(group1, group2, equal_var=False)
                stat_str = f"t={stat:.2f}"
            elif chosen_stat == "mannwhitney":
                stat, p = mannwhitneyu(group1, group2, alternative="two-sided")
                stat_str = f"U={stat:.1f}"
            else:
                stat_str = "Unknown test"
            if p < signif_level:
                stars, p_str = significance_stars_and_exact_p(p)
                stat_str += f" {stars} [{p_str}]"
            ax.text(
                0.98,
                0.02,
                stat_str,
                transform=ax.transAxes,
                fontsize=label_size - 2,
                color="black",
                ha="right",
                va="bottom",
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=1),
            )

        # Y-tick logic
        if i == 0:
            ax.tick_params(axis="y", which="both", left=False, labelleft=True, labelsize=label_size)
            if target_label:
                ax.set_ylabel(target_label, fontsize=label_size)
            else:
                ax.set_ylabel(target, fontsize=label_size)
        else:
            ax.tick_params(
                axis="y", which="both", left=False, labelleft=(not sharey), labelsize=label_size
            )
            ax.set_ylabel("")

        if format_ticks is not None:
            ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: format_ticks(y)))
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.grid(axis="y", linestyle="--", alpha=0.5, zorder=0)
    if title:
        fig.suptitle(title, fontsize=title_size)
    plt.tight_layout()
    if use_legend:
        if plot_type == "swarm" and subgroupby is not None:
            legend_labels = subgroup_labels
            legend_handles = []
            for i, g in enumerate(subgroups):
                if subgroup_palette_mapping is not None:
                    palette_name = subgroup_palette_mapping.get(g, "Set1")
                    palette = sns.color_palette(palette_name, len(groups))
                    mid_color = palette[len(palette) // 2]
                    legend_handles.append(Patch(facecolor=mid_color, label=legend_labels[i]))
                else:
                    legend_handles.append(
                        Patch(facecolor=subgroup_colors[g], label=legend_labels[i])
                    )
            # --- Add marker type legend if central tendency is set ---
            if central_tendency:
                legend_handles.append(
                    Line2D(
                        [0],
                        [0],
                        marker="o",
                        color="lightgrey",
                        linestyle="None",
                        markersize=marker_size,
                        label=groupby.capitalize(),
                    )
                )
                legend_handles.append(
                    Line2D(
                        [0],
                        [0],
                        marker="D",
                        color="lightgrey",
                        linestyle="None",
                        markersize=central_marker_size,
                        label=central_tendency.capitalize(),
                        markeredgecolor="black",
                    )
                )
        else:
            legend_labels = group_labels
            legend_handles = [
                Patch(facecolor=colors[g], label=legend_labels[i]) for i, g in enumerate(groups)
            ]
        fig.legend(
            handles=legend_handles,
            loc="lower center",
            ncol=len(legend_handles),
            frameon=False,
            fontsize=label_size,
        )
        bottom_padding = 0.35 if tilt_x_labels else 0.10
        plt.subplots_adjust(bottom=bottom_padding)
    if save_path:
        plt.savefig(save_path)
        print(f"Saved {plot_type}plot figure to {save_path}")
    plt.close(fig)


def plot_correlation_multi(
    dfs,
    x,
    y,
    df_label_mapping=None,
    colors=None,
    title=None,
    x_label=None,
    y_label=None,
    save_path=None,
    combine=False,
    alpha=0.7,
    legend_loc="best",
    figsize=None,
    title_size=18,
    label_size=14,
    statistic="r",  # 'r' or 'r2'
    test="pearson",  # 'pearson' or 'spearman'
    line_alpha=0.4,
    line_color="gray",
    format_yticks=None,
    filter_outliers=False,  # <-- Add this
    outlier_k=3.0,  # <-- And this
):
    """Plots correlation (scatter) plots for a dictionary of DataFrames, with regression line and correlation stats.

    Args:
        dfs (dict): Dictionary of DataFrames to plot.
        x (str or list): Column name(s) for x-axis.
        y (str): Column name for y-axis.
        df_label_mapping (dict): Mapping of DataFrame keys to plot labels.
        colors (dict, optional): Mapping of DataFrame keys to colors.
        title (str, optional): Plot title.
        x_label (str or list, optional): X-axis label(s).
        y_label (str, optional): Y-axis label.
        save_path (str, optional): Where to save the figure.
        combine (bool): If True, plot all dataframes in one plot; else, one subplot per DataFrame.
        alpha (float): Scatter point transparency.
        legend_loc (str): Legend location.
        figsize (tuple, optional): Figure size.
        title_size (int, optional): Font size for the title.
        label_size (int, optional): Font size for the labels.
        statistic (str): 'r' for correlation coefficient, 'r2' for R squared.
        test (str): 'pearson' or 'spearman'.
        line_alpha (float): Alpha for regression line.
        line_color (str): Color for regression line.
        format_yticks (callable, optional): Function to format y-tick labels.
    """

    def corr_and_line(xvals, yvals, test):
        # Remove NaN/Inf and degenerate data
        mask = np.isfinite(xvals) & np.isfinite(yvals)
        xvals = xvals[mask]
        yvals = yvals[mask]
        if len(xvals) < 2 or np.all(xvals == xvals[0]):
            return np.nan, np.nan, np.nan, np.nan
        if test == "pearson":
            r, p = pearsonr(xvals, yvals)
        elif test == "spearman":
            r, p = spearmanr(xvals, yvals)
        else:
            raise ValueError("test must be 'pearson' or 'spearman'")
        # Fit regression line
        slope, intercept = np.polyfit(xvals, yvals, 1)
        return r, p, slope, intercept

    # Handle x as list or str
    if isinstance(x, list):
        if not combine:
            raise ValueError("If x is a list, combine must be True.")
        xs = x
    else:
        xs = [x]

    def stat_str(r, p, statistic):
        if statistic == "r2":
            stat_val = r**2
            stat_label = r"$R^2$"
        else:
            stat_val = r
            stat_label = r"$R$"
        stars, p_str = significance_stars_and_exact_p(p)
        return f"{stat_label}={stat_val:.2f}, {stars} [{p_str}]"

    # --- Support single DataFrame input ---
    if isinstance(dfs, pd.DataFrame):
        if df_label_mapping is None:
            df_label_mapping = {"data": "Data"}
        dfs = {"data": dfs}
    if df_label_mapping is None:
        raise ValueError("df_label_mapping must be provided if dfs is a dict.")

    n_xs = len(xs)
    n_sets = len(df_label_mapping)
    if colors is None:
        palette = sns.color_palette("Set2", n_sets)
        colors = {k: palette[i] for i, k in enumerate(df_label_mapping.keys())}

    if combine:
        if n_xs == 1:
            if figsize is None:
                figsize = (6, 5)
            fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
            axes = [ax]
        else:
            if figsize is None:
                figsize = (6 * n_xs, 5)
            fig, axes = plt.subplots(1, n_xs, figsize=figsize, constrained_layout=True, sharey=True)
            if n_xs == 1:
                axes = [axes]
        for idx, x_col in enumerate(xs):
            ax = axes[idx]
            for df_key, label in df_label_mapping.items():
                data = dfs[df_key].reset_index()
                xvals = data[x_col].values
                yvals = data[y].values
                if filter_outliers:
                    xvals, yvals = filter_xy_outliers(xvals, yvals, outlier_k)
                ax.scatter(
                    xvals,
                    yvals,
                    label=label,
                    color=colors[df_key],
                    alpha=alpha,
                    edgecolor="k",
                    s=50,
                )
                # Regression line and stats
                if len(xvals) > 1:
                    r, p, slope, intercept = corr_and_line(xvals, yvals, test)
                    xlim = ax.get_xlim()
                    xfit = np.linspace(xlim[0], xlim[1], 100)
                    yfit = slope * xfit + intercept
                    ax.plot(xfit, yfit, color=line_color, alpha=line_alpha, zorder=2)
                    # Annotate
                    ax.text(
                        0.98,
                        0.02 + 0.08 * list(df_label_mapping.keys()).index(df_key),
                        stat_str(r, p, statistic),
                        transform=ax.transAxes,
                        fontsize=label_size - 2,
                        color=colors[df_key],
                        ha="right",
                        va="bottom",
                        bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=1),
                    )
            if isinstance(x_label, list):
                ax.set_xlabel(x_label[idx], fontsize=label_size)
            else:
                ax.set_xlabel(x_col if x_label is None else x_label, fontsize=label_size)
            if format_yticks is not None:
                ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: format_yticks(y)))
            if title and n_xs == 1:
                ax.set_title(title, fontsize=title_size)
            if idx == 0:
                ax.tick_params(
                    axis="y", which="both", left=True, labelleft=True, labelsize=label_size
                )
                if y_label:
                    ax.set_ylabel(y_label, fontsize=label_size)
                else:
                    ax.set_ylabel(y, fontsize=label_size)
            else:
                ax.tick_params(axis="y", which="both", left=False, labelleft=False)
                ax.set_ylabel("")
            ax.grid(True, linestyle="--", alpha=0.5)
        axes[0].set_ylabel(y_label if y_label else y, fontsize=label_size)
        axes[-1].legend(loc=legend_loc, fontsize=label_size, frameon=False)
        if title and n_xs > 1:
            fig.suptitle(title, fontsize=title_size)
    else:
        if isinstance(x, list):
            if len(x) > 1:
                raise ValueError("If x is a list of more than 1 target, combine must be True.")
            x = x[0]
        if figsize is None:
            figsize = (6 * n_sets, 5)
        fig, axes = plt.subplots(1, n_sets, figsize=figsize, sharey=True, constrained_layout=True)
        if n_sets == 1:
            axes = [axes]
        for idx, (ax, (df_key, label)) in enumerate(
            zip(axes, df_label_mapping.items(), strict=False)
        ):
            data = dfs[df_key].reset_index()
            xvals = data[x].values
            yvals = data[y].values
            ax.scatter(xvals, yvals, color=colors[df_key], alpha=alpha, edgecolor="k", s=50)
            # Regression line and stats
            if len(xvals) > 1:
                r, p, slope, intercept = corr_and_line(xvals, yvals, test)
                xlim = ax.get_xlim()
                xfit = np.linspace(xlim[0], xlim[1], 100)
                yfit = slope * xfit + intercept
                ax.plot(xfit, yfit, color=line_color, alpha=line_alpha, zorder=2)
                ax.text(
                    0.98,
                    0.02,
                    stat_str(r, p, statistic),
                    transform=ax.transAxes,
                    fontsize=label_size - 2,
                    color=colors[df_key],
                    ha="right",
                    va="bottom",
                    bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=1),
                )
            # Only set subplot title if there are multiple subplots
            if n_sets > 1:
                ax.set_title(label, fontsize=label_size)
            else:
                ax.set_title("")
            ax.set_xlabel(x_label if x_label else x, fontsize=label_size)
            if format_yticks is not None:
                ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: format_yticks(y)))
        # Only set y-label and y-ticks for the first (left-most) subplot
        for idx, ax in enumerate(axes):
            if idx == 0:
                ax.tick_params(
                    axis="y", which="both", left=True, labelleft=True, labelsize=label_size
                )
                if y_label:
                    ax.set_ylabel(y_label, fontsize=label_size)
                else:
                    ax.set_ylabel(y, fontsize=label_size)
            else:
                ax.tick_params(axis="y", which="both", left=False, labelleft=False)
                ax.set_ylabel("")
            ax.grid(True, linestyle="--", alpha=0.5)
        if title:
            fig.suptitle(title, fontsize=title_size)

    if save_path:
        plt.savefig(save_path)
        print(f"Saved correlation plot to {save_path}")
    plt.close(fig)


def auto_statistic(group1, group2, alpha=0.05):
    # Test normality for both groups
    p1 = shapiro(group1)[1] if len(group1) >= 3 else 0  # shapiro needs at least 3 samples
    p2 = shapiro(group2)[1] if len(group2) >= 3 else 0
    if p1 > alpha and p2 > alpha:
        return "t-test"
    return "mannwhitney"


def significance_stars_and_exact_p(p):
    if p < 0.0001:
        stars = "****"
    elif p < 0.001:
        stars = "***"
    elif p < 0.01:
        stars = "**"
    elif p < 0.05:
        stars = "*"
    else:
        stars = ""
    return stars, f"p={p:.1e}"


def filter_xy_outliers(xvals, yvals, k=3.0):
    # Remove outliers based on IQR for both x and y
    mask = np.ones(len(xvals), dtype=bool)
    for arr in [xvals, yvals]:
        q1 = np.percentile(arr, 25)
        q3 = np.percentile(arr, 75)
        iqr = q3 - q1
        lower = q1 - k * iqr
        upper = q3 + k * iqr
        mask &= (arr >= lower) & (arr <= upper)
    return xvals[mask], yvals[mask]


# Script
metrics_csvs = [
    "test_metrics.csv",
    "shift_metrics.csv",
    "ood_metrics.csv",
]
optimizers = ["adam", "sgd"]
losses = ["ce", "repulsive"]

sources = [
    "results/mlp/ce_adam_lr=0.001",
    "results/mlp/repulsive_adam_lr=0.001",
]

dfs = {
    "test": load_and_combine_metrics(sources, "test_metrics.csv"),
    "shift": load_and_combine_metrics(sources, "shift_metrics.csv"),
    "ood": load_and_combine_metrics(sources, "ood_metrics.csv"),
}

# filter out runs that did not converge properly
for key in dfs:
    dfs[key] = dfs[key][
        ~(
            (
                (dfs[key].index.get_level_values("model") == "batched-rank-7")
                & (dfs[key].index.get_level_values("seed") == "69")
            )
            | (
                (dfs[key].index.get_level_values("model") == "batched-rank-6")
                & (dfs[key].index.get_level_values("seed") == "456")
            )
        )
    ]


# plot_metric_boxplots(
#     df['ood'],
#     metric='ECE',
#     groupby='loss',
#     title="OOD Dataset (FashionMNIST)",
#     x_label='ECE',
#     y_labels=['CE', 'Repulsive CE'],
#     save_path='results/mlp/ood_ece_by_loss_adam_lr=0.001.png',
#     format_xtick=lambda x: f"{x*100:.1f}%"
# )

# filter dfs for repulsive loss
# dfs = {k: v[v.index.get_level_values('loss') == 'ce'] for k, v in dfs.items()}

plot_boxplots_multi(
    dfs,
    df_label_mapping={
        "test": "In-Distribution (MNIST)",
        "shift": "Shifted (MNISTc)",
        "ood": "OOD (FashionMNIST)",
    },
    target="Ens Entr",
    target_label="Mean Entropy",
    save_path="results/mlp/swarm_all_mean_entr_gb_by_loss_adam_lr=0.001.png",
    sharey=False,
    # format_ticks=lambda y: f"{y*100:.1f}%",
    groupby="model",
    group_label_mapping={
        # 'standard': 'Single',
        "batched-rank-1": "Rank-1",
        "gb-batched-rank-1": "Rank-1 (GB)",
        # 'batched-rank-2': 'Rank-2',
        # 'batched-rank-3': 'Rank-3',
        # 'batched-rank-4': 'Rank-4',
        # 'batched-rank-5': 'Rank-5',
        # 'batched-rank-6': 'Rank-6',
        # 'batched-rank-7': 'Rank-7',
        "batched-rank-full": "Full Rank",
        "gb-batched-rank-full": "Full Rank (GB)",
        "ensemble": "Ensemble",
    },
    subgroupby="loss",
    subgroup_label_mapping={
        "ce": "Cross Entropy Loss",
        "repulsive": "Repulsive Cross Entropy Loss",
    },
    # subgroup_palette_mapping={
    #     'ce': 'Blues',
    #     'repulsive': 'Oranges',
    # },
    subgroup_colors={
        "ce": "#1f77b4",  # Blue
        "repulsive": "#ff7f0e",  # Orange
    },
    # use_legend=True,
    statistic="auto",
    # filter_outliers=True,
    plot_type="swarm",
    central_tendency="mean",
)

# # Combine into one DataFrame
# combined_df = pd.concat(dfs.values(), keys=dfs.keys())
# # filter dfs for mi and disagreement columns
# combined_df = combined_df[
#     combined_df['Ens MI'].notna() &
#     combined_df['Ens Entr'].notna()
# ]

# plot_correlation_multi(
#     dfs,
#     x=['Ens MI', 'Ens Entr'],
#     # x='Ens MI',
#     y='NLL',
#     df_label_mapping={
#         'test': 'In-Distribution (MNIST)',
#         'shift': 'Shifted (MNISTc)',
#         # 'ood': 'OOD (FashionMNIST)'
#     },
#     combine=True,  # or False for subplots
#     x_label=['Mutual Information', 'Mean Entropy'],
#     # x_label='Mutual Information',
#     y_label='Negative Log-Likelihood',
#     save_path="results/mlp/correlation_nll_mi_mean_entr.png",
#     statistic='r2',
#     filter_outliers=False,
#     # format_yticks=lambda y: f"{y*100:.1f}%",
# )
