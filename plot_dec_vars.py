import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.lines import Line2D
from data_processing import process_decvars_inverse


def plot_decision_vars(dec_vars, dec_vars_max_min, files_root_directory,
                       utilities_names, order_of_plotting_dvs,
                       solutions_to_highlight,
                       grid_dims=(3, 3), size=(8.5, 5.5)):
    n_dec_vars = 0
    for un in utilities_names:
        n_dec_vars_util = len(list(dec_vars.values())[0][un])
        if n_dec_vars_util > n_dec_vars:
            n_dec_vars = n_dec_vars_util

    with plt.style.context('classic'):
        fig, axes = plt.subplots(grid_dims[0], grid_dims[1],
                                 figsize=(7, 8.5))  # , sharey=True)
        plt.subplots_adjust(bottom=0.2, right=0.95, hspace=0.45,
                            wspace=0.4, left=0.06)

        for sol_name, c in zip(list(dec_vars.keys()),
                               solutions_to_highlight['colors']):
            plot_dec_vars_paxis(dec_vars[sol_name],
                                dec_vars_max_min, axes, c,
                                sol_name, utilities_names,
                                order_of_plotting_dvs, n_dec_vars)

        # Add white zone to legend
        add_legend(solutions_to_highlight, axes, grid_dims)

        plt.suptitle(
            'Policy Variables Values',
            **{'fontname': 'Gill Sans MT', 'size': 14}
        )

        add_table_infra_rank(dec_vars, utilities_names, axes)

        plt.savefig(files_root_directory + 'dec_vars.svg')


def add_table_infra_rank(dec_vars, utilities_names, axes):
    infra_names = []
    sol_0 = list(dec_vars.keys())[0]
    for u in utilities_names:
        infra_names += list(dec_vars[sol_0]
                            ['Infrastructure Order'][u].keys())
    infra_names = np.array(infra_names)
    infra_names = np.unique(infra_names)

    cell_text = calculate_infra_rankings(infra_names,
                                         utilities_names, dec_vars)

    table = axes[-1, -1].table(
        rowLabels=infra_names,
        colLabels=utilities_names,
        cellText=cell_text,
        loc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    for a in table.get_children():
        a.set_text_props(**{'fontfamily': 'Open Sans Condensed Light'})

    axes[-1, -1].text(0, 1.06, 'Numbers in each cell represent ranking\nfor each solution in legend order.',
                      **{'family': 'Open Sans Condensed Light',
                         'size': 10}
                      )


def calculate_infra_rankings(rows, cols, dec_vars_infra):
    ranking_table = [['-' for i in range(len(cols))] for j in range(len(rows))]

    for sol in dec_vars_infra.keys():
        dvs = dec_vars_infra[sol]['Infrastructure Order']
        for c in range(len(cols)):
            ranks = np.array(list(dvs[cols[c]].values()))
            names = np.array(list(dvs[cols[c]].keys()))

            order = np.argsort(ranks)
            name_sorted = names[order]
            for v, dv in zip(np.arange(len(names)) + 1, name_sorted):
                pos_infra = int(np.where(rows == dv)[0][0])
                ranking_table[pos_infra][c] += (', ' + str(v))

    for r in range(len(rows)):
        for c in range(len(cols)):
            if len(ranking_table[r][c]) > 1:
                ranking_table[r][c] = ranking_table[r][c][2:]

    return ranking_table


def add_legend(solutions_to_highlight, axes, grid_dims):
    handles = [Line2D((0, 0), (0, 1), color=c)
               for c in solutions_to_highlight['colors']]
    labels = solutions_to_highlight['labels_long']

    axes[-1, int(grid_dims[1] / 2)].legend(
        handles,
        labels,
        loc='upper center',
        bbox_to_anchor=(0.5, -0.2),
        ncol=1,
        prop={'family': 'Open Sans Condensed Light',
              'size': 12}
    )


def clear_axis(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)


def plot_dec_vars_paxis(dec_vars, max_mins, axes, c, label, utilities,
                        decvars_order, n_dec_vars):
    axes_ravel = axes.ravel()
    for dv_name, ax in zip(decvars_order, axes_ravel):
        utilities_to_plot = []
        dv_data = []
        for u_name in utilities:
            if dv_name in dec_vars[u_name]:
                utilities_to_plot.append(u_name)
                dv_data.append(dec_vars[u_name][dv_name])

        ax.tick_params(axis=u'both', which=u'both', length=0)
        ax.set_ylim(max_mins[dv_name])
        ax.set_yticks(max_mins[dv_name])
        ax.set_yticklabels(
            max_mins[dv_name],
            **{'fontname': 'Open Sans Condensed Light', 'size': 10}
        )
        ax.plot(dv_data, c=c, label=label)
        ax.set_xticks(range(len(utilities_to_plot)))
        ax.set_xticklabels(utilities_to_plot,
                           {'fontname': 'Open Sans Condensed Light', 'size': 11})
        ax.set_title(dv_name, {'fontname': 'Gill Sans MT', 'size': 12})

    for i in range(n_dec_vars, len(axes_ravel)):
        clear_axis(axes_ravel[i])
