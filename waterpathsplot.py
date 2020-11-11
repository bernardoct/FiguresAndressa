import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, colors, colorbar
from copy import deepcopy
import os
import pandas as pd
from plot_dec_vars import plot_decision_vars

def get_pathways_by_utility_realization(pathways_sol):
    # Reformat utility data
    pathways_list_utility = []
    for u in range(int(max(pathways_sol[:, 1])) + 1):
        pathways_list = []
        for r in range(int(max(pathways_sol[:, 0])) + 1):
            ur = (pathways_sol[:, [0, 1]] == np.array([r, u])).all(axis=1)
            if np.sum(ur) > 0:
                pathways_list.append(pathways_sol[ur][:, [0, 2, 3]].T)
        pathways_list_utility.append(pathways_list)

    return pathways_list_utility


def plot_colormap_pathways(pathways_utility, nweeks, source_colormap_id,
                           solution, rdm, n_existing_sources, ax, ax_cb, cb_pos,
                           savefig_directory='', nrealizations=1000, sort_by=(),
                           sources=(), utility_name='', year0=0, suffix='', 
                           cmap_name='tab20c'):
    tick_font_size = 10
    axis_font_size = 12

    x, y, pathways = get_mesh_pathways(pathways_utility, nweeks,
                                       n_existing_sources, len(sources),
                                       nrealizations=nrealizations)

    cmap = cm.get_cmap(cmap_name)
    cmap_mod = colors.LinearSegmentedColormap.\
        from_list('Custom cmap',
                  [cmap(i) for i in source_colormap_id[:, 1]],
                  len(source_colormap_id[:, 1]))

    pathways_cp = deepcopy(pathways)
    for i in range(len(source_colormap_id)):
        pathways[pathways_cp == source_colormap_id[i, 0]] = i

    if len(sort_by) == 0:
        pathways = np.array(sorted(pathways, key=lambda x: sum(x)))
    else:
        pathways = np.array(pathways)[sort_by]

    body_font = 'Open Sans Condensed Light'

    ax.imshow(pathways, origin='lower', cmap=cmap_mod, aspect=float(nweeks) / nrealizations,
              vmin=0, vmax=cmap_mod.N)
    ax.set_xlabel('Year', **{'fontname': body_font, 'size': axis_font_size})
    # ax.set_ylabel('Realization',
    #               **{'fontname': body_font, 'size': axis_font_size})

    ax.grid(False)
    ax.set_yticks([0, nrealizations])
    ax.set_yticklabels(['Significant and early\nnew infrastructure',
                        'Little and late\nnew infrastructure'],
                       {'fontname': body_font, 'size': tick_font_size})
    xticks_at = np.arange(0, nweeks, 52.1 * 15)
    ax.set_xticks(xticks_at)
    ax.set_xticklabels((xticks_at / 52.1).astype(int) + year0,
                       {'fontname': body_font, 'size': tick_font_size})

    pos = ax.get_position()
    new_pos = [pos.x0, 0.1, 0.7 - pos.x0, 0.9]
    # ax.set_position(new_pos)
    # ax_cb = fig.add_axes([0.75, 0.1, 0.03, 0.8])

    bounds = np.arange(len(source_colormap_id) + 1)
    norm = colors.BoundaryNorm(bounds, cmap_mod.N)

    cb = colorbar.ColorbarBase(ax_cb, cmap=cmap_mod, norm=norm,
                               orientation='horizontal',
                               spacing='proportional', ticks=bounds,
                               boundaries=bounds)

    ax_cb.set_position(cb_pos)
    cb.set_ticks(np.arange(len(source_colormap_id)) + 0.5)
    cb.set_ticklabels(sources[source_colormap_id[:, 0]])
    # for l in cb.ax.yaxis.get_ticklabels():
    #     l.set_family(body_font)
    #     l.set_size(tick_font_size)
    for l in cb.ax.xaxis.get_ticklabels():
        l.set_rotation(35)
        l.set_horizontalalignment('right')
        l.set_verticalalignment('top')
        l.set_family(body_font)
        l.set_fontsize(tick_font_size)
    cb.ax.set_xlabel('Infrastructure option',
                     **{'fontname': body_font, 'size': axis_font_size})


def get_mesh_pathways(pathways_utility, nweeks, n_existing_sources, n_sources,
                      nrealizations=-1):
    if nrealizations == -1:
        nrealizations = len(pathways_utility)

    x, y = np.meshgrid(range(nrealizations), range(nweeks))

    z = np.ones((nrealizations, nweeks)) * \
        (n_sources - 1)  # - n_existing_sources
    for p in pathways_utility:
        r = p[0][0]
        z[r] = create_fixed_length_pathways_array(
            p[1], p[2], nweeks, n_sources)  # - \
        # n_existing_sources

    return x, y, z


def create_fixed_length_pathways_array(weeks_vector,
                                       infra_option_or_npv, length, n_sources):
    fixed_length_array = np.ones(length) * (n_sources - 1)

    for i in range(1, len(weeks_vector)):
        fixed_length_array[weeks_vector[i - 1]: weeks_vector[i]] = \
            infra_option_or_npv[i - 1]

    fixed_length_array[weeks_vector[-1]: -1] = infra_option_or_npv[-1]

    return fixed_length_array


if __name__ == "__main__":
    pathways = []
    # solutions = [21, 34, 35, 44, 199, 240, 249, 288]
    solutions = [150]
    # ### A LINHA ABAIXO GERA O ARRAY ['source 1', 'source 2', ..., 'source 30']. SUBSTITUA A LINHA ABAIXO POR UM ARRAY COM OS NOMES DAS INFRAESTRUTURAS. ###
    sources = np.array(['source {}'.format(s) for s in range(30)])
    utils_ids = [0, 1]
    # files_root_directory = r'C:\Users\Bernardo\Downloads\andressa\completo\\'
    files_root_directory = r'C:\Users\Bernardo\Downloads\andressa\\'
    for s in solutions:
        pathways_all_utilities = np.loadtxt(
            # files_root_directory + 'Pathways_s{}_original.out'.format(s), # ARQUIVO DE PATHWAYS
            files_root_directory + 'Pathways_s{}.out'.format(s), # ARQUIVO DE PATHWAYS
            delimiter='\t',
            comments='R',
            dtype=int)
        for u in utils_ids:
            pu = pathways_all_utilities[:, 1] == u
            pathways.append([pathways_all_utilities[pu]])

    ### A MATRIX ABAIXO ATRIBUI UMA COR DO COLORMAP TAB20C (https://matplotlib.org/3.1.0/tutorials/colors/colormaps.html) A CADA PROJETO DE INFRAESTRUTURA--O PADRÃO É [[SOURCE, COR], [SOURCE, COR], ...]. ###
    ### PARA MUDAR AS CORES DA FIGURA É SÓ MUDAR O cmap_name NA ANTEPENÚLTIMA LINHA E A MATRIX ABAIXO ###
    source_colormap_id = np.array([[4, 4],
                                   [5, 1],
                                   [7, 12],
                                   [8, 13],
                                   [9, 11],
                                   [10, 10],
                                   [11, 9],
                                   [12, 8],
                                   [13, 19]])

    cb_pos = [0.2, 0.15, 0.7, 0.01]

    fig, ax = plt.subplots(2, 2, figsize=(6, 6))
    for u in utils_ids:
        pathways_list_utility_high = \
            get_pathways_by_utility_realization(pathways[u][0])

        utility_pathways_high_copy = deepcopy(pathways_list_utility_high)

        plot_colormap_pathways(utility_pathways_high_copy[u], 2500, source_colormap_id, solutions[0], solutions[0], 5, ax[0, u], ax[1, 1], cb_pos, sources=sources, cmap_name='tab20c')

    # # COMENTE A LINHA ABAIXO E DESCOMENTE A SEGUINTE PARA GERAR UM SVG QUE PODE SER EDITADO NO INKSCAPE.
    plt.show()
    # # plt.savefig('pathways.svg')

    # dec_vars_torto = pd.read_csv('completo/decvars_torto.txt', sep=',')
    # dec_vars_descoberto = pd.read_csv('completo/decvars_descoberto.txt', sep=',')
    # dec_vars_max_min = {
    #     'Nome da 1a variável': [0., 1.],  # nome como no arquivo de dados, valor max e min
    #     'Nome da 2a variável': [5., 100.] 
    #     }
    # utilities_names = ['Descoberto', 'Torto']

    # solutions_to_highlight = {}
    # solutions_to_highlight['colors'] = ['blue', 'red', 'green']
    # solutions_to_highlight['labels'] = ['essa política', 'aquela política', 'outra política']

    # order_of_plotting_dvs = list(range(20))
    # plot_decision_vars([dec_vars_torto, dec_vars_descoberto], dec_vars_max_min, files_root_directory,
    #                    utilities_names, order_of_plotting_dvs,
    #                    solutions_to_highlight,
    #                    grid_dims=(3, 3), size=(8.5, 5.5))