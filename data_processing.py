import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from copy import deepcopy
from multiprocessing import Pool
from functools import partial
# from bootstrap_analysis import bootstrap_analysis_single_sol
from matplotlib import cm, colors
from sklearn.ensemble import GradientBoostingClassifier


def backcalc_objs_sample(objs):
    # Back-calculate objectives described by mean (all but wcc)
    backcalc_objs = objs.mean(axis=2)

    # Back-calculate objectives described by worse first percentile (wcc only)
    backcalc_objs[:, :, 4] = np.percentile(objs[:, :, :, 4], 99, axis=2)

    return backcalc_objs


def load_objectives(obj_data_dir, rdm_data_dir, n_rdms, n_sols, n_objs, n_utilities,
                    reload_csvs=False):
    try:
        if not reload_csvs:
            data = np.load('objs_by_sol.npy')
            complete_rdm = np.loadtxt('complete.csv')
            print('Loaded {} rdms from {} to {}.'.format(len(complete_rdm),
                                                         np.min(complete_rdm),
                                                         np.max(complete_rdm)))
        else:
            complete_rdm, data = load_raw_data_and_completeness_check(
                obj_data_dir, n_rdms, n_sols)
    except IOError:
        print('Reloading all csv data.')
        complete_rdm, data = load_raw_data_and_completeness_check(
            obj_data_dir,
            n_rdms, n_sols
        )

    data = data.reshape(n_sols, n_rdms, n_utilities, n_objs)
    data = np.swapaxes(data, 0, 2)
    data = np.swapaxes(data, 1, 2)
    rdm_dmp, rdm_utilities, rdm_inflows, rdm_water_sources = \
        load_rdm_factors(rdm_data_dir)

    return complete_rdm, data, rdm_dmp, rdm_utilities, \
        rdm_inflows, rdm_water_sources


def load_rdm_factors(rdm_dir):
    rdm_dmp = np.loadtxt('{}rdm_dmp_test_problem_reeval.csv'.format(rdm_dir),
                         delimiter=',')
    rdm_utilities = np.loadtxt('{}rdm_utilities_test_problem_reeval.csv'
                               .format(rdm_dir), delimiter=',')
    rdm_inflows = np.loadtxt('{}rdm_inflows_test_problem_reeval.csv'
                             .format(rdm_dir), delimiter=',')
    rdm_water_sources = np.loadtxt('{}rdm_water_sources_test_problem_reeval.csv'
                                   .format(rdm_dir), delimiter=',')

    return rdm_dmp, rdm_utilities, rdm_inflows, rdm_water_sources


def load_raw_data_and_completeness_check(rdm_dir, n_rdms, n_sols):

    objectives = np.loadtxt('{}Objectives_RDM0_sols0_to_{}.csv'.
                            format(rdm_dir, n_sols), delimiter=',')
    n_objs = objectives.shape[1]

    incomplete_rdm = []
    not_run_or_crashed_rdm = []
    complete_rdms = []
    data = []
    for rdm in range(n_rdms):
        try:
            objectives = np.loadtxt('{}Objectives_RDM{}_sols0_to_229.csv'.
                                    format(rdm_dir, rdm), delimiter=',')
        except Exception:
            not_run_or_crashed_rdm.append(rdm)

        if len(objectives) < n_sols:
            incomplete_rdm.append(rdm)
        else:
            complete_rdms.append(rdm)
            data.append(objectives)

    np.savetxt('incomplete.csv', incomplete_rdm, fmt='%d')
    np.savetxt('complete.csv', complete_rdms, fmt='%d')
    np.savetxt('not_run_or_crashed.csv', not_run_or_crashed_rdm, fmt='%d')

    data_transposed = np.array(data).transpose((1, 0, 2))
    np.save('objs_by_sol.npy', data_transposed)

    return complete_rdms, data_transposed


def check_rdm_meet_criteria(objectives, crit_objs, crit_vals):
    # check max and min criteria for each objective
    meet_low = objectives[:, crit_objs] > crit_vals[0]
    meet_high = objectives[:, crit_objs] < crit_vals[1]

    # check if max and min criteria are met at the same time
    robustness_utility_solution = np.hstack((meet_low, meet_high)).all(axis=1)

    return robustness_utility_solution



def process_decvars_inverse(dec_vars, utilities_names, dec_vars_names_columns,
                            utility_transfer_source_name):
    data = {}
    for dv, col in dec_vars_names_columns.iteritems():
        data_dv = {}

        for name, u in zip(utilities_names, range(len(utilities_names))):
            if not (dv == 'Transfer Trigger' and
                    name == utility_transfer_source_name):
                data_dv[name] = dec_vars[col + u]

        # If lake allocation variable, normalize data if needed
        if 'Lake Allocation' in dv:
            total_alloc = sum([data_dv[name] for name in utilities_names])
            if total_alloc > 1.:
                for name in utilities_names:
                    data_dv[name] /= total_alloc

        data[dv] = data_dv

    return data


def load_dec_vars(data_dir, sols, sol_names):
    dec_vars = np.loadtxt(data_dir + 'reference_final_filtered.reference')

    # normalize Lake Michael\nallocation
    sum_alloc = dec_vars[:, 5:8].sum(axis=1)
    sum_alloc[sum_alloc < 1.] = 1.

    dec_vars[:, 5:8] /= np.tile(sum_alloc, (3, 1)).T

    # decision vars max and min values as in the cpp input file
    dec_vars_max_min = {
        'Restriction Trigger': [0, 1],
        'Transfer Trigger': [0, 1],
        'Insurance Trigger': [0, 1],
        'Annual Contingency\nFund Contribution': [0, 0.1],
        'Infrastructure\n(Long-Term) Trigger': [0, 1],
        'Lake Michael\nAllocation': [0, 1.],
        'Insurance Payment': [0, 0.03]
    }

    order_of_plotting = [
        'Restriction Trigger',
        'Transfer Trigger',
        'Insurance Trigger',
        'Annual Contingency\nFund Contribution',
        'Infrastructure\n(Long-Term) Trigger',
        'Lake Michael\nAllocation',
        'Insurance Payment'
    ]

    # reorganize decision variables
    dvs_sols = {}
    for s, n in zip(sols, sol_names):
        dv_dict = {'Watertown': {}, 'Dryville': {},
                   'Fallsland': {}, 'Infrastructure Order': {}}
        dvs = dec_vars[s]

        dv_dict['Watertown'] = {
            'Restriction Trigger': dvs[0],
            'Lake Michael\nAllocation': dvs[5],
            'Annual Contingency\nFund Contribution': dvs[8],
            'Insurance Trigger': dvs[11],
            'Insurance Payment': dvs[14],
            'Infrastructure\n(Long-Term) Trigger': dvs[17]
        }
        dv_dict['Dryville'] = {
            'Restriction Trigger': dvs[1],
            'Transfer Trigger': dvs[3],
            'Lake Michael\nAllocation': dvs[6],
            'Annual Contingency\nFund Contribution': dvs[9],
            'Insurance Trigger': dvs[12],
            'Insurance Payment': dvs[15],
            'Infrastructure\n(Long-Term) Trigger': dvs[18]
        }
        dv_dict['Fallsland'] = {
            'Restriction Trigger': dvs[2],
            'Transfer Trigger': dvs[4],
            'Lake Michael\nAllocation': dvs[7],
            'Annual Contingency\nFund Contribution': dvs[10],
            'Insurance Trigger': dvs[13],
            'Insurance Payment': dvs[16],
            'Infrastructure\n(Long-Term) Trigger': dvs[19]
        }
        dv_dict['Infrastructure Order'] = {
            'Watertown': {
                'New River': dvs[20],
                'College Rock expansion low': dvs[21],
                'College Rock expansion high': dvs[22],
                'Watertown reuse I': dvs[23],
                'Watertown reuse II': dvs[27]
            },
            'Dryville': {
                'Sugar creek': dvs[24],
                'Dryville reuse': dvs[28]
            },
            'Fallsland': {
                'New River': dvs[26],
                'Fallsland reuse': dvs[29]
            }
        }

        dvs_sols[n] = dv_dict

    return dec_vars_max_min, dvs_sols, order_of_plotting
