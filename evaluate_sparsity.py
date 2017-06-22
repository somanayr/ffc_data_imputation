#!/usr/bin/python

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def calc_percent(df):
    ps = []
    ncols = len(df.columns)
    for ri in df.index:
        # 'challengeID' is included, so subtract accordingly
        ps.append(((ncols - 1) - (sum(df.loc[ri,:] >= 0) - 1)) / (float(ncols) - 1))

    return ps


def calc_sixes(df):
    ps = []
    ncols = len(df.columns)
    for ri in df.index:
        ps.append(sum(df.loc[ri,:] == -6) / (float(ncols) - 1))
    return ps


def calc_sevens(df):
    ps = []
    ncols = len(df.columns)
    for ri in df.index:
        ps.append(sum(df.loc[ri,:] == -7) / (float(ncols) - 1))
    return ps


def calc_both(df):
    ps = []
    ncols = len(df.columns)
    for ri in df.index:
        ps.append(sum((df.loc[ri,:] == -6) | (df.loc[ri,:] == -7)) / (float(ncols) - 1))
    return ps


def flag_indices(df, cutoff):
    flagged = []

    # Remove `challengeID` from calculation
    ncols = len(df.columns) - 1
    for ri in df.index:
        if (((ncols) - (sum(df.loc[ri,:] >= 0) - 1)) / (float(ncols))) >= cutoff:
            # flagged.append(df.loc[ri,'challengeID'])
            flagged.append(ri)
        else:
            print cutoff, ncols, sum(df.loc[ri,:] >= 0) - 1, (((ncols) - (sum(df.loc[ri,:] >= 0) - 1)) / (float(ncols)))

    return flagged


# def flag_indices_for_all_labels(df, cutoff):
#     lcdict = {}
#     lcounts = {}

#     ag = pd.read_csv("ffc_data/imputed/aggregate_year5_unrestricted_nostrings_noids_with-challengeid.csv", low_memory=False)
#     # bg = pd.read_csv("ffc_data/original/background.csv", low_memory=False)
#     # bg = df
#     data = df
#     labels = pd.read_csv("ffc_data/original/train.csv", low_memory=False)

#     expanded_names = ag.columns.tolist()

#     # qnames = ['challengeID']

#     # for n in expanded_names:
#     #     test = '_'.join(n.split('_')[:-1])
#     #     if test in bg.columns and not test in qnames:
#     #         qnames.append(test)
#     #     elif n in bg.columns and not n in qnames:
#     #         qnames.append(test)

#     # data = bg.loc[:,qnames]

#     labnames = ["gpa","grit","materialHardship","eviction","layoff","jobTraining"]

#     lrows = {}

#     labeled_cids = {}

#     # percents = {}
#     # flagged_indices = {}

#     # for lname in labnames:
#         # print lname
#         # labeled_cids[lname] = list(labels[~np.isnan(labels[lname])]['challengeID'])
#         # lrows[lname] = data[data['challengeID'].isin(labeled_cids[lname])]
#         # percents[lname] = calc_percent(lrows[lname])
#         # flagged_indices[lname] = flag_indices(lrows[lname], cutoff)
#         # sps = sorted(list(set(percents[lname])))
#         # pc = 0

#         # lcdict[lname] = {}
#         # for p in percents[lname]:
#             # if not p in lcdict[lname]:
#                 # lcdict[lname][p] = 0
#             # lcdict[lname][p] += 1

#         # lcounts[lname] = []
#         # for p in sps:
#             # pc += lcdict[lname][p]
#             # lcounts[lname].append(pc)

#     flagged_indices = flag_indices(df, cutoff)

#     return flagged_indices


def main():
    lcdict = {}
    lcounts = {}

    ag = pd.read_csv("ffc_data/imputed/aggregate_year5_unrestricted_nostrings_noids_with-challengeid.csv", low_memory=False)
    bg = pd.read_csv("ffc_data/original/background.csv", low_memory=False)
    labels = pd.read_csv("ffc_data/original/train.csv", low_memory=False)

    expanded_names = ag.columns.tolist()

    qnames = ['challengeID']

    for n in expanded_names:
        test = '_'.join(n.split('_')[:-1])
        if test in bg.columns and not test in qnames:
            qnames.append(test)
        elif n in bg.columns and not n in qnames:
            qnames.append(test)

    # print qnames
    # print len(qnames)

    data = bg.loc[:,qnames]

    labnames = ["gpa","grit","materialHardship","eviction","layoff","jobTraining"]

    lrows = {}

    labeled_cids = {}

    percents = {}

    xlab = "Cutoff percentage of invalid/missing entries"
    ylab = "Number of rows with invalid/missing entry portions <= cutoff"
    tit = "Rows remaining after pruning by sparsity - "

    for lname in labnames:
        print lname
        labeled_cids[lname] = list(labels[~np.isnan(labels[lname])]['challengeID'])
        lrows[lname] = data[data['challengeID'].isin(labeled_cids[lname])]
        percents[lname] = calc_percent(lrows[lname])
        sps = sorted(list(set(percents[lname])))
        pc = 0

        lcdict[lname] = {}
        for p in percents[lname]:
            if not p in lcdict[lname]:
                lcdict[lname][p] = 0
            lcdict[lname][p] += 1

        lcounts[lname] = []
        for p in sps:
            pc += lcdict[lname][p]
            lcounts[lname].append(pc)

        plt.clf()
        plt.plot(sps, lcounts[lname])
        plt.xlabel(xlab)
        plt.ylabel(ylab)
        plt.title(tit + lname)
        plt.tight_layout()
        plt.savefig("images/rows_remaining_after_sparsity_pruning_"+lname+".pdf")
#          sps = sorted(list(set(percents[lname])), reverse=True)
# ...     plt.plot(sps, lcounts[lname])
# ...     plt.xlabel(xlab)
# ...     plt.ylabel(ylab)
# ...     plt.title(tit + lname)
# ...     plt.savefig("images/missing_percentages_year5_"+lname+".pdf")
# ...     plt.clf()



if __name__ == "__main__":
    main()