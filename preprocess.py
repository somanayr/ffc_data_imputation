#!/usr/bin/python

import re
import sys
import math
import time
# import scipy
import random
import argparse
import scipy.stats
import numpy as np
import pandas as pd
# import impute_codebook
from os import makedirs
from os.path import exists, join

# python preprocess.py -d ffc_data/original/background.csv -t codebook_notes/noid_aggregate_year5.tsv -m imputation_notes/aggregate_year5.tsv -v -o ffc_data/imputed/aggregate_year5_unrestricted_nostrings.csv -sr
# python preprocess.py -d ffc_data/original/background.csv -t codebook_notes/noid_aggregate_year5.tsv -m imputation_notes/aggregate_year5.tsv -v -o ffc_data/imputed/naive_with-challengeid_nostrings.csv -sr -nv


base_dir = "ffc_data"
orig_dir = join(base_dir, "original")
imputed_dir = join(base_dir, "imputed")
trimmed_dir = join(base_dir, "trimmed")
predictions_dir = join(base_dir, "predictions")

###############################################
#                                             #
#                  Main Loop                  #
#                                             #
###############################################

def main():
    maybe_format_data_directory()
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data_filename", help="Path to original data file", required=True)
    parser.add_argument("-t", "--question_type_filename", help="Path to question type curation file (TSV)", required=True)
    parser.add_argument("-m", "--imputation_method_filename", help="Path to imputation method curation file (TSV)", required=True)
    parser.add_argument("-r", "--restricted", help="Restrict to only look at the time waves mentioned in curation files", action="store_true")
    parser.add_argument("-o", "--output_filename", help="Path to output filename", required=True)
    parser.add_argument("-v", "--verbose", help="Verbose logging", action="store_true")
    parser.add_argument("-sr", "--string_removal", help="Remove columns with string values", action="store_true")
    parser.add_argument("-nv", "--naive_imputation", help="Use mode imputation instead", action="store_true")
    options = parser.parse_args()

    run(options)


def run(options):
    if options.verbose:
        log("Loading original data set")
    original_data = pd.read_csv(options.data_filename, low_memory=False)

    if options.verbose:
        log("Parsing question type curation file")
    qtype_file = open(options.question_type_filename)
    qtypes = parse_qtypes(qtype_file)
    remappings = parse_remappings(qtype_file)
    qtype_file.close()

    if options.verbose:
        log("Parsing imputation methods")
    method_file = open(options.imputation_method_filename)
    methods = parse_imputation_methods(method_file, options.restricted)
    method_file.close()

    if options.verbose:
        log("Collecting curated question names")
    # qnames = list(set(qtypes.keys()).intersection(set(methods.keys())))
    qnames = list(set(qtypes.keys()))
    # if not 'challengeID' in qnames:
    #     qnames = ['challengeID'] + qnames

    if options.verbose and options.string_removal:
        log("Removing columns containing non-numerical values")
    if options.string_removal:
        qnames = prune_question_names(qnames, original_data)

    if options.verbose:
        log("Pruning uncurated questions")
    pruned = prune_uncurated(original_data, qnames)
    methods = prune_annotations(methods, qnames)
    remappings = prune_annotations(remappings, qnames)
    qtypes = prune_annotations(qtypes, qnames)

    if options.naive_imputation:
        if options.verbose:
            log("Naively imputing data")
        naive = impute_naively(pruned, verbose=options.verbose)

        if options.verbose:
            log("Saving to file %s" % options.output_filename)
        naive.to_csv(options.output_filename, index=False)

        return naive
    else:
        if options.verbose:
            log("Imputing missing data")
        imputed = impute_data(pruned, methods, verbose=options.verbose)

        if options.verbose:
            log("Remap specified values")
        remapped = fix_remappings(imputed, remappings)

        if options.verbose:
            log("Expanding categorical values")
        expanded = expand_categorical(remapped, qtypes)
        # expanded = expand_categorical(imputed, qtypes)

        if options.verbose:
            log("Saving to file %s" % options.output_filename)
        expanded.to_csv(options.output_filename, index=False)

        return expanded


###############################################
#                                             #
#                 Auxiliary                   #
#                                             #
###############################################


def log(message):
    sys.stdout.write(time.strftime("%H:%M:%S") + " :: " + message + "\n")
    sys.stdout.flush()


def round_(val, col):
    if col.dtype == np.int64:
        return np.int64(np.round_(val))
    elif col.dtype == np.float64:
        return val
    else:
        raise TypeError("Column contains non-numeric values")


def prune_question_names(qnames, data):
    good_names = []

    for qname in qnames:
        col = data[qname]
        if (col.dtype == np.int64 or col.dtype == np.float64):
            good_names.append(qname)

    return good_names


def prune_annotations(annotations, qnames):
    qref = dict(zip(qnames, [True for q in qnames]))

    new_annotations = {}
    for qname in annotations.iterkeys():
        if qname in qref:
            new_annotations[qname] = annotations[qname]

    return new_annotations


# From Ryan Amos
question_re = re.compile(r"^([^\d]+)(\d+)(.*)$")
def last_observed_value(data, ridx, question_id):
    """
    Finds the last valid observation for the given question on the given row
    Assume question names do not change between years
    Returns none if no valid past observations found
    """
    question = question_re.match(question_id)
    if question == None:
        # print question_id
        return None
        
    who = question.group(1)
    when = int(question.group(2))
    what = question.group(3)

    while when > 1:
        when -= 1
        prev_question = "%s%d%s" % (who, when, what)
        try:
            prev_val = data[ridx][prev_question]
            if not math.isnan(prev_val) and (prev_val >= 0 or i != math.floor(prev_val)):
                return prev_val
        except KeyError:
            pass
    return None


def impute_with_last_observations(data, qname, code):
    if code == 'nan':
        qcodemask = (np.isnan(data[qname]))
    else:
        qcodemask = (data[qname] == code)
    qcodeindices = qcodemask.index.values

    for ridx in qcodeindices:
        lov = last_observed_value(data, ridx, qname)
        if lov:
            data.loc[ridx,qname] = lov

    return data


def sample_from_distribution(valid_vals, ranges, num_samples):
    choices = []

    for val in valid_vals:
        for r in ranges:
            if val >= r[0] and val <= r[1]:
                choices.append(val)
                break

    # return random.sample(choices, num_samples)
    samples = []

    for i in xrange(num_samples):
        samples.append(random.sample(choices, 1)[0]) # Enforce replacement

    return samples


# Takes f = open(options.question_type_filename)
def parse_qtypes(infile):
    infile.seek(0)

    qtypes = {}

    for line in infile:
        parts = line.strip().split("\t")
        if len(parts) > 1:
            qtypes[parts[0]] = parts[1]

    return qtypes


# Takes f = open(options.question_type_filename)
def parse_remappings(infile):
    infile.seek(0)

    remappings = {}

    for line in infile:
        parts = line.strip().split("\t")
        if len(parts) > 3:
            remappings[parts[0]] = {}
            for pair in parts[2:]:
                original, target = pair.strip().split(",")
                try:
                    original = int(original)
                    target = int(target)
                except ValueError:
                    original = float(original)
                    target = float(target)
                remappings[parts[0]][original] = target
    return remappings


# Takes f = open(options.imputation_method_filename)
def parse_imputation_methods(infile, restricted):
    infile.seek(0)

    methods = {}

    for line in infile:
        parts = line.strip().split("\t")
        qname = parts[0]
        codemethods = parts[1].split(";")

        methods[qname] = {}

        for codemethod in codemethods:
            # code, method = codemethod.strip().split(" ")
            parts = codemethod.strip().split(" ")
            code = parts[0]
            method = ' '.join(parts[1:])
            try:
                code = int(code)
            except ValueError:
                code = float(code)
            if method.upper().startswith("R"):
                if restricted:
                    method = method[1:]
            methods[qname][code] = method

    return methods


# `data` is a pandas dataframe
def prune_uncurated(data, curated_question_list):
    if not 'challengeID' in curated_question_list:
        newlist = ['challengeID'] + curated_question_list
        # print newlist
    else:
        newlist = curated_question_list
    # print 'challengeID' in data.columns
    return data.loc[:,newlist]


def fix_remappings(data, remappings):
    for qname, remap in remappings.iteritems():
        original = []
        target = []
        for s,t in remap.iteritems():
            # try:
            #     s = int(s)
            #     t = int(t)
            # except ValueError:
            #     s = float(s)
            #     t = float(t)
            original.append(s)
            target.append(s)
        data[qname].replace(original, target)

    return data


# A - Mean
# D - Median
# O - Mode
# N - Min
# X - Max
# R<A> - LOCF (restricted)
# L<A> - LOCF (unrestricted)
# C | C.# - Chosen value (can be outside or inside range)
# T | T.#.#,#.#... - Distribution

# NOTE: This is desctructive to the input `data`
def impute_data(data, methods, verbose=False):
    # print len(methods.keys())
    i = 0
    num_questions = len(methods.keys())

    curated_qnames = methods.keys()

    todrop = []
    for qname in data.columns:
        if (not qname in curated_qnames) and (not (data[qname] > 0).any()):
            todrop.append(qname)

    data = data.drop(todrop, axis=1)

    for qname in methods.iterkeys():
        if verbose:
            log("**** %s/%s" % (str(i+1), str(num_questions)))
        # print "********************************", i
        i += 1
        # err_vals = filter(lambda x: math.isnan(x) or (x < 0 and x == math.floor(x)), col.unique())
        
        # NOTE: valid_vals includes duplicates (ideal for empirical distribution sampling)
        valid_vals = np.array(filter(lambda x: (isinstance(x, (int, long, float))) and (not math.isnan(x)) and (x >= 0 or x != math.floor(x)), data[qname])) # See Ryan Amos' code, `impute_codebook`

        vmean = round_(np.mean(valid_vals), data[qname])
        # vmean = np.mean(valid_vals), col
        vmedian = round_(np.median(valid_vals), data[qname])
        vmode = scipy.stats.mode(valid_vals).mode[0]
        vmax = max(valid_vals)
        vmin = min(valid_vals)

        # prev_qname = last_observation(data, qname)

        # original = []
        # target = []

        for code in methods[qname].iterkeys():
            method = methods[qname][code]
            mtype = method.split(' ')[0].upper()

            # print "COOOOODE: ", code
            # if code == -9:
            #     print 'hola'
            #     print method
            # print "!!!!!!!!!!!!!!!!!", qname, code, method

            # last_observations = find_last_observations(data, qname, code)

            if np.isnan(code):
                qcodemask = (np.isnan(data[qname]))
            else:
                qcodemask = (data[qname] == code)
            qcodeindices = data[qcodemask].index.values

            # if code ==-9:
            #     print qcodeindices

            # if mtype == 'L' or mtype == 'R':
            #     methods = methods[1:]
            #     mtype = methods[0].upper()

            #     # data.loc[qcodeindices,qname] = data.loc[qcodeindices,prev_qname]
            #     data.loc[qcodeindices,qname] = last_observations
            if mtype.startswith('L') or mtype.startswith('R'):
                data = impute_with_last_observations(data, qname, code)
                # mtype = mtype[1:]
                method = method[1:]
                mtype = method.split(' ')[0].upper()

                if np.isnan(code):
                    qcodemask = (np.isnan(data[qname]))
                else:
                    qcodemask = (data[qname] == code)
                qcodeindices = data[qcodemask].index.values

            if mtype.startswith('T'):
                num_samples = len(qcodeindices)
                ranges = []

                # if not "." in method:
                #     choices = data.loc[data[qname] > 0, qname].values # NOTE: Currently just counts >0 values as valid choices by default
                if len(method.split(' ')) <= 2:
                    ranges.append((np.min(valid_vals), np.max(valid_vals)))
                else:
                    # choices = []
                    # for new_choice in method.split(".")[1:]:
                    #     if ',' in new_choice:
                    #         s = int(new_choice.split(",")[0])
                    #         e = int(new_choice.split(",")[1])
                    #         choices += range(s,e)
                    #     else:
                    #         choices.append(int(new_choice))
                    for new_range in method.split(' ')[1:]:
                        if ',' in new_range:
                            s = int(new_range.split(',')[0])
                            e = int(new_range.split(',')[1])
                            ranges.append((s,e))
                        else:
                            ranges.append((int(new_range), int(new_range)))

                # data.loc[qcodeindices, qname] = random.sample(choices, len(qcodeindices))

                sampled_values = sample_from_distribution(valid_vals, ranges, num_samples)
                data.loc[qcodeindices, qname] = sampled_values

            # else:
            #     # Only add original if we are using replacement later
            #     original.append(code)

            # if code == -9:
            #     print method
            #     print qcodeindices
            chosen_replacement = None

            if mtype.startswith('C'):
                if len(method.split(' ')) == 1:
                    method += " "
                    newval = int(np.floor(vmax) + 1)
                    method += str(newval)

                try:
                    chosen_val = int(method.split(' ')[1])
                except ValueError:
                    chosen_val = float(method.split(' ')[1])
                    # if code == 'nan':
                    #     data = data.fillna({qname:chosen_val})
                    # else:
                    #     data[qname] = data[qname].replace(code, chosen_val)
                chosen_replacement = chosen_val
                # original.append(code)
                # target.append(chosen_val)
            elif mtype.startswith('A'):
                # original.append(code)
                # target.append(vmean)
                # if code == 'nan':
                #     data = data.fillna({qname:vmean})
                # else:
                #     data[qname] = data[qname].replace(code, vmean)
                chosen_replacement = vmean
            elif mtype.startswith('D'):
                # original.append(code)
                # target.append(vmedian)
                chosen_replacement = vmedian
            elif mtype.startswith('O'):
                # original.append(code)
                # target.append(vmode)
                chosen_replacement = vmode
            elif mtype.startswith('N'):
                # original.append(code)
                # target.append(vmin)
                chosen_replacement = vmin
            elif mtype.startswith('X'):
                # original.append(code)
                # target.append(vmax)
                chosen_replacement = vmax

            if code == 'nan':
                data = data.fillna({qname:chosen_replacement})
            else:
                data[qname] = data[qname].replace(code, chosen_replacement)

            # else:
            #     target.append(vmode)

        # data[qname].replace(original, target)

    return data


def impute_naively(data, verbose=False):
    # From FFC example imputation code
    #
    # df = df.fillna(df.mode().iloc[0])
    # # if still NA, replace with 1
    # df = df.fillna(value=1)
    # # replace negative values with 1
    # num = df._get_numeric_data()
    # num[num < 0] = 1
    # # write output csv
    # num.to_csv(background_file, index=False)
    
    data = data.fillna(data.mode().iloc[0])
    data = data.fillna(value=1)
    num = data._get_numeric_data()
    num[num < 0] = 1
    return num


# data is a pandas dataframe
def expand_categorical(data, qtypes):
    new_data = pd.DataFrame()

    for question in data.columns:
        if question in qtypes:
            if not qtypes[question].startswith('nom'):
                new_data[question] = data[question]
            else:
                for value in data[question].unique():
                    new_data[question+"_"+str(value)] = ((data[question] == value)*1)
        elif question == 'challengeID':
            new_data[question] = data[question]

    return new_data

def maybe_format_data_directory():
  '''
  Create subdirectories in data directory if they don't already exist.
  '''
  if not exists(orig_dir): makedirs(orig_dir)
  if not exists(imputed_dir): makedirs(imputed_dir)
  if not exists(trimmed_dir): makedirs(trimmed_dir)
  if not exists(predictions_dir): makedirs(predictions_dir)



# Run script from the command line
if __name__ == "__main__":
    main()





