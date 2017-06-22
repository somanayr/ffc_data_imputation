#!/usr/bin/python

import signal
import sys, os
import argparse
import pandas as pd
import numpy as np
import scipy.stats
import math
import re

BREAK_FLAG = False
SCREEN_CLEAR = chr(27) + "[2J"
ANNOTATIONS = 'x.?'

# Exit cleanly on SIGINT. See https://docs.python.org/2/library/signal.html
def signal_handler(signum, frame):
    global BREAK_FLAG
    BREAK_FLAG = True
signal.signal(signal.SIGINT, signal_handler)


###############################################
#                                             #
#                  Main Loop                  #
#                                             #
###############################################


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--input_codebook", help="Path to input codebook", required=True)
    parser.add_argument("-i", "--input_labels", help="Path to input codebook", required=True)
    # parser.add_argument("-l", "--load_filename", help="Path to in-progress annotation", required=True)
    parser.add_argument("-o", "--output_filename", help="Path to output tsv file", required=True)
    parser.add_argument("-d", "--data_filename", help="Path to FFC data file", required=True)
    parser.add_argument("-ow", "--overwrite", help="[OPTIONAL] Overwrite the specified output file, if any data exists", action="store_true")
    options = parser.parse_args()

    run(options)


def run(options):
    infile = open(options.input_codebook)
    labelfile = open(options.input_labels)
    data = get_data_file(options.data_filename)

#    for i in list(data):
#        if any((h > 0 for v, h in get_locf_stats(data, i))):
#            print i
#            sys.exit(0)
#    sys.exit(0)
            
    question_num = 0

    seen_questions = {}

    # Check if output file already contains annotations
    #    (Assumed true if file exists)
    output_directory, output_filename = os.path.split(options.output_filename)

    if output_directory == "":
        output_directory = "./"

    if (output_filename in os.listdir(output_directory)) and not (options.overwrite):
        # File exists, load previous data
        outfile = open(options.output_filename, "r+")
        for question in load_question_names(outfile):
            seen_questions[question] = True
    else:
        outfile = open(options.output_filename, "w")

    for question_block in parse_questions(infile):
        question_id = get_question_id(question_block)
        
        label_breakdown = labelfile.readline().strip().split('\t')
        if question_id != label_breakdown[0]:
            raise ValueError("Mismatched ids %s, %s" % (question_id, label_breakdown[0]))
        if label_breakdown[1][-1] in ANNOTATIONS:
            annotation = label_breakdown[1][-1]
            qtype = label_breakdown[1][:-1]
        else:
            annotation = ''
            qtype = label_breakdown[1]

        question_num += 1

        if question_id == "idnum": continue
        
        col = data[question_id]

        err_vals = filter(lambda x: x == "Missing"
                                    or x == ""
                                    or (not isinstance(x, basestring) and math.isnan(x))
                                    or (not isinstance(x, basestring) and x < 0 and x == math.floor(x)), col.unique())
        valid_vals = np.array(filter(lambda x: not isinstance(x, basestring) and not math.isnan(x) and (x >= 0 or x != math.floor(x)), col))

        if len(valid_vals) == 0 or len(err_vals) == 0: continue
        if "NOM" in qtype.upper() and not question_id in seen_questions:
            if 'nan' in [str(e) for e in err_vals]:
                actions = ["%s %s" % (str('nan'), 'C')]
                outfile.write(question_id)
                outfile.write("\t")
                outfile.write(";".join(actions))
                outfile.write("\n")
                outfile.flush()
                continue
            else:
                continue
#        print col.unique()
#        print question_block
#        print question_num
                
#        print err_vals
#        print valid_vals
        vmean = round_(np.mean(valid_vals), col)
        vmedian = round_(np.median(valid_vals), col)
        vmode = scipy.stats.mode(valid_vals).mode[0]
        vmax = max(valid_vals)
        vmin = min(valid_vals)
        valid_locf = get_locf_stats(data, question_id)
#        locf_str = ""
        locf_str = ", ".join(("%s -> %d%%" % x for x in valid_locf ))

#        locf_str = valid_locf

        
        # Do not re-process previously evaluated questions
        if not question_id in seen_questions:

            i = 0
            actions = []
            while True:
                err_val = err_vals[i]
                # See http://stackoverflow.com/questions/2084508/clear-terminal-in-python
                print SCREEN_CLEAR
                print question_block
                print question_num
                # Use suffix "?" to denote more research required
                #            "x" to denote likely candidate for removal
                #            "." to denote more processing required (e.g. string variables)
#                print "Annotations:\n? - Flag question\nx - Flag for removal\n. - Flag answer"
                print """
    Q to quit
    A - Mean         {0}
    D - Median       {1}
    O - Mode         {2}
    N - Min          {3}
    X - Max          {4}
    R<A> - LOCF (restricted) - {5}
    L<A> - LOCF (unrestricted)
    C | C # - Chosen value (can be outside or inside range)
    T | T # #,# #... - Distribute""".format(vmean, vmedian, vmode, vmin, vmax, locf_str)
                if err_val == -9:
                    action = 'LT'
                else:
                    action = raw_input("Action %s: " % str(err_val)).upper()
                if action == 'q' or action == 'Q':
                    break

                if len(action) == 0: continue

                try:
                    if not verify_action(action, col):
                        continue
                except:
                    continue
                actions.append("%s %s" % (str(err_val), action))
                i += 1
                if i >= len(err_vals): break

            if action == 'q' or action == 'Q':
                break
                
            outfile.write(question_id)
            outfile.write("\t")
            outfile.write(";".join(actions))
            #if len(remap) > 0:
            #    for s,t in remap.iteritems():
            #        outfile.write("\t")
            #        outfile.write(str(s))
            #        outfile.write(",")
            #        outfile.write(str(t))
            outfile.write("\n")
            outfile.flush()

    outfile.flush()
    outfile.close()

    # outfile.write("testing\tagain??\n")
    # outfile.flush()
    # outfile.close()

    # print seen_questions


###############################################
#                                             #
#             Auxiliary Methods               #
#                                             #
###############################################


# Generator for question names in an existing input file
def load_question_names(infile):
        for line in infile:
            yield line.strip().split("\t")[0]


def parse_questions(infile):
    current_block = ""
    flag = -2

    while flag < 0:
        next_line = infile.readline()
        if next_line.startswith("--"):
            if flag == -2:
                flag = -1
            else:
                current_block += next_line
                flag = 0

    for line in infile:
        if flag > 0:
            if line.startswith("--"):
                current_block += line # For nice formatting of output, duplicate delimiter line
                yield current_block
                current_block = ""
                current_block += line
                flag = 0
            else:
                current_block += line
        elif flag == 0:
            current_block += line
            if line.startswith("-"):
                flag = 1
        # elif flag == 1:
        #     current_block += line
        #     if line.startswith("-"):
        #         flag = 2


def get_question_id(block):
    return block.split("\n")[1].strip().split()[0]

def get_labels(block):
    idx = block.index("tabulation")
    if idx == -1: return DEFAULT_MAPPING
    mapping = {float('nan'): 'NaN Redacted'}
    idx = block.index("\n", idx)
    for l in block[idx:].split('\n'):
        numeric = int(l[33:40])
        label = l[42:]
        mapping[numeric] = label
    return mapping
        

def get_data_file(datafn):
    df = pd.read_csv(datafn, low_memory=False)
    # Fix date bug
    df.cf4fint = ((pd.to_datetime(df.cf4fint) - pd.to_datetime('1960-01-01')) / np.timedelta64(1, 'D')).astype(int)
    return df

def verify_action(action, col):
    if action[0] in 'LR' and len(action) > 1:
        ret, act2 = verify_action(action[1:], col)
        return ret, action[0] + act2
    elif action[0] in 'ADONX':
        return len(action) == 1, action
    elif action[0] == 'C':
        if len(action) == 1: #Auto assign category of max+1
            return True, action + ' ' + str(max(col) + 1)
        else:
            if len(action) < 3 or action[1] != '.':
                return False, action
            try:
                if col.dtype == np.int64:
                    val = int(action[2:])
                elif col.dtype == np.float64:
                    val = float(action[2:])
                else:
                    raise TypeError("Unknown column type")
            except ValueError:
                return False, action
            return True, action
    elif action[0] == 'T':
        if len(action) < 3:
            return len(action) == 1, action
        for x in action[2:].split(' '):
            if ',' in x:
                try:
                    s,e = x.split(',')
                    float(s)
                    float(e)
                    return True, action
                except ValueError:
                    return False, action
            else:
                try:
                    float(x)
                    return True, action
                except ValueError:
                    return False, action
    return False

def round_(val, col):
    if col.dtype == np.int64:
        return np.int64(np.round_(val))
    elif col.dtype == np.float64:
        return val
    else:
        raise TypeError()

def get_locf_stats(data, question_id):
    return last_observed(data, question_id)



#FIXME this needs to be modified for this project and incorporated in get_locf_stats
question_re = re.compile(r"^([^\d]+)(\d+)(.*)$")
def last_observed(data, question_id):
    question = question_re.match(question_id)
    if question == None:
        print question_id
        return []
        
    who = question.group(1)
    if who.startswith("kind"): return []
    when = int(question.group(2))
    what = question.group(3)

    
    col = data[question_id]
    prev_col = None
    
    while when > 1:
        when -= 1
        prev_question = "%s%d%s" % (who, when, what)
        try:
            prev_col = data[prev_question]
            break
        except KeyError:
            pass
    if prev_col is None:
        return []

    pairings = []
    for i in col.unique():
        if not math.isnan(i) and (i >= 0 or i != math.floor(i)):
            continue
        hits = 0
        ct = 0
        #print i
        for j in range(len(col)):
            if col[j] == i or (math.isnan(i) and math.isnan(col[j])):
                ct += 1
                prev_val = prev_col[j]
                if not math.isnan(prev_val) and (prev_val >= 0 or i != math.floor(prev_val)):
                    hits += 1
        pairings.append((i, hits * 100 / ct))
        #print hits, ct
    return pairings
    


# Run the script from command line
if __name__ == "__main__":
    main()
    

