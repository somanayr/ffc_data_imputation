#!/usr/bin/python

import signal
import sys, os
import argparse

# BREAK_FLAG = False
SCREEN_CLEAR = chr(27) + "[2J"

# Exit cleanly on SIGINT. See https://docs.python.org/2/library/signal.html
# def signal_handler(signum, frame):
#     global BREAK_FLAG
#     BREAK_FLAG = True
# signal.signal(signal.SIGINT, signal_handler)


###############################################
#                                             #
#                  Main Loop                  #
#                                             #
###############################################


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_codebook", help="Path to input codebook", required=True)
    # parser.add_argument("-l", "--load_filename", help="Path to in-progress annotation", required=True)
    parser.add_argument("-o", "--output_filename", help="Path to output tsv file", required=True)
    parser.add_argument("-ow", "--overwrite", help="[OPTIONAL] Overwrite the specified output file, if any data exists", action="store_true")
    options = parser.parse_args()

    run(options)


def run(options):
    infile = open(options.input_codebook)

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

        question_num += 1

        # Do not re-process previously evaluated questions
        if not question_id in seen_questions:
            # See http://stackoverflow.com/questions/2084508/clear-terminal-in-python
            print SCREEN_CLEAR
            print question_block
            print question_num
            # Use suffix "?" to denote more research required
            #            "x" to denote likely candidate for removal
            #            "." to denote more processing required (e.g. string variables)
            print "Annotation suffixes: ?, x, ."
            action = raw_input("Data type (go, bo, nom, con) or 'q': ")

            if len(action) == 0: action = ' '
            if len(action) == 1: action = action + ' '
            if action[1] != 'o':
                if action[0] == 'g':
                    action = 'go' + action[1:]
                elif action[0] == 'b':
                    action = 'bo' + action[1:]
                elif action[0] == 'n':
                    action = 'nom' + action[1:]
                elif action[0] == 'c':
                    action = 'con' + action[1:]
            if action[-1] == ' ': action = action[:-1]
            
            remap = {}

            if action == 'q' or action == 'Q':
                break

            if action.lower().startswith('bo'):
                new_remap = ""
                while not new_remap.lower().startswith("x"):
                    new_remap = raw_input("{[original_value] [new_value]} or {x} to end: ")
                    parts = new_remap.strip().split()
                    if len(parts) >= 2:
                        s = parts[0]
                        t = parts[1]
                        remap[s] = t


            outfile.write(question_id)
            outfile.write("\t")
            outfile.write(action)
            if len(remap) > 0:
                for s,t in remap.iteritems():
                    outfile.write("\t")
                    outfile.write(str(s))
                    outfile.write(",")
                    outfile.write(str(t))
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
        if next_line.startswith("-"):
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
            if line.startswith("--"):
                flag = 1
        # elif flag == 1:
        #     current_block += line
        #     if line.startswith("-"):
        #         flag = 2


def get_question_id(block):
    return block.split("\n")[1].strip().split()[0]


# Run the script from command line
if __name__ == "__main__":
    main()

