# Put the license and copyright comments in py files as needed.
# If you are not sure you should run this, then you shouldn't.
# Not very robust; but probably good enough.
# Pipe to a file or run in an editor window so you can review the messages.
# NOTE: when it needs to update a file, it does so in-place.
# NOTE: the copyright line is not pep-8 compliant.

import os
import sys

if len(sys.argv) != 2:
    print ("Usage: python liccopy.py dirname")
    quit()

indir = sys.argv[1]
    
cstmt = "# Copyright 2020 by B. Knueven, D. Mildebrath, C. Muir, J-P Watson, and D.L. Woodruff"
lic = "# This software is distributed under the 3-clause BSD License."

pyomoscent = "#  Pyomo: Python Optimization Modeling Objects"


def _get_file(filepath):
    cline = -1
    licline = -1
    pyomofile = False
    with open(filepath,'r') as f:
        lines = f.readlines()
    for lno, line in enumerate(lines):
        if cstmt in line:
            cline = lno
            print
        if lic in line:
            licline = lno
        if pyomoscent in line:
            pyomofile = True
    return lines, cline, licline, pyomofile


def _write_file(filepath, lines, cline, licline, pyomofile):
    print (filepath, end='')

    if pyomofile:
        print(": seems to have Pyomo header; skipping")
        return
    
    cbefore = -1  # if only adding copyright
    licbefore = -1  # if only adding license
    if cline == -1 and licline != -1:
        cbefore = licline
    if cline != -1 and licline == -1:
        licbefore = licline + 1
    if cline != -1 and licline != -1:
        print(" .")
        return
    
    with open(filepath, "w") as f:
        if cline == -1 and licline == -1:
            f.write(cstmt+'\n')
            f.write(lic+'\n')
            print(": add both")
        for lno, line in enumerate(lines):
            if cbefore == lno:
                f.write(cstmt+'\n')
                print(": add copyright")
            if licbefore == lno:
                f.write(lic+'\n')
                print(": add license")
            f.write(line)


for subdir, dirs, files in os.walk(indir):
    for filename in files:
        filepath = subdir + os.sep + filename

        if filepath.endswith(".py"):
            lines, cline, licline, pyomofile = _get_file(filepath)
            _write_file(filepath, lines, cline, licline, pyomofile)
            
