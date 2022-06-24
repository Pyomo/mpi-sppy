#!/bin/bash
# Sed commands to do some of the work converting to new config in mpi-sppy

if [ $# != 2 ]
then
   echo "Must supply the input file name and output file name"
   exit 1
fi

# just copy then edit in-place
cp ${1} ${2}

sed -i 's/import argparse/from mpisppy\.utils import config/g' ${2}

sed -i 's/parser\.add_argument("--/cfg.add_to_config("/g' ${2}
# in case the first one did not hit; leave them with a bit of a mess...
sed -i 's/iparser\.add_argument/cfg.add_to_config/g' ${2}

sed -i 's/parser\.add_argument/cfg.add_to_config/g' ${2}

# note: you still need to change the - to _ inside arg names
sed -i '/add_to_config/s/-/_/g' ${2}

sed -i 's/help=/description=/g' ${2}
sed -i 's/help =/description=/g' ${2}

sed -i 's/type=/domain=/g' ${2}
sed -i 's/type =/domain=/g' ${2}

# delete the dest line
sed -i '/dest=/d' ${2}

# NOTE: you may still need to deal with booleans "by hand"
sed -i "s/action='store_true'/domain=bool,\n            default=False/g" ${2}
sed -i "s/action ='store_true'/domain=bool,\n            default=False/g" ${2}
sed -i '/parser\.set_defaults/d' ${2}

# The config setup functions do not take arguments
sed -i "/cfg/s/(parser)/(cfg)/g" ${2}
sed -i "/cfg/s/(inparser)/(cfg)/g" ${2}

# use cfg in beans
sed -i '/bean/s/args/cfg/g' ${2}

sed -i 's/args\./cfg\./g' ${2}

# (note that " and ' are not the same for bash. You cannot evaluate ${1} inside single quotes)
# The next two commands might make a mess with actually parsing, but might be better than doing nothing
sed -i "s/args = parser\.parse_args()/cfg\.parse_command_line(${1})/g" ${2}

# if you have a function that sets up the parser, this might almost do the job (except you need to fix the caller; and this might be redundant...)
sed -i "s/return parser/cfg\.parse_command_line(${1})/g" ${2}

# we should not be using argparse to create the parser
sed -i '/argparse\.ArgumentParser/d' ${2}


echo "You probably still need to edit (e.g., look at create_parser and lines near it)"
echo "  ...and there may be more issues associated with parsers; see farmer_cylinders.py for an example"
echo "    ...and check your boolean command line args"
echo
echo "And you probably need cfg = config.Config() in __main__ or someplace high up; and you will need cfg passed in wherever else cfg is used"
