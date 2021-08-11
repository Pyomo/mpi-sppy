#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________
#
# This file was originally part of PySP and Pyomo, available: https://github.com/Pyomo/pysp
# Copied with modification from pysp/phutils.py

from pyomo.core import Objective


class BasicSymbolMap:

    def __init__(self):

        # maps object id()s to their assigned symbol.
        self.byObject = {}

        # maps assigned symbols to the corresponding objects.
        self.bySymbol = {}

    def getByObjectDictionary(self):
        return self.byObject

    def updateSymbols(self, data_stream):
        # check if the input is a generator / iterator,
        # if so, we need to copy since we use it twice
        if hasattr(data_stream, '__iter__') and \
           not hasattr(data_stream, '__len__'):
            obj_symbol_tuples = list(obj_symbol_tuples)
        self.byObject.update((id(obj), label) for obj,label in data_stream)
        self.bySymbol.update((label,obj) for obj,label in data_stream)

    def createSymbol(self, obj ,label):
        self.byObject[id(obj)] = label
        self.bySymbol[label] = obj

    def addSymbol(self, obj, label):
        self.byObject[id(obj)] = label
        self.bySymbol[label] = obj

    def getSymbol(self, obj):
        return self.byObject[id(obj)]

    def getObject(self, label):
        return self.bySymbol[label]

    def pprint(self, **kwds):
        print("BasicSymbolMap:")
        lines = [repr(label)+" <-> "+obj.name+" (id="+str(id(obj))+")"
                 for label, obj in self.bySymbol.items()]
        print('\n'.join(sorted(lines)))
        print("")


#
# a simple utility function to pretty-print an index tuple into a [x,y] form string.
#

_nontuple = (str, int, float)
def indexToString(index):

    if index is None:
        return ''

    # if the input type is a string or an int, then this isn't a tuple!
    # TODO: Why aren't we just checking for tuple?
    if isinstance(index, _nontuple):
        return "["+str(index)+"]"

    result = "["
    for i in range(0,len(index)):
        result += str(index[i])
        if i != len(index) - 1:
            result += ","
    result += "]"
    return result

#
# a simple utility to determine if a variable name contains an index specification.
# in other words, is the reference to a complete variable (e.g., "foo") - which may
# or may not be indexed - or a specific index or set of indices (e.g., "foo[1]" or
# or "foo[1,*]".
#

def isVariableNameIndexed(variable_name):

    left_bracket_count = variable_name.count('[')
    right_bracket_count = variable_name.count(']')

    if (left_bracket_count == 1) and (right_bracket_count == 1):
        return True
    elif (left_bracket_count == 1) or (right_bracket_count == 1):
        raise ValueError("Illegally formed variable name="+variable_name+"; if indexed, variable names must contain matching left and right brackets")
    else:
        return False


#
# related to above, extract the index from the variable name.
# will throw an exception if the variable name isn't indexed.
# the returned variable name is a string, while the returned
# index is a tuple. integer values are converted to integers
# if the conversion works!
#

def extractVariableNameAndIndex(variable_name):

    if not isVariableNameIndexed(variable_name):
        raise ValueError(
            "Non-indexed variable name passed to "
            "function extractVariableNameAndIndex()")

    pieces = variable_name.split('[')
    name = pieces[0].strip()
    full_index = pieces[1].rstrip(']')

    # even nested tuples in pyomo are "flattened" into
    # one-dimensional tuples. to accomplish flattening
    # replace all parens in the string with commas and
    # proceed with the split.
    full_index = full_index.replace("(",",").replace(")",",")
    indices = full_index.split(',')

    return_index = ()

    for index in indices:

        # unlikely, but strip white-space from the string.
        index=index.strip()

        # if the tuple contains nested tuples, then the nested
        # tuples have single quotes - "'" characters - around
        # strings. remove these, as otherwise you have an
        # illegal index.
        index = index.replace("\'","")

        # if the index is an integer, make it one!
        transformed_index = None
        try:
            transformed_index = int(index)
        except ValueError:
            transformed_index = index
        return_index = return_index + (transformed_index,)

    # IMPT: if the tuple is a singleton, return the element itself.
    if len(return_index) == 1:
        return name, return_index[0]
    else:
        return name, return_index


#
# given a component (the real object, not the name) and an
# index template, "shotgun" the index and see which variable
# indices match the template. the cardinality could be >
# 1 if slices are specified, e.g., [*,1].
#

def extractComponentIndices(component, index_template):

    component_index_dimension = component.dim()

    # do special handling for the case where the component is
    # not indexed, i.e., of dimension 0. for scalar components,
    # the match template can be the empty string, or - more
    # commonly, given that the empty string is hard to specify
    # in the scenario tree input data - a single wildcard character.
    if component_index_dimension == 0:
       if (index_template != '') and (index_template != "*"):
          raise RuntimeError(
              "Index template=%r specified for scalar object=%s"
              % (index_template, component.name))
       return [None]

    # from this point on, we're dealing with an indexed component.
    if index_template == "":
        return [ndx for ndx in component]

    # if the input index template is not a tuple, make it one.
    # one-dimensional indices in pyomo are not tuples, but
    # everything else is.
    if type(index_template) != tuple:
        index_template = (index_template,)

    if component_index_dimension != len(index_template):
        raise RuntimeError(
            "The dimension of index template=%s (%s) does match "
            "the dimension of component=%s (%s)"
            % (index_template,
               len(index_template),
               component.name,
               component_index_dimension))

    # cache for efficiency
    iterator_range = [i for i,match_str in enumerate(index_template)
                      if match_str != '*']

    if len(iterator_range) == 0:
        return list(component)
    elif len(iterator_range) == component_index_dimension:
        if (len(index_template) == 1) and \
           (index_template[0] in component):
            return index_template
        elif index_template in component:
            return [index_template]
        else:
            raise ValueError(
                "The index %s is not valid for component named: %s"
                % (str(tuple(index_template)), component.name))

    result = []

    for index in component:

        # if the input index is not a tuple, make it one for processing
        # purposes. however, return the original index always.
        if component_index_dimension == 1:
           modified_index = (index,)
        else:
           modified_index = index

        match_found = True # until proven otherwise
        for i in iterator_range:
            if index_template[i] != modified_index[i]:
                match_found = False
                break

        if match_found is True:
            result.append(index)

    return result
