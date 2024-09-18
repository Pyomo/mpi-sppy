W-based rho
===========

A rho based on W is directly supported at the time of this writing. The file
``mpisppy.utils.find_rho.py`` can compute rho
using a cost input file that is created by the gradient software.
If you want to use W values as the cost, you would need to modify
the code in ``wxbarwriter.py`` to write the final W values
in the right format (i.e., that matches the cost output format
used by the gradient software).

