// CPP file for generic multithreading + overlaid asynchronous MPI
// communication work dispatcher.  The only code here is the little widget to
// invoke performTask with an exposed class pointer.


#include "groupController.hpp"


// Helper function to call subdomain::performTask with an exposed class pointer, so
// that it can be invoked without implicit class context from std::async 

void runTask(subdomain* ptr)
{
        ptr->performTask();
}
