#pragma once


// Generic work manager for MPI ranks that use multithreading to asychronously
// work on a collection of "subdomains", in parallel with a single collective
// coordination process.


#include "utils/utilities.hpp"    // Indirectly includes MPI plus some standard stuff
#include "config/config.hpp"      // CLI-11 based command line interface

#include <sstream>
#include <iostream>
#include <fstream>
#include <chrono>
#include <future>
#include <thread>
#include <vector>
#include <tuple>
#include <ctime>
#include <stdexcept>
#include <cstdlib>  // std::abort

// Declaration of routine to run tasks on subdomains, with an exposed
// subdomain pointer. Defined in the .cpp file and invoked by "async".  The
// subdomain class is defined immediately below

class subdomain;

void runTask(subdomain* ptr);


class subdomain :
virtual public debugLevels
{

public:

    // Constructor

    subdomain() :
    taskType(0),
    abortFlag(false)
    { };


    // Asynchronously start a task of type "taskType_" on this subdomain
    // Help force true async execution. Without a launch policy,
    // std::async is allowed to run "deferred" (never starts until wait/get),
    // which could make taskIsComplete() return false forever.

  void dispatchTask(int taskType_) {
    taskType = taskType_;
    abortFlag = false;

    waiter = std::async(std::launch::async, [this] {
      try {
        runTask(this);
      }
      catch (const std::exception& e) {
        std::cerr << "FATAL: std::exception in subdomain taskType=" << taskType
                  << ": " << e.what() << "\n";
        std::cerr.flush();
        std::abort();

      } catch (...) {
        std::cerr << "FATAL: unknown exception in subdomain taskType=" << taskType << "\n";
        std::cerr.flush();
        std::abort();
      }
    });
  }

    // What task is currently being performed, or was most recently performed

    int currentTaskType(void) { return taskType; };


    // Virtual function to perform the task itself (should look at taskType if
    // it is allowed to have more than one value).  This will get called from
    // runTask.

    virtual void performTask(void) = 0;


    // Signal abort

    virtual void signalAbort(void) { abortFlag = true; };


    // Check if the task is complete

    bool taskIsComplete(void)
    {
        auto notime = std::chrono::seconds::zero();
        return waiter.wait_for(notime) == std::future_status::ready;
    }


    // Destructor (does nothing at present).  Suggestion: make sure no task is
    // presently running if this is called

    virtual ~subdomain() { };


protected:

    // Working data members

    std::future<void> waiter;       // To check when tasks are done
    int               taskType;     // Type of task being performed
    std::atomic<bool> abortFlag;    // To signal the current task to abort, if possible

};



// This class manages dispatch and coordination; it is templated on the kind
// of subdomain

template<class SUBDOMAINCLASS>
class groupController :
virtual public commBased
{
public:

    // Constructor (sets up basic MPI stuff, most of the rest set to some probably
    // inadequate values)

    groupController() :
    numTaskTypes(1),
    subdomainName("subdomain"),
    subdomainObj(NULL),
    commActive(false),
    cycleCount(0),
    maxConcurrentComms(1),
    concurrentCommCount(0),
    commPhase(0),
    terminateFlag(false),
    maxWallTimeSeconds(0.0)
    {
        taskName.resize(numTaskTypes);
        taskName[0] = "default task";
    };


    // Set up configuration options for the command line parser.  These options
    // are potentially common to all groupController-based applications.

    static void registerOptions(Config& cfg)
    {
        cfg.addOption("verbosity",
                      0 /* default value */,
                      "Debugging output verbosity level");

        cfg.addOption("rank0-verbosity",
                      0 /* default value*/,
          "Override debugging output level that applies only to MPI rank 0");

        cfg.addOption("workersPerRank",
                      1 /* default value */,
                      "Number of worker processes per MPI rank");

        cfg.addOption("sleepMsec",
                      20 /* default value */,
                      "Milliseconds controller sleeps when no events appear pending");

        cfg.addOption("localTimeRatios",
                      false /* default value */,
                      "Whether to report local CPU/wall time ratios");

        cfg.addOption("eventLogFile",
                      std::string(""),
                      "File name for event log (leave empty for no log)");

        // Added January 2026: hard wall-clock termination (0 disables)
        cfg.addOptionNonNegative<double>("maxWallTimeSeconds",
                                         0.0,
                                         "Terminate after this many wall-clock seconds (0 disables)");
    };


    // Main setup / initialization routine

    virtual void setup(Config &cfg) {
        cfgPtr = &cfg;
        startWallTime = MPI_Wtime();
        startCPUTime  = clock();

        // Make sure the thread support level is sufficient

        int supportedThreadLevel = MPI_THREAD_SINGLE;
        MPI_Query_thread(&supportedThreadLevel);
        if (supportedThreadLevel == MPI_THREAD_SINGLE)
        {
            // Either the supported thread level was not set, or it was set to
            // MPI_THREAD_SINGLE, which isn't enough

            if (myRank == 0)
                std::cerr << "Error: current MPI environment does not support the "
                          << "required thread level"
                          << std::endl << std::flush;
            MPI_Abort(MPI_COMM_WORLD,1);
        }

        // Inhale option values from the configuration object.  The option
        // "rank0-verbosity" is a special one that overrides the debug level if
        // specified, but only on MPI rank 0

        setVerbosity(cfgPtr->getOptionValue<int>("verbosity"));
        if (myRank == 0 && cfgPtr->explicitlyGiven("rank0-verbosity"))
            setVerbosity(cfgPtr->getOptionValue<int>("rank0-verbosity"));

        numWorkers            = cfgPtr->getOptionValue<int>("workersPerRank");
        sleepMsec             = cfgPtr->getOptionValue<int>("sleepMsec");
        reportLocalTimeRatios = cfgPtr->getOptionValue<bool>("localTimeRatios");

        eventFileName         = cfgPtr->getOptionValue<std::string>("eventLogFile");
        eventLogging          = (eventFileName != "");

        maxWallTimeSeconds    = cfgPtr->getOptionValue<double>("maxWallTimeSeconds");
        if (maxWallTimeSeconds < 0.0)
            throw std::invalid_argument("maxWallTimeSeconds must be >= 0");

        // There is no point in having more worker processes than subdomains
        // they can work on.  This used to issue a warning, but now it's silent
        // because we'll let applications decide whether a warning is needed

        if (numWorkers > numSubdomains)
            numWorkers = numSubdomains;

        // Convert sleep time interval to weird "chrono" datatype

        std::chrono::milliseconds tempTime(sleepMsec);
        sleepTime = tempTime;

        // Reset all working variables

        cycleCount = 0;
        commPhase  = 0;
        commActive = false;

        eligible.clear();
        active.clear();

        if (subdomainObj != NULL)
            delete[] subdomainObj;
        subdomainObj = new SUBDOMAINCLASS[numSubdomains];
        DEBUGPR(3, rankTag << "Subdomain objects allocated/constructed");

        for (int s=0; s < numSubdomains; s++)
            subdomainObj[s].setVerbosity(verbosity);

        dispatchCount.resize(numTaskTypes);
        for (int t=0; t<numTaskTypes; t++)
            dispatchCount[t] = 0;

        requestVector.resize(maxConcurrentComms);
        statusVector.resize(maxConcurrentComms);
        signalCommsDone();

        totalDispatchCount = 0;

        abortCount = 0;

        terminateFlag = false;

        if (eventLogging)
            eventList.clear();

        DEBUGPR(3, rankTag << "GroupController setup complete");
    };


    // Set maximum number of concurrent communication operations (per rank)

    void setMaxConcurrentComms(int maxConcurrentComms_)
    {
        maxConcurrentComms = maxConcurrentComms_;
        requestVector.resize(maxConcurrentComms);
        statusVector.resize(maxConcurrentComms);
    };


    // Standard run sequence -- perform the (probably user-specified) startup
    // sequence, then enter the main loop, and finally perform the (probably
    // user-specified) shutdown sequence

    void run(void)
    {
        startupSequence();

        mainEventLoop();

        terminationSequence();

        endWallTime = MPI_Wtime();
        endCPUTime  = clock();

        if (eventLogging)
            outputEventLog();
    };


    // Placeholder to output results

    virtual void outputResults() { };


    // Print run statistics after process completion (must be called synchronously)

    virtual void printRunStatistics(void)
    {
        const int width = 12;

        int savePrecision = -1;

        // Generic communication information

        printStatRank0<int>(cycleCount,"communication cycle",1,width);

        // Application-specific communication information (if any)

        printApplicationCommStats(width);

        // Generic task statistics

        std::string legend;
        for (int t=0; t<numTaskTypes; t++)
        {
            legend = taskName[t] + "s dispatched";
            sumRunStatPrint<long long int>(dispatchCount[t],
                                           legend.c_str(),
                                           MPI_LONG_LONG,
                                           width);
        }

        legend = subdomainName + "s aborted";
        sumRunStatPrint<long long int>(abortCount, legend.c_str(), MPI_LONG_LONG, width);

        // Application-specific task statistics (if any)

        printApplicationTaskStats(width);

        // Generic timing statistics

        double elapsedWallTime = endWallTime - startWallTime;
        printStatRank0<double>(elapsedWallTime,"wall clock second",1,width);

        double usedCPUTime   = ((double) endCPUTime - startCPUTime) / CLOCKS_PER_SEC;
        double localCPUratio = usedCPUTime / elapsedWallTime;
        double temp          = localCPUratio / commSize;

        sumRunStatPrint<double>(usedCPUTime,
                                "seconds total CPU time\n",
                                MPI_DOUBLE,
                                width,
                                1);           // Set precision to 1 for this print

        sumRunStatPrint<double>(temp,
                                "average local CPU/wall ratio",
                                MPI_DOUBLE,
                                width);

        temp = usedCPUTime / elapsedWallTime;

        sumRunStatPrint<double>(temp,
                                "overall CPU/wall ratio\n",
                                MPI_DOUBLE,
                                width);

        printApplicationTimeStats(width);

        // If the verbosity level is set to 1, print out the individual CPU/wall
        // time ratio on each rank, in order (this is not efficient since it uses
        // a gather operation, but it's only used for performance debugging, and only
        // at the end)

        if (reportLocalTimeRatios)
        {
          std::vector<double> gatherBuffer;  // only rank 0 needs storage
          if (myRank == 0) gatherBuffer.resize(commSize);

          MPI_Gather(&localCPUratio, 1, MPI_DOUBLE,
                     myRank == 0 ? gatherBuffer.data() : nullptr, 1, MPI_DOUBLE,
                     0, comm);

          if (myRank == 0) {
            std::cout << "local CPU/wall ratios:";
            for (int r = 0; r < commSize; ++r)
              std::cout << ' ' << gatherBuffer[r];
            std::cout << "\n\n" << std::flush;
          }
        }
        // Restore the state of the cout stream

        if (myRank == 0)
        {
            std::cout << std::defaultfloat;
            std::cout.precision(savePrecision);
        }
    };


    //// Virtual methods to print application-specific run statistics.
    //// The default versions do nothing

    virtual void printApplicationCommStats(int width) { };

    virtual void printApplicationTaskStats(int width) { };

    virtual void printApplicationTimeStats(int width) { };


    // Destructor

    ~groupController()
    {
        if (subdomainObj != NULL)
            delete[] subdomainObj;
    };


protected :

    // Return a pointer to the next free MPI request object

    MPI_Request* nextFreeRequest(void)
    {
        DEBUGPR(1,rankTag << "nextFreeRequest entered with concurrentCommCount="
                          <<  concurrentCommCount);

        if (concurrentCommCount == maxConcurrentComms)
            throw std::runtime_error("Maximum count of " + std::to_string(maxConcurrentComms)
                                     + " concurrent communications violated in cycle "
                                     + std::to_string(cycleCount) + ", phase "
                                     + std::to_string(commPhase));
        return &requestVector[concurrentCommCount++];
    }

    // when we know all comms are done, set the comm count to zero
    void signalCommsDone() {
        concurrentCommCount = 0;
    }

    // Return the index of the just-returned request and status (basically, in case
    // the application wants to store it to look at the status for some reason)

    int indexOfJustAllocatedRequest(void) { return concurrentCommCount - 1; };


    // Some special action to be performed at the beginning.  The default is
    // simply to start the first kind of task on all subdomains

    virtual void startupSequence(void)
    {
        startTaskOnAllSubdomains(0);
    };


    // Implements any application-specific conditions needed to dispatch
    // a subdomain to a worker.  The default is to allow everything

    virtual bool dispatchAllowed(void) { return true; };


    // Main event loop that orchestrates everything for most of the run

    void mainEventLoop(void)
    {
        DEBUGPR(1,rankTag << "Enter mainEventLoop()");

        int requestDoneFlag = false;

        // Loop indefinitely (termination detection will cause the method to return
        // from inside this loop)

        while (true)
        {
            // NEW: wall-clock termination check (applies even if no comm completes)
            if (!terminateFlag && maxWallTimeSeconds > 0.0)
            {
                double elapsed = MPI_Wtime() - startWallTime;
                if (elapsed >= maxWallTimeSeconds)
                {
                    if (myRank == 0)
                    {
                        std::cerr << "Terminating: reached maxWallTimeSeconds="
                                  << maxWallTimeSeconds
                                  << " (elapsed=" << elapsed << ")"
                                  << std::endl << std::flush;
                    }
                    signalTermination();
                    return;
                }
            }

            DEBUGPR(3,rankTag << "Top of the loop in mainEventLoop");
            // First deal with any completed tasks

            bool somethingFinished = scanForFinishedTasks();

            // If communication is active, next check if all the currently
            // running reductions are done. If so, handle the results as
            // needed.  This might signal termination, in which case we bail
            // out.  If communication was not active, start it if possible.

            if (commActive)
            {
                if (requestVector.size() < concurrentCommCount)
                    throw std::runtime_error("requestVector dimension is "
                                             + std::to_string(requestVector.size())
                                             + ", but " + std::to_string(concurrentCommCount)
                                             + " communicators needed");
                if (statusVector.size() < concurrentCommCount)
                    throw std::runtime_error("statusVector dimension is "
                                             + std::to_string(requestVector.size())
                                             + ", but " + std::to_string(concurrentCommCount)
                                             + " communicators needed");
                MPI_Testall(concurrentCommCount,
                            requestVector.data(),
                            &requestDoneFlag,
                            statusVector.data());
                if (requestDoneFlag)
                {
                    DEBUGPR(1,rankTag << "about to set concurrentCommCount=0; currently="
                          <<  concurrentCommCount);

                    concurrentCommCount = 0;
                    bool shouldTerminate = handleCompletedCommunication();
                    if (shouldTerminate)
                    {
                        onMainLoopTerminate();
                        return;
                    }
                }
            }
            else if (canStartComm())
                startCommCycle();

            // If it's possible to dispatch a new task, find the best one to
            // start and send it off.  It seems likely that only one or a few
            // workers will open up at a time, so we just send one here, so we
            // don't have to do some kind of sort or generalized median
            // operation on the subdomains. If there are multiple workers
            // open, we'll soon loop around and dispatch another task.  We use
            // an iterator to point to the subdomain index, so that it's easy
            // to delete it from the eligible set

            // The basic conditions for dispatch are that there is an available
            // worker and there is a subdomain eligible for dispatch.  Furthermore,
            // dispatchAllowed() implements additional application specific
            // conditions that might have to be met.

            if ((active.size() < numWorkers) && 
                !eligible.empty()            &&
                dispatchAllowed())
            {
                auto [dispatchIt,taskType] = bestTaskToDispatch();
                dispatchFromEligible(dispatchIt, taskType);
            }

            // If it doesn't seem like there's a chance to send out more work right
            // away, take a little rest from constant polling and testing

            if (!somethingFinished && active.size() == numWorkers)
                controllerSleep();
        }
    }


    // Triggered at the end of each communication operation.  Signals
    // termination by returning true.  The default never does signals
    // termination, but can be used as the end of application-specific methods

    virtual bool handleCompletedCommunication(void)
    {
        DEBUGPR(4,rankTag << "generic handleCompletedCommunication, cycleCount="
                          << cycleCount << ", phase=" << commPhase);

        if (canStartComm())
            startCommCycle();
        else
            clearCommActive();
        return false;               // Not terminating
    };


    // Decides which eligible subdomain is the best one to send out and what kind of task
    // to perform on it.  The simple default is to choose the first eligible
    // subdomain and assign task "0".

    virtual std::pair<std::set<int>::iterator,int> bestTaskToDispatch(void)
    {
        return { eligible.begin(), 0 };
    }


    // Method to shut everything down at the end.  The default is just to wait for
    // all active tasks to complete

    virtual void terminationSequence(void)
    {
        DEBUGPR(3,rankTag << "Generic termination sequence entered, with "
                          << active.size() << " active " << subdomainName << 's');

        while (!active.empty())
        {
            bool somethingFinished = scanForFinishedTasks();
            if (!somethingFinished)
                controllerSleep();
        }

        DEBUGPR(3,rankTag << "Generic termination sequence entered, with "
                          << active.size() << " active " << subdomainName << 's');
    };


    // Perform a certain task on every subdomain -- generally not to be
    // overlapped the main dispatch loop, but before it starts or after it
    // ends.  This routine may not be needed in some applications

    void runTaskOnAllSubdomains(int taskType)
    {
        startTaskOnAllSubdomains(taskType);
        finishTaskOnAllSubdomains(taskType);
    };


    // To start solving all scenarios (get as many solves started as possible)
    // At present, to be run before or after main event loop.

    void startTaskOnAllSubdomains(int taskType)
    {
        // Mark every task as either eligible if it's not currently
        // running), or as needed to be run again (needRedo) if it is

        needRedo.clear();
        eligible.clear();

        for (int s=0; s<numSubdomains; s++)
        {
            if (isInSet(s,active))
                needRedo.insert(s);
            else
                makeEligible(s);
        }

        // Now try to start anything we have room for, moving it from
        // eligible to active

        while (active.size() < numWorkers && eligible.size() > 0)
        {
            auto it = eligible.begin();
            int s = *it;
            eligible.erase(it);
            startTask(s, taskType);
            active.insert(s);
        }

        DEBUGPR(4, rankTag << "StartTaskOnAllSubdomains with type " << ((int) taskType)
                           << ", active=" << setToString(active)
                           << ", eligible=" << setToString(eligible)
                           << ", needRedo=" << setToString(needRedo));
    }



    // To finish solving all scenarios (finish everything, loading more as you go)
    // At present, to be run before or after main event loop

    void finishTaskOnAllSubdomains(int type)
    {
        while (!active.empty() || !eligible.empty())
        {
            // Process any completed tasks and make them no longer active

            bool somethingFinished = scanForFinishedTasks();

            // If any of just-finished subdomains needed to be re-run, make
            // them eligible

            if (somethingFinished)
            {
                while (justFinished.size() > 0)
                {
                    int s = justFinished.front();
                    justFinished.pop_front();
                    auto it = needRedo.find(s);
                    if (it != needRedo.end())
                    {
                        makeEligible(s);
                        needRedo.erase(it);
                    }
                }

            }

            // If there's room for any more subdomains to run and we have some that
            // need solving, launch them (in no particular order)

            while (active.size() < numWorkers && !eligible.empty())
            {
                auto firstEligible = eligible.begin();
                dispatchFromEligible(firstEligible, type);
            }

            // Brief sleep to avoid hammering on the workers too much to check whether
            // they're done (I hope this is correct!)

            if (!somethingFinished)
                controllerSleep();
        }
    }



    // Scans active subdomains and deals with any that say they're done.
    // Returns true if it found any completed tasks.  Now has a side effect
    // of leaving a list of just-finished tasks, in case an applicatione
    // needs that information

  bool scanForFinishedTasks(void)
  {
    DEBUGPR(10, rankTag << "Scanning for finished tasks");

    justFinished.clear();

    // Snapshot at entry (helps see if we're pinned at active==1, etc.)
    DEBUGPR(4, rankTag << "scanForFinishedTasks ENTER"  
            << ": active=" << active.size() << "/" << numWorkers
            << " activeSet=" << setToString(active)
            << ", eligibleSize=" << eligible.size()
            << " eligibleSet=" << setToString(eligible)
            << ", commActive=" << commActive
            << ", commPhase=" << commPhase);

    int completedCount = 0;
    for (auto it = active.begin(); it != active.end(); )
      {
        int s = *it;
        if (taskIsComplete(s))
          {
            completedCount++;

            // Log completion before we mutate sets (easier to interpret)
            DEBUGPR(4, rankTag << "Task COMPLETE for " << subdomainName << ' ' << s
                    << " taskType=" << subdomainObj[s].currentTaskType()
                    << " (before removal)"
                    << ": active=" << active.size() << "/" << numWorkers
                    << " activeSet=" << setToString(active)
                    << ", eligibleSize=" << eligible.size()
                    << ", commActive=" << commActive
                    << ", commPhase=" << commPhase);

            if (eventLogging)
              recordEvent(s, 0);            // State 0 means done

            justFinished.push_back(s);
            active.erase(it++);

            // Application hook (may enqueue, make eligible, etc.)
            handleCompletedTask(s);
            DEBUGPR(4, rankTag << "Removed task " << s
                    << " from active after completion"
                    << ": active=" << active.size() << "/" << numWorkers
                    << " activeSet=" << setToString(active)
                    << ", eligibleSize=" << eligible.size()
                    << " eligibleSet=" << setToString(eligible)
                    << ", commActive=" << commActive
                    << ", commPhase=" << commPhase);
          }
        else
          {
            ++it;
          }
      }

    // Summary at exit (so you can see whether anything finished this scan)
    DEBUGPR(4, rankTag << "scanForFinishedTasks EXIT"
            << ": completedCount=" << completedCount
            << ", active=" << active.size() << "/" << numWorkers
            << " activeSet=" << setToString(active)
            << ", eligibleSize=" << eligible.size()
            << ", commActive=" << commActive
            << ", commPhase=" << commPhase);

    return completedCount > 0;
  }

    // Make a task eligible to run -- instrumented for visualization

    void makeEligible(int s)
    {
        eligible.insert(s);
        if (eventLogging)
            recordEvent(s,1);   // State 1 means eligible
    }


    // Set the communication phase -- instrumented for visualization

    void setCommPhase(int p)
    {
        commPhase  = p;
        commActive = true;
        if (eventLogging)
            recordCommEvent(p + 1);   // Comm state 0 reserved for idle
    }


    // To indicate no communication is currently happening

    void clearCommActive(void)
    {
        commActive = false;
        if (eventLogging)             // In the event log, state 0 indicates
            recordCommEvent(0);       // idle communication
    }


    // Take a specified subdomain from the eligible set (indicated by an iterator),
    // Start performing a specific task on it, and put it in the active set

  void dispatchFromEligible(std::set<int>::iterator& it, int taskType)
  {
    // add by DLW Dec 7, 2025 to allow FWAPH to prevent dispatch during communication
    if (taskType < 0) return;

    int s = *it;

    // Start the work (may immediately consume a worker thread)
    startTask(s, taskType);

    // Put in active and remove from eligible
    active.insert(s);
    eligible.erase(it);

    DEBUGPR(4, rankTag << "Dispatch task type " << taskType << " to " << subdomainName
            << ' ' << s
            << ": active=" << active.size() << "/" << numWorkers
            << " activeSet=" << setToString(active)
            << ", eligibleSize=" << eligible.size()
            << " eligibleSet=" << setToString(eligible)
            << ", commActive=" << commActive
            << ", commPhase=" << commPhase);
  };

    // Start a task of a given type.  For specific applications, this will often
    // be overridden to also transfer information to the subdomain first

    virtual void startTask(int s, int taskType)
    {
        subdomainObj[s].dispatchTask(taskType);
        if (eventLogging)
            recordEvent(s,taskType + 2);  // First 2 states reserved for idle, eligible
        dispatchCount[taskType]++;
        totalDispatchCount++;
    };


    // Determine whether subdomain s has finished its task

    bool taskIsComplete(int s) { return subdomainObj[s].taskIsComplete(); };


    // Triggered when a task ends.  The default just makes the subdomain
    // eligible again and probably won't be used (unless terminating).  The
    // subdomain's task type can be used to determine what happens in more
    // interesting cases.

    virtual void handleCompletedTask(int s)
    {
        if (!terminateFlag)
            makeEligible(s);
        DEBUGPR(4, rankTag << "Generic completed task type "
                           << subdomainObj[s].currentTaskType()
                           << " for " << subdomainName << " " << s
                           << ": active=" << setToString(active)
                           << ", eligible=" << setToString(eligible));
    };


    //// The following relate to the communication cycle

    // Briefly sleep the controller so as not to be constantly hammering
    // on the worker memory or MPI requests

    void controllerSleep(void)
    {
        if (sleepMsec > 0)
            std::this_thread::sleep_for(sleepTime);
    };


    // Whether it's OK to start a communication cycle.  The default is yes,
    // unless we're terminating (although that combination of conditions
    // should not acutally occur)

    virtual bool canStartComm(void) { return (cycleCount == 0) || !terminateFlag; };


    // To start communication (pure virtual)

    virtual void startCommCycle(void) = 0;


    // Called when mainEventLoop decides to exit (handleCompletedCommunication
    // returned true). Default calls signalTermination(). Override in stepping
    // mode to exit cleanly without aborting tasks.
    virtual void onMainLoopTerminate(void) { signalTermination(); }

    // Used to signal termination.  Sets terminate flag to true, signals all
    // active tasks to abort, and empties the set of eligible subdomains

    void signalTermination(void)
    {
        terminateFlag = true;
        abortCount += active.size();
        for(auto it = active.begin(); it != active.end(); it++)
            subdomainObj[*it].signalAbort();
        eligible.clear();
        clearCommActive();
    };


    // Record a logging event for subdomain 's' entering state 'state'

    void recordEvent(int s, int state)
    {
        eventList.push_back({MPI_Wtime(), s, state});
    };


    // Record a communication event for state 'state' (communication is subdomain -1)

    void recordCommEvent(int state)
    {
        recordEvent(-1, state);
    }


    // Output the event log

    void outputEventLog(void)
    {
        // Flag all subdomains as idle

        for (int s=0; s<numSubdomains; s++)
            recordEvent(s,0);

        // Write file, round-robin

        const int logTag = 7651;             // Arbitrary unique tag for log tokens
        auto fileMode = std::ios::out;       // Rank 0 opens in normal output mode
        int  dummyBuffer;
        MPI_Status logStatus;

        if (myRank != 0)
        {
            fileMode = std::ios::app;   // Everybody else appends and waits for a token
            MPI_Recv(&dummyBuffer,
                     0,
                     MPI_INT,
                     myRank - 1,        // Should come from previous rank
                     logTag,
                     comm,
                     &logStatus);
        }

        // Open the file

        std::ofstream logStream(eventFileName,fileMode);
        if (!logStream.is_open())
            throw std::runtime_error("Could not open event log file " + eventFileName);
        logStream << std::setprecision(14);

        DEBUGPR(1, rankTag << "Writing event log file\n");

        // The first line of the log contains the number of ranks, the number
        // of subdomains per rank, the start time, and a dummy

        if (myRank == 0)
            logStream << commSize << ", "
                      << numSubdomains << ", "
                      << startWallTime << ", "
                      << numWorkers << std::endl;

        // Now output the whole log

        while (eventList.size() > 0)
        {
            auto logEntry = eventList.front();
            eventList.pop_front();
            logStream << myRank << ", "
                      << std::get<1>(logEntry) << ", "        // The subdomain is 2nd
                      << std::get<0>(logEntry) << ", "        // The time is 1st
                      << std::get<2>(logEntry) << std::endl;  // The state is last
        }

        // Close file.  If not the last rank, pass on the token

        logStream.close();

        int nextRank = myRank + 1;

        if (nextRank < commSize)
            MPI_Send(&dummyBuffer,
                     0,
                     MPI_INT,
                     nextRank,
                     logTag,
                     comm);
    };


    //// Data members

    // Control parameters (some typically set by the application, others from
    // standardized command line options)

    // DLW doesn't like a null argument ctor, but is trying to be mature.
    Config *cfgPtr = nullptr;     // whence many options come

    int numWorkers;     // Maximum number of concurrent active tasks allowed
                        // Each of these might use subthreads
    int numSubdomains; // Number of subdomains per rank (not total!)

    int sleepMsec;      // Milliseconds to sleep when event loops seem to be waiting

    int numTaskTypes;   // Number of task types

    // NEW: terminate after N wall-clock seconds since setup(); 0 disables
    double maxWallTimeSeconds;

    int maxConcurrentComms;  // Maximum number of concurrent communication
                             // operations per rank

    bool reportLocalTimeRatios;  // Report local CPU/wall time ratios at the end?

    std::vector<std::string> taskName;         // What to call each task in output
    std::string              subdomainName;    // What to call each subdomain in output

    // Working data

    std::vector<long long int> dispatchCount;
    long long int              totalDispatchCount;
    long long int              abortCount;

    std::chrono::milliseconds sleepTime;     // Same thing as a funky chrono duration

    // Subdomain tracking and data structures

    SUBDOMAINCLASS* subdomainObj; // Holds the subdomain objects

    std::set<int>  eligible;      // Eligible to run -- may eventually need
                                  //   to be replicated by task type
    std::set<int>  active;        // Subdomains that are currently running

    std::set<int>  needRedo;      // Used by {start,finish}taskOnAllsubdomains

    std::list<int> justFinished;  // Tasks just completed (some applications
                                  //   might need this)

    // Communication tracking etc.

    bool terminateFlag;       // Signals termination

    bool commActive;          // Whether communication is running right now
    int  cycleCount;          // How many communications cycles have completed
    int  commPhase;           // What (application-dependent) phase of the current
                              // communication cycle we're in (might not be used
                              // in some simple applications)

    int concurrentCommCount;  // How many communications have been issued in the current phase

    std::vector<MPI_Request> requestVector;   // MPI requests for ongoing communication
    std::vector<MPI_Status>  statusVector;    // MPI statuses of recent communications

    // Timers

    double  startWallTime;    // MPI wall-clock times
    double  endWallTime;
    clock_t startCPUTime;     // CPU times
    clock_t endCPUTime;

    // Event logging

    bool eventLogging;
    std::string eventFileName;
    std::list<std::tuple<double,int,int>> eventList;

};



// Standardized "driver" main program sequence for simple executables

template<class CONTROLLERCLASS>
int standardDriver(int argc, char** argv)
{
    // Correctly initialize MPI to run multiple threads, although only the main
    // thread will perform communication with other ranks

    int ignoreForNow = 0;
    MPI_Init_thread(&argc,&argv,MPI_THREAD_FUNNELED,&ignoreForNow);

    {
        // Create the group controller object for the application.  This
        // operation is within a { ... } block so that the controller object
        // will be destructed before we shut down MPI.

        CONTROLLERCLASS controller;

        // Give the controller object the MPI context

        controller.setupComm(MPI_COMM_WORLD);

        // Parse the command line.  This might also instruct the application
        // how to read problem data.

        CLI::App app{"groupController application"};
        Config cfg(app);
        CONTROLLERCLASS::registerOptions(cfg);
        CLI11_PARSE(app, argc, argv);

        // Print current option settings, plus some vertical space (only on rank 0).
        // The least annoying way to get the MPI rank is by asking the controller.

        if (controller.getRank() == 0)
        {
            std::cout << std::endl;
            cfg.pprint(std::cout);
            std::cout << std::endl;
        }

        // Set up the application based on the options on the command line

        controller.setup(cfg);

        // Run the controller application.  This does all the real work.

        controller.run();

        // To output results; defaults to a stub

        controller.outputResults();

        // Print performance summary

        controller.printRunStatistics();
    }

    MPI_Finalize();

    return 0;
}
