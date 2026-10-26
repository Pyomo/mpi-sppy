// pybind11 bindings for aphfw C++ library
// Exposes FWAPH for use as an mpi-sppy hub

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "FWAPH.hpp"
#include "groupController.hpp"
#include "scenario.hpp"
#include "config/config.hpp"
#include "utils/timer.hpp"
#include "CLI11.hpp"

#include <mpi.h>
#include <atomic>
#include <string>
#include <vector>
#include <unordered_map>

namespace py = pybind11;

// Convert a Python mpi4py communicator handle to MPI_Comm.
// mpi4py._handleof(comm) returns an integer (the address of the
// ompi_communicator_t for OpenMPI, or the integer handle for MPICH).
static MPI_Comm py_comm_to_mpi(py::handle py_comm)
{
    py::module_ mpi4py = py::module_::import("mpi4py.MPI");
    py::object handle = mpi4py.attr("_handleof")(py_comm);
    std::uintptr_t addr = handle.cast<std::uintptr_t>();
    return reinterpret_cast<MPI_Comm>(addr);
}


// Wrapper class that owns the FWAPH controller and manages its lifecycle
class APHFWLibrary {
public:
    APHFWLibrary() = default;
    ~APHFWLibrary() { finalize(); }

    // Initialize with an MPI communicator (from mpi4py), problem directory,
    // and a dict of config options.
    // If set_explicit_scenario_names was called before init, apply the names
    // after creating the controller but before setup.
    void init(py::handle py_comm,
              const std::string& problem_dir,
              py::dict config_dict)
    {
        comm_ = py_comm_to_mpi(py_comm);

        int rank;
        MPI_Comm_rank(comm_, &rank);

        // Set up the global timer
        timer_ = std::make_unique<GlobalTimer>(rank);
        install_global_timer(timer_.get());
        global_toc("aphfw_cpp init start");

        controller_ = std::make_unique<FWAPH>();
        controller_->setupComm(comm_);

        // Apply deferred scenario names (set before init)
        if (!deferred_scenario_names_.empty()) {
            controller_->setExplicitScenarioNames(std::move(deferred_scenario_names_));
            deferred_scenario_names_.clear();
        }

        // Build argc/argv from the config dict
        // We construct a fake command line from the dict entries
        std::vector<std::string> arg_strings;
        arg_strings.push_back("aphfw_cpp");  // program name

        // Always add problem-name
        arg_strings.push_back("--problem-name");
        arg_strings.push_back(problem_dir);

        // Add all config options from the dict
        for (auto item : config_dict) {
            std::string key = py::str(item.first);
            py::handle val = item.second;

            // Convert key to CLI flag format (underscores to dashes, prefix --)
            std::string flag = "--" + key;
            for (char& c : flag) {
                if (c == '_') c = '-';
            }

            // Handle boolean values specially
            if (py::isinstance<py::bool_>(val)) {
                if (val.cast<bool>()) {
                    arg_strings.push_back(flag);
                }
                // false bools: just omit the flag
            } else {
                arg_strings.push_back(flag);
                arg_strings.push_back(py::str(val).cast<std::string>());
            }
        }

        // Build argv
        std::vector<char*> argv;
        for (auto& s : arg_strings)
            argv.push_back(s.data());
        int argc = static_cast<int>(argv.size());

        // Parse
        CLI::App app{"FW APH (pybind11)"};
        config_ = std::make_unique<Config>(app);
        FWAPH::registerOptions(*config_);
        groupController<Scenario>::registerOptions(*config_);

        try {
            app.parse(argc, argv.data());
        } catch (const CLI::ParseError& e) {
            throw std::runtime_error("CLI parse error: " + std::string(e.what()));
        }

        // Print config on rank 0
        if (controller_->getRank() == 0) {
            config_->pprint(std::cout);
            std::cout << std::endl;
        }

        controller_->setup(*config_);
        global_toc("aphfw_cpp init complete");
        initialized_ = true;
    }

    void set_hub_mode(bool on) {
        check_initialized();
        controller_->setHubMode(on);
    }

    void set_external_terminate_flag(std::uintptr_t flag_addr) {
        check_initialized();
        auto* flag = reinterpret_cast<std::atomic<int>*>(flag_addr);
        controller_->setExternalTerminateFlag(flag);
    }

    int run_cycles(int n) {
        check_initialized();
        return controller_->runNCycles(n);
    }

    void get_all_local_nonants(py::array_t<double> buf) {
        check_initialized();
        auto info = buf.request();
        if (info.ndim != 1)
            throw std::runtime_error("get_all_local_nonants: expected 1-D array");
        controller_->getAllLocalNonants(
            static_cast<double*>(info.ptr),
            static_cast<int>(info.shape[0]));
    }

    void get_all_local_prox_x(py::array_t<double> buf) {
        check_initialized();
        auto info = buf.request();
        if (info.ndim != 1)
            throw std::runtime_error("get_all_local_prox_x: expected 1-D array");
        controller_->getAllLocalProxX(
            static_cast<double*>(info.ptr),
            static_cast<int>(info.shape[0]));
    }

    void get_all_local_xmean(py::array_t<double> buf) {
        check_initialized();
        auto info = buf.request();
        if (info.ndim != 1)
            throw std::runtime_error("get_all_local_xmean: expected 1-D array");
        controller_->getAllLocalXmean(
            static_cast<double*>(info.ptr),
            static_cast<int>(info.shape[0]));
    }

    void get_all_local_w(py::array_t<double> buf) {
        check_initialized();
        auto info = buf.request();
        if (info.ndim != 1)
            throw std::runtime_error("get_all_local_w: expected 1-D array");
        controller_->getAllLocalW(
            static_cast<double*>(info.ptr),
            static_cast<int>(info.shape[0]));
    }

    void get_all_local_xbar(py::array_t<double> buf) {
        check_initialized();
        auto info = buf.request();
        if (info.ndim != 1)
            throw std::runtime_error("get_all_local_xbar: expected 1-D array");
        controller_->getAllLocalXbar(
            static_cast<double*>(info.ptr),
            static_cast<int>(info.shape[0]));
    }

    void get_all_local_rho(py::array_t<double> buf) {
        check_initialized();
        auto info = buf.request();
        if (info.ndim != 1)
            throw std::runtime_error("get_all_local_rho: expected 1-D array");
        controller_->getAllLocalRho(
            static_cast<double*>(info.ptr),
            static_cast<int>(info.shape[0]));
    }

    void set_all_local_rho(py::array_t<double> buf) {
        check_initialized();
        auto info = buf.request();
        if (info.ndim != 1)
            throw std::runtime_error("set_all_local_rho: expected 1-D array");
        controller_->setAllLocalRho(
            static_cast<const double*>(info.ptr),
            static_cast<int>(info.shape[0]));
    }

    void get_all_local_omicron(py::array_t<double> buf) {
        check_initialized();
        auto info = buf.request();
        if (info.ndim != 1)
            throw std::runtime_error("get_all_local_omicron: expected 1-D array");
        controller_->getAllLocalOmicron(
            static_cast<double*>(info.ptr),
            static_cast<int>(info.shape[0]));
    }

    void freeze_omicron(double multiplier) {
        check_initialized();
        controller_->freezeOmicron(multiplier);
    }

    bool omicron_is_frozen() {
        check_initialized();
        return controller_->omicronIsFrozen();
    }

    void get_all_local_nonant_bounds(py::array_t<double> lb, py::array_t<double> ub) {
        check_initialized();
        auto lb_info = lb.request();
        auto ub_info = ub.request();
        if (lb_info.ndim != 1 || ub_info.ndim != 1)
            throw std::runtime_error("get_all_local_nonant_bounds: expected 1-D arrays");
        if (lb_info.shape[0] != ub_info.shape[0])
            throw std::runtime_error("get_all_local_nonant_bounds: lb/ub size mismatch");
        controller_->getAllLocalNonantBounds(
            static_cast<double*>(lb_info.ptr),
            static_cast<double*>(ub_info.ptr),
            static_cast<int>(lb_info.shape[0]));
    }

    void set_all_local_nonant_bounds(py::array_t<double> lb, py::array_t<double> ub) {
        check_initialized();
        auto lb_info = lb.request();
        auto ub_info = ub.request();
        if (lb_info.ndim != 1 || ub_info.ndim != 1)
            throw std::runtime_error("set_all_local_nonant_bounds: expected 1-D arrays");
        if (lb_info.shape[0] != ub_info.shape[0])
            throw std::runtime_error("set_all_local_nonant_bounds: lb/ub size mismatch");
        controller_->setAllLocalNonantBounds(
            static_cast<const double*>(lb_info.ptr),
            static_cast<const double*>(ub_info.ptr),
            static_cast<int>(lb_info.shape[0]));
    }

    bool is_converged() const {
        check_initialized();
        return controller_->isConverged();
    }

    int get_current_cycle() const {
        check_initialized();
        return controller_->getCurrentCycle();
    }

    std::vector<std::string> get_local_scenario_names() const {
        check_initialized();
        return controller_->getLocalScenarioNames();
    }

    int get_local_nonant_length() const {
        check_initialized();
        return controller_->getLocalNonantLength();
    }

    double get_current_lagrangian_bound() const {
        check_initialized();
        return controller_->getCurrentLagrangianBound();
    }

    double compute_final_lagrangian_bound() {
        check_initialized();
        return controller_->computeFinalLagrangianBound();
    }

    void set_explicit_scenario_names(std::vector<std::string> names) {
        if (controller_) {
            // Already initialized — apply immediately
            controller_->setExplicitScenarioNames(std::move(names));
        } else {
            // Defer until init() creates the controller
            deferred_scenario_names_ = std::move(names);
        }
    }

    void print_results() {
        check_initialized();
        controller_->printResults();
    }

    void print_run_statistics() {
        check_initialized();
        controller_->printRunStatistics();
    }

    void save_w_trace() {
        check_initialized();
        controller_->saveWTrace();
    }

    void finalize() {
        if (controller_) {
            controller_->saveWTrace();
            global_toc("aphfw_cpp finalize");
        }
        controller_.reset();
        config_.reset();
        timer_.reset();
        initialized_ = false;
    }

private:
    void check_initialized() const {
        if (!initialized_)
            throw std::runtime_error("APHFWLibrary not initialized; call init() first");
    }

    MPI_Comm comm_ = MPI_COMM_NULL;
    std::unique_ptr<FWAPH> controller_;
    std::unique_ptr<Config> config_;
    std::unique_ptr<GlobalTimer> timer_;
    std::vector<std::string> deferred_scenario_names_;
    bool initialized_ = false;
};


PYBIND11_MODULE(aphfw_cpp, m) {
    m.doc() = "C++ FW-APH algorithm for mpi-sppy integration";

    py::class_<APHFWLibrary>(m, "APHFWLibrary")
        .def(py::init<>())
        .def("init", &APHFWLibrary::init,
             py::arg("comm"), py::arg("problem_dir"), py::arg("config_dict"),
             "Initialize FWAPH with MPI comm, problem directory, and config options")
        .def("set_hub_mode", &APHFWLibrary::set_hub_mode)
        .def("set_external_terminate_flag", &APHFWLibrary::set_external_terminate_flag,
             "Set pointer to external atomic<int> terminate flag (pass ctypes.addressof)")
        .def("run_cycles", &APHFWLibrary::run_cycles,
             "Run n cycles, return number actually completed")
        .def("get_all_local_nonants", &APHFWLibrary::get_all_local_nonants,
             "Fill numpy array with all local candidate-solution values "
             "(points of K_s, for the xhat spokes and the denouement)")
        .def("get_all_local_prox_x", &APHFWLibrary::get_all_local_prox_x,
             "Fill numpy array with the proximal minimizers the algorithm "
             "iterates on (what the dynamic rho extensions want)")
        .def("get_all_local_xmean", &APHFWLibrary::get_all_local_xmean,
             "Fill numpy array with the probability-weighted mean of the "
             "scenario solutions (PH's xbar; APH's z is a different object)")
        .def("get_all_local_w", &APHFWLibrary::get_all_local_w,
             "Fill numpy array with all local W values")
        .def("get_all_local_xbar", &APHFWLibrary::get_all_local_xbar,
             "Fill numpy array with all local xbar (consensus) values")
        .def("get_all_local_rho", &APHFWLibrary::get_all_local_rho,
             "Fill numpy array with all local rho values")
        .def("set_all_local_rho", &APHFWLibrary::set_all_local_rho,
             "Update rho for all local scenarios from numpy array")
        .def("get_all_local_omicron", &APHFWLibrary::get_all_local_omicron,
             "Fill numpy array with all local omicron values")
        .def("freeze_omicron", &APHFWLibrary::freeze_omicron,
             "One-shot: snapshot omicron = multiplier * current rho. Throws if already frozen.")
        .def("omicron_is_frozen", &APHFWLibrary::omicron_is_frozen,
             "Returns True if omicron has been frozen on every local scenario.")
        .def("get_all_local_nonant_bounds", &APHFWLibrary::get_all_local_nonant_bounds,
             "Fill numpy arrays with all local nonant lower and upper bounds")
        .def("set_all_local_nonant_bounds", &APHFWLibrary::set_all_local_nonant_bounds,
             "Update nonant bounds for all local scenarios from numpy arrays")
        .def("is_converged", &APHFWLibrary::is_converged)
        .def("get_current_cycle", &APHFWLibrary::get_current_cycle)
        .def("get_local_scenario_names", &APHFWLibrary::get_local_scenario_names)
        .def("get_local_nonant_length", &APHFWLibrary::get_local_nonant_length)
        .def("get_current_lagrangian_bound", &APHFWLibrary::get_current_lagrangian_bound)
        .def("compute_final_lagrangian_bound", &APHFWLibrary::compute_final_lagrangian_bound,
             "Run a Lagrangian solve on every local scenario at the current W "
             "and return the reduced expected bound. Collective over the "
             "communicator passed to init(): all ranks must call it.")
        .def("set_explicit_scenario_names", &APHFWLibrary::set_explicit_scenario_names)
        .def("print_results", &APHFWLibrary::print_results)
        .def("print_run_statistics", &APHFWLibrary::print_run_statistics)
        .def("save_w_trace", &APHFWLibrary::save_w_trace)
        .def("finalize", &APHFWLibrary::finalize);
}
