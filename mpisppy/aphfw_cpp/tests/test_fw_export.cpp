// Guards for the FW export split (design/fw-export-split.md).
//
// fw.hpp now keeps two answers rather than one: finalNonantSolution is always
// the proximal minimizer the algorithm iterates on, and finalVertexSolution is
// a point of K_s for the candidate-solution consumers.  These tests drive
// FWAPH a cycle at a time and read both through the accessors the mpi-sppy
// bindings use, so they see the split at every exit rather than only at the
// end of a run.
//
// smallFWAPH is the fixture: two set-cover scenarios whose three variables are
// all binary and all nonants, so "a point of K_s" is checkable by looking at
// the exported values themselves.

#include <gtest/gtest.h>
#include <mpi.h>

#include <cmath>
#include <string>
#include <vector>

#include "FWAPH.hpp"
#include "groupController.hpp"
#include "scenario.hpp"
#include "config/config.hpp"   // also pulls in CLI11
#include "utils/timer.hpp"

namespace {

constexpr double kIntTol = 1e-6;

// Releases the gtest stdout capture even when the run throws, so a test that
// expects a throw does not leave the capture armed for the next one.
struct CaptureGuard {
    bool armed = true;
    CaptureGuard() { testing::internal::CaptureStdout(); }
    std::string release() {
        armed = false;
        return testing::internal::GetCapturedStdout();
    }
    ~CaptureGuard() { if (armed) (void) testing::internal::GetCapturedStdout(); }
};

// What one stepped run recorded, one entry per completed cycle.
struct SteppedRun {
    int nonantLen = 0;                            // local nonants, all scenarios
    int numScens = 0;
    std::vector<std::vector<double>> exported;    // getAllLocalNonants
    std::vector<std::vector<double>> proxX;       // getAllLocalProxX
    // Subproblem objective at the exported point, per scenario, per cycle, as
    // measured when the export was chosen.  NaN in a cycle a scenario did not
    // take an FW step in (iteration 0 goes down the Lagrangian path).
    std::vector<std::vector<double>> exportSubObj;
    std::string out;                              // captured stdout
};

// Run smallFWAPH one cycle at a time, reading the export and the iterate after
// each.  Everything the controller needs stays alive for the whole loop, which
// is why the stepping happens here rather than in the test bodies.
//
// extraArgs carries the FW settings; CLI11 rejects a repeated option, so the
// list below holds only what no caller overrides.
SteppedRun runStepped(const std::vector<std::string>& extraArgs, int maxCycles)
{
    std::vector<std::string> argStrings = {
        "aphfw_export_test",
        "--problem-name", "smallFWAPH/mps_dir",
        "--nl-json",   "smallFWAPH/nl.json",
        "--grad-json", "smallFWAPH/grad.json",
        "--iter0-json","smallFWAPH/iter0.json",
        "--aph-max-cycles", std::to_string(maxCycles),
        "--primal-term-thresh", "0.001",
        "--dual-term-thresh", "0.001",
        // Block synchronism so the two export policies produce the same
        // dispatch order and the runs stay comparable cycle by cycle.
        "--block-sync",
        "--solver-log-file", "solver.log",
        // 1, not 0: the exit-path tests count the "FW terminates on" lines,
        // which are printed only from verbosity 1 up.
        "--verbosity", "1",
    };
    for (const auto& s : extraArgs) argStrings.push_back(s);

    std::vector<char*> argvVec;
    argvVec.reserve(argStrings.size());
    for (auto& s : argStrings) argvVec.push_back(s.data());
    int argc = static_cast<int>(argvVec.size());
    char** argv = argvVec.data();

    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    // Function-local so the installed pointer stays valid after we return;
    // global_toc fires from inside the controller throughout the run.
    static GlobalTimer gt(rank);
    install_global_timer(&gt);

    SteppedRun res;
    CaptureGuard capture;

    FWAPH controller;
    controller.setupComm(MPI_COMM_WORLD);

    CLI::App app{"FW APH export test"};
    Config config(app);
    FWAPH::registerOptions(config);
    groupController<Scenario>::registerOptions(config);
    app.parse(argc, argv);

    controller.setup(config);

    res.nonantLen = controller.getLocalNonantLength();
    res.numScens  = static_cast<int>(controller.getLocalScenarioNames().size());

    for (int c = 0; c < maxCycles; ++c) {
        if (controller.runNCycles(1) <= 0) break;

        std::vector<double> e(res.nonantLen), p(res.nonantLen);
        controller.getAllLocalNonants(e.data(), res.nonantLen);
        controller.getAllLocalProxX(p.data(), res.nonantLen);

        // The subproblem objective at the exported point, scenario by scenario.
        // This is read from the FW object rather than recomputed here: it is
        // the value in force when the export was chosen, and by the time the
        // cycle ends W and z have moved, so nothing computed out here would be
        // the quantity best-column actually minimizes.
        std::vector<double> objs;
        for (int s = 0; s < res.numScens; ++s) {
            objs.push_back(controller.scenario(s).getLastExportObjVal());
        }

        res.exported.push_back(std::move(e));
        res.proxX.push_back(std::move(p));
        res.exportSubObj.push_back(std::move(objs));
    }

    res.out = capture.release();
    return res;
}

bool isIntegral(double v) { return std::abs(v - std::round(v)) <= kIntTol; }

int countOccurrences(const std::string& hay, const std::string& needle)
{
    int n = 0;
    for (size_t p = hay.find(needle); p != std::string::npos;
         p = hay.find(needle, p + needle.size())) ++n;
    return n;
}

}  // namespace

// --------------------------------------------------------------------------
// The export is a point of K_s at the tMax exit.
//
// With --FW-tmax 1 the inner loop leaves by the tMax exit as soon as one MILP
// solve has happened, which is the exit that used to hand the spokes the
// proximal minimizer.  Every variable here is binary, so membership in K_s is
// visible as integrality.  The two solve-failure exits (numerical bailout,
// cutoff infeasible) share the same setVertexExport(nullptr) call as tMax and
// cannot be provoked on demand, so tMax is what stands for them.
// --------------------------------------------------------------------------
TEST(FWExport, ExportIsIntegralAtTMaxExit)
{
    auto r = runStepped({"--FW-tmax", "1", "--FW-tau", "0.01"}, 8);

    ASSERT_GT(r.exported.size(), 0u) << "no cycles completed\n" << r.out;

    // The run has to actually reach the exit under test, or the check is empty.
    EXPECT_GT(countOccurrences(r.out, "FW terminates on tMax"), 0)
        << "--FW-tmax 1 was expected to produce tMax exits.\n" << r.out;

    for (size_t c = 0; c < r.exported.size(); ++c) {
        for (int i = 0; i < r.nonantLen; ++i) {
            EXPECT_TRUE(isIntegral(r.exported[c][i]))
                << "cycle " << (c + 1) << " exported nonant " << i << " = "
                << r.exported[c][i] << " is not integral, so it is not a point "
                   "of K_s.\n" << r.out;
        }
    }
}

// Same claim at the tau exit, which reaches setVertexExport with a fresh
// vertex in hand rather than a null one.  A loose tau stops the inner loop on
// its gap test on the first pass.
TEST(FWExport, ExportIsIntegralAtTauExit)
{
    auto r = runStepped({"--FW-tmax", "4", "--FW-tau", "1e6"}, 8);

    ASSERT_GT(r.exported.size(), 0u) << "no cycles completed\n" << r.out;
    EXPECT_GT(countOccurrences(r.out, "FW terminates on tau"), 0)
        << "--FW-tau 1e6 was expected to produce tau exits.\n" << r.out;

    for (size_t c = 0; c < r.exported.size(); ++c) {
        for (int i = 0; i < r.nonantLen; ++i) {
            EXPECT_TRUE(isIntegral(r.exported[c][i]))
                << "cycle " << (c + 1) << " exported nonant " << i << " = "
                << r.exported[c][i] << " is not integral.\n" << r.out;
        }
    }
}

// --------------------------------------------------------------------------
// The negative control: what the shipped build exported.
//
// The prox policy puts the proximal minimizer in finalVertexSolution, which is
// what every exit but tau used to hand the spokes.  Under --FW-tmax 1 that
// point is fractional, so the two tests above would fail against it -- which
// is the regression they exist to catch.
// --------------------------------------------------------------------------
TEST(FWExport, ProxPolicyReproducesTheOldFractionalExport)
{
    auto r = runStepped({"--FW-tmax", "1", "--FW-tau", "0.01",
                         "--FW-export-policy", "prox"}, 8);

    ASSERT_GT(r.exported.size(), 0u) << "no cycles completed\n" << r.out;

    bool sawFractional = false;
    for (size_t c = 0; c < r.exported.size(); ++c) {
        for (int i = 0; i < r.nonantLen; ++i) {
            EXPECT_DOUBLE_EQ(r.exported[c][i], r.proxX[c][i])
                << "cycle " << (c + 1) << " nonant " << i
                << ": the prox policy must export the iterate verbatim.";
            if (!isIntegral(r.exported[c][i])) sawFractional = true;
        }
    }

    EXPECT_TRUE(sawFractional)
        << "the old export was integral throughout, so ExportIsIntegralAtTMaxExit "
           "would pass either way on this fixture.\n" << r.out;
}

// --------------------------------------------------------------------------
// The two accessors are genuinely two things.
//
// This is what keeps the integrality tests from passing vacuously: if the
// export were still the proximal minimizer, the values would agree everywhere.
// --FW-tmax 4 lets the inner loop build a convex combination, so the iterate
// leaves the vertex set.
// --------------------------------------------------------------------------
TEST(FWExport, ProxXDiffersFromTheExport)
{
    auto r = runStepped({"--FW-tmax", "4", "--FW-tau", "0.01"}, 8);

    ASSERT_GT(r.exported.size(), 0u) << "no cycles completed\n" << r.out;

    bool sawDifference = false;
    bool sawFractionalProx = false;
    for (size_t c = 0; c < r.exported.size(); ++c) {
        for (int i = 0; i < r.nonantLen; ++i) {
            if (std::abs(r.exported[c][i] - r.proxX[c][i]) > kIntTol)
                sawDifference = true;
            if (!isIntegral(r.proxX[c][i]))
                sawFractionalProx = true;
        }
    }

    EXPECT_TRUE(sawDifference)
        << "getAllLocalNonants and getAllLocalProxX agreed in every cycle, so "
           "the integrality tests prove nothing.\n" << r.out;
    EXPECT_TRUE(sawFractionalProx)
        << "the proximal minimizer was integral in every cycle, so this "
           "fixture does not exercise the split.\n" << r.out;
}

// --------------------------------------------------------------------------
// best-column does not cost anything at the tau exit.
//
// The tau exit is the one place the shipped build already exported a vertex
// (the newest one), so it is the only place the export changes for the xhat
// spokes.  last-vertex reproduces the old answer.  xExport feeds no algorithm
// state, so the two runs follow the same trajectory and the comparison is
// cycle by cycle on identical iterates.
// --------------------------------------------------------------------------
TEST(FWExport, BestColumnNoWorseThanLastVertexAtTauExit)
{
    auto best = runStepped({"--FW-tmax", "4", "--FW-tau", "1e6",
                            "--FW-export-policy", "best-column"}, 8);
    auto last = runStepped({"--FW-tmax", "4", "--FW-tau", "1e6",
                            "--FW-export-policy", "last-vertex"}, 8);

    ASSERT_GT(best.exported.size(), 0u) << "no cycles completed\n" << best.out;
    ASSERT_EQ(best.exported.size(), last.exported.size())
        << "the two policies produced different cycle counts, so xExport is "
           "feeding back into the algorithm.";

    // Same trajectory: the iterate must be untouched by the export policy.
    for (size_t c = 0; c < best.proxX.size(); ++c) {
        for (int i = 0; i < best.nonantLen; ++i) {
            EXPECT_NEAR(best.proxX[c][i], last.proxX[c][i], 1e-9)
                << "cycle " << (c + 1) << " nonant " << i
                << ": the export policy changed the algorithm's iterate.";
        }
    }

    // The comparison below is only worth making if the two policies actually
    // pick different columns somewhere; on this fixture they part company in
    // two of the eight cycles.
    int differingCycles = 0;
    for (size_t c = 0; c < best.exported.size(); ++c) {
        for (int i = 0; i < best.nonantLen; ++i) {
            if (std::abs(best.exported[c][i] - last.exported[c][i]) > kIntTol) {
                ++differingCycles;
                break;
            }
        }
    }
    EXPECT_GT(differingCycles, 0)
        << "best-column and last-vertex exported the same point in every "
           "cycle, so the objective comparison proves nothing.";

    // The comparison itself, on the quantity best-column minimizes and at the
    // moment it minimizes it.  Both runs follow the same trajectory (asserted
    // above), so cycle c scenario s faced the same W, z and rho in each.
    //
    // last-vertex's pick is always inside best-column's candidate set, so this
    // holds by construction; what the assertion guards is that the two stay
    // wired to the same candidate set and the same objective.
    int comparedCycles = 0;
    for (size_t c = 0; c < best.exportSubObj.size(); ++c) {
        for (size_t s = 0; s < best.exportSubObj[c].size(); ++s) {
            const double b = best.exportSubObj[c][s];
            const double l = last.exportSubObj[c][s];
            // NaN where the scenario took no FW step that cycle; that happens
            // identically in both runs, so skipping is symmetric.
            if (std::isnan(b) || std::isnan(l)) continue;
            ++comparedCycles;
            EXPECT_LE(b, l + 1e-9)
                << "cycle " << (c + 1) << " scenario " << s
                << ": best-column exported a point with subproblem objective "
                << b << ", worse than last-vertex's " << l << ".";
        }
    }
    EXPECT_GT(comparedCycles, 0)
        << "every cycle was NaN on both sides, so no comparison was made.\n"
        << best.out;
}

// An unknown policy name has to be rejected rather than silently defaulted.
TEST(FWExport, BadExportPolicyThrows)
{
    bool threw = false;
    std::string what;
    try {
        (void) runStepped({"--FW-tmax", "4", "--FW-tau", "0.01",
                           "--FW-export-policy", "bogus"}, 2);
    } catch (const std::exception& e) {
        threw = true;
        what = e.what();
    }
    EXPECT_TRUE(threw) << "expected a throw for --FW-export-policy bogus";
    EXPECT_NE(what.find("FW-export-policy"), std::string::npos)
        << "the message should name the option; got: " << what;
}
