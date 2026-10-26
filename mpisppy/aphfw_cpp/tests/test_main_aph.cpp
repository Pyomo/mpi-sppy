///////////////////////////////////////////////////////////////////////////////
// mpi-sppy: MPI-based Stochastic Programming in PYthon
//
// Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
// Sustainable Energy, LLC, The Regents of the University of California, et al.
// All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
// full copyright and license information.
///////////////////////////////////////////////////////////////////////////////
#include <gtest/gtest.h>
#include <mpi.h>

#include <string>
#include <vector>
#include <iostream>
#include <sstream>
#include <fstream>
#include <cstdio>

#include "appRun.hpp"  // runFWAPHWithArgs

// Extract a floating-point value from the first occurrence of
// "<prefix> <value>" in haystack.  Returns false if the prefix is not found
// or parsing fails.
static bool extractBound(const std::string& haystack,
                         const std::string& prefix,
                         double& value)
{
    auto pos = haystack.find(prefix);
    if (pos == std::string::npos) return false;
    pos += prefix.size();
    while (pos < haystack.size() && haystack[pos] == ' ') ++pos;
    std::istringstream iss(haystack.substr(pos));
    return static_cast<bool>(iss >> value);
}

// Build a baseline arg list. Individual tests may extend it (extraArgs appended).
static std::vector<std::string> buildArgStrings(const std::vector<std::string>& extraArgs = {},
                                                const std::string& tracePrefix = "SolveTimesTrace_test")
{
    std::vector<std::string> argStrings = {
        "aphfw_main_test",
        "--aph-max-cycles", "8",
        "--problem-name", "smallFWAPH/mps_dir",
        "--nl-json", "smallFWAPH/nl.json",
        "--grad-json", "smallFWAPH/grad.json",
        "--iter0-json", "smallFWAPH/iter0.json",
        "--FW-tmax", "4",
        "--FW-tau", "0.01",
        "--primal-term-thresh", "0.001",
        "--dual-term-thresh", "0.001",
        "--solver-log-file", "solver.log",
        "--display-progress",
        "--verbosity", "1",
        "--solve-times-trace-prefix", tracePrefix
    };
    for (const auto& s : extraArgs) argStrings.push_back(s);
    return argStrings;
}

// Same shape as buildArgStrings, but for smallQuadFWAPH: the same set-cover
// problem with a diagonal quadratic added to the objective, so the original
// problem is nonlinear.
static std::vector<std::string> buildQuadArgStrings(const std::vector<std::string>& extraArgs = {},
                                                    const std::string& tracePrefix = "SolveTimesTrace_quad_test")
{
    std::vector<std::string> argStrings = {
        "aphfw_main_test",
        "--aph-max-cycles", "8",
        "--problem-name", "smallQuadFWAPH/mps_dir",
        "--nl-json", "smallQuadFWAPH/nl.json",
        "--grad-json", "smallQuadFWAPH/grad.json",
        "--iter0-json", "smallQuadFWAPH/iter0.json",
        "--FW-tmax", "4",
        "--FW-tau", "0.01",
        "--primal-term-thresh", "0.001",
        "--dual-term-thresh", "0.001",
        "--solver-log-file", "solver.log",
        "--verbosity", "1",
        "--solve-times-trace-prefix", tracePrefix
    };
    for (const auto& s : extraArgs) argStrings.push_back(s);
    return argStrings;
}

// global_toc emits this when startup ends with a synchronous FW step.
static bool tookIter0FWStep(const std::string& combined)
{
    return combined.find("Begin Iter0 FW step") != std::string::npos;
}

// Run runFWAPHWithArgs with the given args, returning rc, captured stdout, captured stderr.
struct RunResult {
    int rc;
    std::string out;
    std::string err;
};
// Holds both captures for one runFWAPHWithArgs call and releases whatever is
// still armed when it goes out of scope.  Without that, a throw out of the call
// under test leaves GoogleTest capturing: the exception's message is swallowed,
// so is every later test's output, and the next test aborts inside
// CaptureStream with no diagnostic at all -- the whole run reads as a bare
// SIGABRT after the first "[ RUN ]" line.  Running the binary from a directory
// that does not hold the test data is enough to trigger it, since the data
// paths are relative to ctest's WORKING_DIRECTORY.  The single-stream version
// of this is CaptureGuard in test_fw_export.cpp.
struct BothStreamsCapture {
    bool armed = true;
    BothStreamsCapture() {
        testing::internal::CaptureStdout();
        testing::internal::CaptureStderr();
    }
    void release(std::string& out, std::string& err) {
        armed = false;
        out = testing::internal::GetCapturedStdout();
        err = testing::internal::GetCapturedStderr();
    }
    ~BothStreamsCapture() {
        if (armed) {
            (void) testing::internal::GetCapturedStdout();
            (void) testing::internal::GetCapturedStderr();
        }
    }
};

static RunResult runWithArgs(std::vector<std::string>& argStrings)
{
    std::vector<char*> argvVec;
    argvVec.reserve(argStrings.size());
    for (auto& s : argStrings) argvVec.push_back(s.data());
    int argc = static_cast<int>(argvVec.size());
    char** argv = argvVec.data();

    std::string out;
    std::string err;
    BothStreamsCapture cap;
    int rc = runFWAPHWithArgs(argc, argv, MPI_COMM_WORLD);
    cap.release(out, err);
    return {rc, out, err};
}

TEST(FWAPH_Main, RunsWithSmallExample)
{
    auto args = buildArgStrings();
    auto r = runWithArgs(args);

    // -----------------------------
    // Basic success check
    // -----------------------------
    ASSERT_EQ(r.rc, 0) << "runFWAPHWithArgs did not return success";

    // -----------------------------
    // Parse and compare Lagrangian bounds
    // -----------------------------
    const std::string combined = r.out + "\n" + r.err;
    double iter0Bound = 0.0, termBound = 0.0;
    bool foundIter0 = extractBound(combined, "Iter0 Lagrangian bound is", iter0Bound);
    bool foundTerm  = extractBound(combined, "Termination Lagrangian bound is", termBound);

    ASSERT_TRUE(foundIter0)
        << "Did not find 'Iter0 Lagrangian bound is' in output.\n\n"
        << "----- Captured stdout -----\n" << r.out << "\n"
        << "----- Captured stderr -----\n" << r.err << "\n";

    ASSERT_TRUE(foundTerm)
        << "Did not find 'Termination Lagrangian bound is' in output.\n\n"
        << "----- Captured stdout -----\n" << r.out << "\n"
        << "----- Captured stderr -----\n" << r.err << "\n";

    ASSERT_GT(termBound, iter0Bound)
        << "Expected Termination bound (" << termBound
        << ") > Iter0 bound (" << iter0Bound << ").\n\n"
        << "----- Captured stdout -----\n" << r.out << "\n"
        << "----- Captured stderr -----\n" << r.err << "\n";

    // -----------------------------
    // Verify SolveTimesTrace CSV
    // -----------------------------
    // This test runs as a single MPI rank, so the trace is always rank0.
    const std::string csvFile = "SolveTimesTrace_test_rank0.csv";

    std::ifstream csv(csvFile);
    ASSERT_TRUE(csv.is_open()) << "SolveTimesTrace CSV not found: " << csvFile;

    // Check header
    std::string header;
    ASSERT_TRUE(std::getline(csv, header)) << "CSV is empty";
    EXPECT_EQ(header, "wall_time,dispatch_cycle,scenario,fw_total_time,nl_time,milp_time,milp_solves");

    // Parse data rows and check fw_total_time >= nl_time + milp_time
    int rowCount = 0;
    std::string line;
    while (std::getline(csv, line)) {
        if (line.empty()) continue;
        std::istringstream ss(line);
        std::string tok;
        std::vector<std::string> cols;
        while (std::getline(ss, tok, ',')) cols.push_back(tok);
        ASSERT_EQ(cols.size(), 7u) << "Wrong column count in row: " << line;

        double fw_total = std::stod(cols[3]);
        double nl_time  = std::stod(cols[4]);
        double milp_time = std::stod(cols[5]);
        EXPECT_GE(fw_total, nl_time + milp_time)
            << "fw_total_time < nl_time + milp_time in row: " << line;
        ++rowCount;
    }
    EXPECT_GT(rowCount, 0) << "SolveTimesTrace CSV has no data rows";

    csv.close();
    std::remove(csvFile.c_str());  // clean up
}

// --omicron-all freezes omicron at setup with a uniform scalar. With 1.0 the
// weighted-u-norm kernel is mathematically identical to the unweighted one;
// the test exercises the CLI plumbing, the setup-time freeze path in
// ProblemWithSolvers, and the kernel's use of the omicron vector.
TEST(FWAPH_Main, OmicronAllOverrideRunsSuccessfully)
{
    auto args = buildArgStrings({"--omicron-all", "1.0"},
                                "SolveTimesTrace_omicron_all_test");
    auto r = runWithArgs(args);

    ASSERT_EQ(r.rc, 0) << "runFWAPHWithArgs with --omicron-all 1.0 failed\n"
                       << "stdout:\n" << r.out << "\nstderr:\n" << r.err;

    const std::string combined = r.out + "\n" + r.err;
    double iter0Bound = 0.0, termBound = 0.0;
    ASSERT_TRUE(extractBound(combined, "Iter0 Lagrangian bound is", iter0Bound));
    ASSERT_TRUE(extractBound(combined, "Termination Lagrangian bound is", termBound));
    ASSERT_GT(termBound, iter0Bound)
        << "Expected improvement in bound with --omicron-all 1.0.\n"
        << "stdout:\n" << r.out << "\nstderr:\n" << r.err;

    std::remove("SolveTimesTrace_omicron_all_test_rank0.csv");
}

// --omicron-all with a value exceeding --omicron-max (default 1e6) should
// cause setOmicron to throw during setup. runFWAPHWithArgs does not catch
// exceptions, so the test wraps the call and verifies the throw.
// Guards the setOmicron bounds check.
TEST(FWAPH_Main, OmicronBoundsViolationThrows)
{
    auto argStrings = buildArgStrings({"--omicron-all", "1e10"},
                                      "SolveTimesTrace_omicron_bad_test");
    std::vector<char*> argvVec;
    argvVec.reserve(argStrings.size());
    for (auto& s : argStrings) argvVec.push_back(s.data());
    int argc = static_cast<int>(argvVec.size());
    char** argv = argvVec.data();

    bool threw = false;
    std::string what;
    try {
        (void) runFWAPHWithArgs(argc, argv, MPI_COMM_WORLD);
    } catch (const std::exception& e) {
        threw = true;
        what = e.what();
    }
    EXPECT_TRUE(threw) << "Expected setOmicron to throw with --omicron-all 1e10 (>1e6 max).";
    EXPECT_NE(what.find("omicron"), std::string::npos)
        << "Exception message should mention omicron. Got: " << what;

    std::remove("SolveTimesTrace_omicron_bad_test_rank0.csv");
}

// --------------------------------------------------------------------------
// Startup FW step (--iter0-fw-step).
//
// The iter-0 Lagrangian minimizes over K_i rather than conv K_i. When the
// original objective is nonlinear that means y0 = 0 is not a subgradient at
// xbase, so startup takes one synchronous FW step to build a valid (x0,y0).
// With a linear objective the pair is already valid and the step is skipped.
// --------------------------------------------------------------------------

// smallFWAPH has a linear objective, so 'auto' must not take the step.
TEST(FWAPH_Main, Iter0FWStepAutoSkippedForLinearObjective)
{
    auto args = buildArgStrings({}, "SolveTimesTrace_fwstep_linear_test");
    auto r = runWithArgs(args);

    ASSERT_EQ(r.rc, 0) << "stdout:\n" << r.out << "\nstderr:\n" << r.err;

    const std::string combined = r.out + "\n" + r.err;
    EXPECT_FALSE(tookIter0FWStep(combined))
        << "smallFWAPH is linear, so --iter0-fw-step auto should skip the step.\n"
        << "stdout:\n" << r.out << "\nstderr:\n" << r.err;

    std::remove("SolveTimesTrace_fwstep_linear_test_rank0.csv");
}

// Forcing the step on for a linear problem is legal and must still converge.
TEST(FWAPH_Main, Iter0FWStepForcedOnForLinearObjective)
{
    auto args = buildArgStrings({"--iter0-fw-step", "on"},
                                "SolveTimesTrace_fwstep_on_test");
    auto r = runWithArgs(args);

    ASSERT_EQ(r.rc, 0) << "stdout:\n" << r.out << "\nstderr:\n" << r.err;

    const std::string combined = r.out + "\n" + r.err;
    EXPECT_TRUE(tookIter0FWStep(combined))
        << "--iter0-fw-step on should take the step regardless of objective type.\n"
        << "stdout:\n" << r.out << "\nstderr:\n" << r.err;

    double iter0Bound = 0.0, termBound = 0.0;
    ASSERT_TRUE(extractBound(combined, "Iter0 Lagrangian bound is", iter0Bound));
    ASSERT_TRUE(extractBound(combined, "Termination Lagrangian bound is", termBound));
    EXPECT_GT(termBound, iter0Bound);

    std::remove("SolveTimesTrace_fwstep_on_test_rank0.csv");
}

// smallQuadFWAPH carries a diagonal quadratic objective, so 'auto' must detect
// it and take the step.
TEST(FWAPH_Main, Iter0FWStepAutoTakenForQuadraticObjective)
{
    auto args = buildQuadArgStrings();
    auto r = runWithArgs(args);

    ASSERT_EQ(r.rc, 0) << "stdout:\n" << r.out << "\nstderr:\n" << r.err;

    const std::string combined = r.out + "\n" + r.err;
    EXPECT_TRUE(tookIter0FWStep(combined))
        << "smallQuadFWAPH has a quadratic objective, so auto should take the step.\n"
        << "stdout:\n" << r.out << "\nstderr:\n" << r.err;

    double iter0Bound = 0.0, termBound = 0.0;
    ASSERT_TRUE(extractBound(combined, "Iter0 Lagrangian bound is", iter0Bound));
    ASSERT_TRUE(extractBound(combined, "Termination Lagrangian bound is", termBound));
    EXPECT_GT(termBound, iter0Bound);

    std::remove("SolveTimesTrace_quad_test_rank0.csv");
}

// The escape hatch: 'off' suppresses the step even for a nonlinear objective.
TEST(FWAPH_Main, Iter0FWStepOffForQuadraticObjective)
{
    auto args = buildQuadArgStrings({"--iter0-fw-step", "off"},
                                    "SolveTimesTrace_quad_off_test");
    auto r = runWithArgs(args);

    ASSERT_EQ(r.rc, 0) << "stdout:\n" << r.out << "\nstderr:\n" << r.err;

    const std::string combined = r.out + "\n" + r.err;
    EXPECT_FALSE(tookIter0FWStep(combined))
        << "--iter0-fw-step off should suppress the step.\n"
        << "stdout:\n" << r.out << "\nstderr:\n" << r.err;

    std::remove("SolveTimesTrace_quad_off_test_rank0.csv");
}

TEST(FWAPH_Main, Iter0FWStepBadValueThrows)
{
    auto argStrings = buildArgStrings({"--iter0-fw-step", "bogus"},
                                      "SolveTimesTrace_fwstep_bad_test");
    std::vector<char*> argvVec;
    argvVec.reserve(argStrings.size());
    for (auto& s : argStrings) argvVec.push_back(s.data());
    int argc = static_cast<int>(argvVec.size());
    char** argv = argvVec.data();

    bool threw = false;
    std::string what;
    try {
        (void) runFWAPHWithArgs(argc, argv, MPI_COMM_WORLD);
    } catch (const std::exception& e) {
        threw = true;
        what = e.what();
    }
    EXPECT_TRUE(threw) << "Expected a throw for --iter0-fw-step bogus.";
    EXPECT_NE(what.find("iter0-fw-step"), std::string::npos)
        << "Exception message should name the option. Got: " << what;

    std::remove("SolveTimesTrace_fwstep_bad_test_rank0.csv");
}

// --iter0-w-multiplier m starts from W = m * rho * (xbase - z0) instead of
// W = 0. The value must be accepted (including 0.0, which is the default and
// which a PositiveNumber check would reject) and the run must still converge.
TEST(FWAPH_Main, Iter0WMultiplierRunsSuccessfully)
{
    for (const char* m : {"0.0", "1.0"}) {
        auto args = buildArgStrings({"--iter0-w-multiplier", m},
                                    "SolveTimesTrace_w0_test");
        auto r = runWithArgs(args);

        ASSERT_EQ(r.rc, 0) << "--iter0-w-multiplier " << m << " failed\n"
                           << "stdout:\n" << r.out << "\nstderr:\n" << r.err;

        const std::string combined = r.out + "\n" + r.err;
        double iter0Bound = 0.0, termBound = 0.0;
        ASSERT_TRUE(extractBound(combined, "Iter0 Lagrangian bound is", iter0Bound));
        ASSERT_TRUE(extractBound(combined, "Termination Lagrangian bound is", termBound));
        EXPECT_GT(termBound, iter0Bound)
            << "Expected improvement with --iter0-w-multiplier " << m << ".\n"
            << "stdout:\n" << r.out << "\nstderr:\n" << r.err;

        std::remove("SolveTimesTrace_w0_test_rank0.csv");
    }
}

// -----------------------------------------------------------------------
// Custom main that does one-shot MPI init/finalize for all tests.
// -----------------------------------------------------------------------
int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    ::testing::InitGoogleTest(&argc, argv);
    int rc = RUN_ALL_TESTS();
    MPI_Finalize();
    return rc;
}
