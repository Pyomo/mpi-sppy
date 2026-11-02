///////////////////////////////////////////////////////////////////////////////
// mpi-sppy: MPI-based Stochastic Programming in PYthon
//
// Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
// Sustainable Energy, LLC, The Regents of the University of California, et al.
// All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
// full copyright and license information.
///////////////////////////////////////////////////////////////////////////////
// Guards for --no-fw, where each subproblem is one proximal solve by
// ProblemWithSolvers' proxSolver instead of Frank-Wolfe.
//
// Fixtures: scenario 0 of smallFWAPH (x binary) and of smallLPFWAPH (the
// same model with 0 <= x <= 1), min x1 + 4 x2 + 9 x3 s.t. x1 + x2 + x3 >= 1,
// all three variables nonants.  The tests use W = 0 and z = (0, 0, 1), so the
// proximal objective is f(x) + (rho/2)||x - z||^2.
//
// Binary: the three single-variable covers cost x1: 1 + rho, x2: 4 + rho,
// x3: 9, so x1 wins for rho = 1 and x3 for rho = 20; with x1 fixed to 0 and
// rho = 1, x2 wins.
//
// Continuous (KKT, mu the multiplier of the cover row): x_i = (mu - c_i)/rho
// clipped to [0, 1], with c = (1, 4, 9 - rho) after folding in z.  rho = 1
// gives (1, 0, 0); rho = 20 gives mu = 14/3 and x = (11, 2, 47)/60; with x1
// fixed to 0 and rho = 1, (0, 1, 0).  HiGHS can do these (continuous QPs),
// so the continuous fixture is the one every backend runs.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "ProblemWithSolvers.hpp"
#include "config/config.hpp"   // also pulls in CLI11

namespace {

const std::string kBinScen = "smallFWAPH/mps_dir/scen0";
const std::string kLPScen  = "smallLPFWAPH/mps_dir/scen0";

// CLI11 keeps a pointer to the app, so the app and the Config live together.
struct ParsedConfig {
    CLI::App app{"no-fw test"};
    std::unique_ptr<Config> cfg;
    explicit ParsedConfig(const std::vector<std::string>& extraArgs) {
        cfg = std::make_unique<Config>(app);
        ProblemWithSolvers::registerOptions(*cfg);
        std::vector<std::string> args = {
            "no_fw_test",
            "--iter0-json", "smallFWAPH/iter0.json",
            "--solver-log-file", "solver.log",
        };
        for (const auto& a : extraArgs) args.push_back(a);
        std::vector<char*> argv;
        for (auto& a : args) argv.push_back(a.data());
        app.parse(static_cast<int>(argv.size()), argv.data());
    }
};

const std::vector<std::string> kNoFWArgs = {"--no-fw", "--prox-json", "smallFWAPH/prox.json"};
const std::vector<std::string> kFWArgs = {"--nl-json", "smallFWAPH/nl.json",
                                          "--grad-json", "smallFWAPH/grad.json"};

// One scenario's solver objects, with W and z owned here as FWAPH owns them.
struct Subproblem {
    ParsedConfig pc;
    std::vector<double> W, z;
    ProblemWithSolvers pws;
    Subproblem(const std::string& scen, const std::vector<std::string>& args) : pc(args) {
        pws.setup(*pc.cfg, scen, W, z);
        // Assign in place: the solver objects hold pointers to these vectors.
        const std::vector<double> z0 = {0.0, 0.0, 1.0};
        std::copy(z0.begin(), z0.end(), z.begin());
        std::fill(W.begin(), W.end(), 0.0);
    }
    void setRho(double r) { pws.updateRho(std::vector<double>(W.size(), r)); }
};

void expectVec(const std::vector<double>& got, const std::vector<double>& want,
               double tol = 1e-5) {
    ASSERT_EQ(got.size(), want.size());
    for (size_t i = 0; i < want.size(); ++i)
        EXPECT_NEAR(got[i], want[i], tol) << "index " << i;
}

// doProxSolve gives the proximal minimizer, exports the same point, and
// agrees with a direct proximal solve on the Lagrangian object (the check
// aph-fw's prox_solver makes against FW).  The rho values have different
// answers, which shows that updateRho reaches proxSolver.
void checkProxSolve(const std::string& scen,
                    const std::vector<std::pair<double, std::vector<double>>>& cases) {
    for (const auto& [rho, want] : cases) {
        SCOPED_TRACE(scen + " rho=" + std::to_string(rho));
        Subproblem noFW(scen, kNoFWArgs);
        ASSERT_TRUE(noFW.pws.noFW());
        noFW.setRho(rho);
        std::vector<double> x(3, -1.0), xExport(3, -1.0);
        noFW.pws.doProxSolve(x, xExport);
        expectVec(x, want);
        expectVec(xExport, x, 0.0);

        Subproblem direct(scen, kFWArgs);
        direct.setRho(rho);
        direct.pws.enableLagrangianProxAt(direct.z);
        std::vector<double> xd(3, -1.0);
        direct.pws.doLagrangianSolve(xd);
        expectVec(x, xd);
    }
}

}  // namespace

TEST(NoFW, ProxSolveMatchesDirectProxContinuous) {
    checkProxSolve(kLPScen, {{1.0, {1.0, 0.0, 0.0}},
                             {20.0, {11.0 / 60, 2.0 / 60, 47.0 / 60}}});
}

// Unlike nl, proxSolver solves over K_s, so the reduced-cost bounds must
// reach it.
TEST(NoFW, NonantBoundsReachProxSolver) {
    Subproblem noFW(kLPScen, kNoFWArgs);
    noFW.setRho(1.0);
    noFW.pws.updateNonantBounds({0.0, 0.0, 0.0}, {0.0, 1.0, 1.0});  // x1 = 0
    std::vector<double> x(3), xExport(3);
    noFW.pws.doProxSolve(x, xExport);
    expectVec(x, {0.0, 1.0, 0.0});
}

// No FW object is created, so its accessors report "no FW solve" and the FW
// entry point refuses; the FW build refuses the prox entry point.
TEST(NoFW, NoFrankWolfeObjects) {
    Subproblem noFW(kLPScen, kNoFWArgs);
    EXPECT_TRUE(std::isnan(noFW.pws.getLastFWNLTime()));
    EXPECT_TRUE(std::isnan(noFW.pws.getLastFWMILPTime()));
    EXPECT_EQ(noFW.pws.getLastFWMILPSolves(), 0);
    EXPECT_TRUE(std::isnan(noFW.pws.getLastExportObjVal()));
    EXPECT_FALSE(noFW.pws.hasNumericalBailout());
    std::vector<double> x(3), xExport(3);
    EXPECT_THROW(noFW.pws.doFWsolve(x, xExport), std::logic_error);

    Subproblem fw(kLPScen, kFWArgs);
    EXPECT_FALSE(fw.pws.noFW());
    EXPECT_THROW(fw.pws.doProxSolve(x, xExport), std::logic_error);
}

// Each FW-only option is refused under --no-fw, even at its default value.
TEST(NoFW, RefusesFWOnlyOptions) {
    const std::vector<std::vector<std::string>> fwOnly = {
        {"--FW-tmax", "1"}, {"--FW-tau", "0.01"}, {"--FW-obj-cutoff"},
        {"--FW-obj-cutoff-tol", "0.001"}, {"--FW-export-policy", "prox"},
        {"--nl-json", "smallFWAPH/nl.json"}, {"--grad-json", "smallFWAPH/grad.json"}};
    for (const auto& extra : fwOnly) {
        SCOPED_TRACE(extra[0]);
        auto args = kNoFWArgs;
        args.insert(args.end(), extra.begin(), extra.end());
        ParsedConfig pc(args);
        std::vector<double> W, z;
        ProblemWithSolvers pws;
        EXPECT_THROW(pws.setup(*pc.cfg, kLPScen, W, z), std::runtime_error);
    }
}

TEST(NoFW, ProxJsonNeedsNoFW) {
    auto args = kFWArgs;
    args.insert(args.end(), {"--prox-json", "smallFWAPH/prox.json"});
    ParsedConfig pc(args);
    std::vector<double> W, z;
    ProblemWithSolvers pws;
    EXPECT_THROW(pws.setup(*pc.cfg, kLPScen, W, z), std::runtime_error);
}

TEST(NoFW, NeedsProxJson) {
    ParsedConfig pc({"--no-fw"});
    std::vector<double> W, z;
    ProblemWithSolvers pws;
    EXPECT_THROW(pws.setup(*pc.cfg, kLPScen, W, z), std::runtime_error);
}

#ifdef USE_HIGHS
// A prox solve on a model with integer variables is an MIQP; HiGHS refuses
// it when the objects are created, not at the first solve.
TEST(NoFW, HiGHSRefusesIntegerModel) {
    ParsedConfig pc(kNoFWArgs);
    std::vector<double> W, z;
    ProblemWithSolvers pws;
    EXPECT_THROW(pws.setup(*pc.cfg, kBinScen, W, z), std::runtime_error);
}
#else
// The MIQP case, which is what --no-fw is for.
TEST(NoFW, ProxSolveMatchesDirectProxBinary) {
    checkProxSolve(kBinScen, {{1.0, {1.0, 0.0, 0.0}}, {20.0, {0.0, 0.0, 1.0}}});
}
#endif  // USE_HIGHS
