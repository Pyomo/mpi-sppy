#
# Imports
#
import os
from queue import Empty
import sys
import pyomo
import pyomo.environ as pyo
import mpisppy
import mpisppy.phbase
import mpisppy.opt.ph
import mpisppy.opt.aph
import mpisppy.scenario_tree as scenario_tree
import pyomo.environ as pyo
from mpisppy.extensions.xhatspecific import XhatSpecific
import mpisppy.utils.sputils as sputils

# EMPRISE model class definition
class EmpriseModel:
    """
    Environment for Modelling and Planning Robust Investments in Sector-integrated Energy systems (EMPRISE) Model
    """

    def __repr__(self):
        return "EmpriseModel()"

    def __str__(self):
        return "Environment for Modelling and Planning Robust Investments in Sector-integrated Energy systems (EMPRISE) Model"

    def __init__(self, exclude_component_list=[]):
        """
        Create Abstract Pyomo model for the EMPRISE problem
        """
        self.THEORETICAL_CAP_LIMIT = 200.0  # To avoid unboundedness in oracle calls
        self.exclude_components = self._parse_exlude_components(exclude_component_list)
        self.abstract_model = self._create_abstract_model()

    def _parse_exlude_components(self, exclude_component_list):
        exclude_components = {"electricity_storage": False, "thermal_generation": False, "renewable_generation": False}
        if any("electricity_storage" in s for s in exclude_component_list):
            exclude_components["electricity_storage"] = True

        return exclude_components

    def _create_abstract_model(self):
        model = pyo.AbstractModel()
        model.name = "EMPRISE abstract model"

        model.numberOfStages = pyo.Param(within=pyo.NonNegativeIntegers)

        # Set definitions #############################################################################################################
        model.STAGE_MODEL = pyo.RangeSet(model.numberOfStages + 1)
        model.STAGE = pyo.RangeSet(model.numberOfStages)  # , within=model.STAGE_MODEL)

        def stageOperational_init(model):
            return [(x, y) for x in model.STAGE for y in model.STAGE if x <= y]

        model.STAGE_OPERATIONAL = pyo.Set(initialize=stageOperational_init, within=model.STAGE * model.STAGE)

        def stageDecommissioning_init(model):
            return [(x, y) for x in model.STAGE for y in model.STAGE if x < y]

        model.STAGE_DECOMMISSIONING = pyo.Set(initialize=stageDecommissioning_init, within=model.STAGE * model.STAGE)

        model.FUEL = pyo.Set()
        model.LOAD = pyo.Set()
        model.AREA = pyo.Set()
        model.TIME = pyo.Set()

        model.NODE = pyo.Set()  # Example: DEU1

        model.GEN_THERMAL = pyo.Set()
        model.GEN_THERMAL_TYPE = pyo.Set()  # Example: OCGT
        model.GEN_RENEWABLE = pyo.Set()
        model.GEN_RENEWABLE_TYPE = pyo.Set()  # Example: ONSHORE_IEC_1
        model.GEN_TYPE = model.GEN_THERMAL_TYPE | model.GEN_RENEWABLE_TYPE

        model.STORAGE = pyo.Set()
        model.STORAGE_TYPE = pyo.Set()  # Example: LI_ION

        model.BRANCH = pyo.Set(dimen=2)

        def nodesOut_init(model, node):
            retval = []
            for (i, j) in model.BRANCH:
                if i == node:
                    retval.append(j)
            return retval

        model.NODE_OUT = pyo.Set(model.NODE, initialize=nodesOut_init)

        def nodesIn_init(model, node):
            retval = []
            for (i, j) in model.BRANCH:
                if j == node:
                    retval.append(i)
            return retval

        model.NODE_IN = pyo.Set(model.NODE, initialize=nodesIn_init)

        # Parameters =============================================================================
        # - General
        # model.samplefactor = pyo.Param(model.TIME, within=pyo.NonNegativeReals)
        model.financePresentValueInterestRate = pyo.Param(within=pyo.Reals)
        model.financeDeprecationPeriodGenerationThermal = pyo.Param(model.GEN_THERMAL_TYPE, within=pyo.NonNegativeIntegers)
        model.financeDeprecationPeriodGenerationRenewable = pyo.Param(model.GEN_RENEWABLE_TYPE, within=pyo.NonNegativeIntegers)
        model.willingnessToPay = pyo.Param(within=pyo.NonNegativeReals)
        model.yearsPerStage = pyo.Param(within=pyo.NonNegativeIntegers)
        model.periodWeightFactor = pyo.Param(within=pyo.NonNegativeReals)
        model.emissionFactor = pyo.Param(model.FUEL, within=pyo.NonNegativeReals)  # tCO2eq/GWh_th

        # - Operational contributions of different technology development stages
        model.operationalStageContributionGeneration = pyo.Param(
            model.STAGE_OPERATIONAL, model.STAGE, model.GEN_THERMAL_TYPE | model.GEN_RENEWABLE_TYPE, within=pyo.PercentFraction
        )  # model.STAGE represents the technology development stages
        model.operationalStageContributionStorage = pyo.Param(model.STAGE_OPERATIONAL, model.STAGE, model.STORAGE_TYPE, within=pyo.PercentFraction)  # model.STAGE represents the technology development stages

        # - (Re)investment stage information for market uptake bounds
        model.investmentStageGeneration = pyo.Param(model.STAGE_OPERATIONAL, model.GEN_THERMAL_TYPE | model.GEN_RENEWABLE_TYPE, within=pyo.Boolean)
        model.investmentStageStorage = pyo.Param(model.STAGE_OPERATIONAL, model.STORAGE | model.STORAGE_TYPE, within=pyo.Boolean)

        # - Generation thermal
        model.generationThermalNode = pyo.Param(model.GEN_THERMAL, within=model.NODE)
        model.generationThermalType = pyo.Param(model.GEN_THERMAL, within=model.GEN_THERMAL_TYPE)
        model.generationThermalFuel = pyo.Param(model.GEN_THERMAL, within=model.FUEL)
        model.generationThermalEta = pyo.Param(model.GEN_THERMAL, within=pyo.NonNegativeReals)
        model.generationThermalPotentialMax = pyo.Param(model.STAGE, model.GEN_THERMAL, within=pyo.Reals)
        model.generationThermalMarketGrowthMin = pyo.Param(model.STAGE, model.GEN_THERMAL, within=pyo.Reals)
        model.generationThermalMarketGrowthMax = pyo.Param(model.STAGE, model.GEN_THERMAL, within=pyo.Reals)

        # - Generation renewable
        model.generationRenewableNode = pyo.Param(model.GEN_RENEWABLE, within=model.NODE)
        model.generationRenewableType = pyo.Param(model.GEN_RENEWABLE, within=model.GEN_RENEWABLE_TYPE)
        model.generationRenewableIec = pyo.Param(model.GEN_RENEWABLE, within=pyo.Integers)
        model.generationRenewableLcoe = pyo.Param(model.GEN_RENEWABLE, within=pyo.Any)
        model.generationRenewableProfile = pyo.Param(model.STAGE, model.GEN_RENEWABLE, model.TIME, within=pyo.Reals)
        model.generationRenewablePotentialMax = pyo.Param(model.STAGE, model.GEN_RENEWABLE, within=pyo.Reals)
        model.generationRenewableMarketGrowthMin = pyo.Param(model.STAGE, model.GEN_RENEWABLE, within=pyo.Reals)
        model.generationRenewableMarketGrowthMax = pyo.Param(model.STAGE, model.GEN_RENEWABLE, within=pyo.Reals)

        # - Conventional electricity load
        model.convLoadNode = pyo.Param(model.LOAD, within=model.NODE)
        model.convLoadAnnualDemand = pyo.Param(model.LOAD, within=pyo.Reals)
        model.convLoadProfile = pyo.Param(model.STAGE, model.LOAD, model.TIME, within=pyo.Reals)

        # - Storage
        model.storageNode = pyo.Param(model.STORAGE, within=model.NODE)
        model.storageType = pyo.Param(model.STORAGE, within=model.STORAGE_TYPE)
        model.storageEtaOut = pyo.Param(model.STORAGE, within=pyo.PercentFraction)
        model.storageEtaIn = pyo.Param(model.STORAGE, within=pyo.PercentFraction)
        model.storageRatioVolume = pyo.Param(model.STORAGE, within=pyo.NonNegativeReals)
        model.storageSelfDischargeRate = pyo.Param(model.STORAGE, within=pyo.PercentFraction)
        model.storageDepthOfDischarge = pyo.Param(model.STORAGE, within=pyo.PercentFraction)
        model.storagePotentialMax = pyo.Param(model.STAGE, model.STORAGE, within=pyo.Reals)
        model.storageMarketGrowthMin = pyo.Param(model.STAGE, model.STORAGE, within=pyo.Reals)
        model.storageMarketGrowthMax = pyo.Param(model.STAGE, model.STORAGE, within=pyo.Reals)

        # - Transmission flows
        model.branchDistance = pyo.Param(model.BRANCH, within=pyo.NonNegativeReals)
        model.branchExistingCapacity = pyo.Param(model.BRANCH, within=pyo.NonNegativeReals)
        model.branchType = pyo.Param(model.BRANCH, within=pyo.Any)  # within=model.BRANCH_TYPE
        model.branchExistingExpand = pyo.Param(model.BRANCH, within=pyo.Boolean)

        # - Costs
        model.costSystemOperationEmissionPrice = pyo.Param(model.STAGE, within=pyo.NonNegativeReals)
        model.costSystemOperationFuel = pyo.Param(model.STAGE, model.FUEL, within=pyo.NonNegativeReals)

        # --- Generation (thermal and renewable technologies)
        model.costGenerationCapex = pyo.Param(model.STAGE, model.GEN_TYPE, within=pyo.NonNegativeReals)
        model.costGenerationDepreciationPeriod = pyo.Param(model.STAGE, model.GEN_TYPE, within=pyo.NonNegativeIntegers)
        model.costGenerationInterestRate = pyo.Param(model.STAGE, model.GEN_TYPE, within=pyo.NonNegativeReals)
        model.costGenerationOpexVariable = pyo.Param(model.STAGE, model.GEN_TYPE, within=pyo.NonNegativeReals)
        model.costGenerationOpexFixed = pyo.Param(model.STAGE, model.GEN_TYPE, within=pyo.NonNegativeReals)

        model.multiPeriodCostGenerationTotalInvestment = pyo.Param(model.STAGE, model.GEN_TYPE, within=pyo.NonNegativeReals)
        model.multiPeriodCostGenerationDecommissioning = pyo.Param(model.STAGE_DECOMMISSIONING, model.GEN_TYPE, within=pyo.NonNegativeReals)

        # --- Storage
        model.costStorageCapex = pyo.Param(model.STAGE, model.STORAGE_TYPE, within=pyo.NonNegativeReals)
        model.costStorageDepreciationPeriod = pyo.Param(model.STAGE, model.STORAGE_TYPE, within=pyo.NonNegativeIntegers)
        model.costStorageInterestRate = pyo.Param(model.STAGE, model.STORAGE_TYPE, within=pyo.NonNegativeReals)
        model.costStorageOpexVariable = pyo.Param(model.STAGE, model.STORAGE_TYPE, within=pyo.NonNegativeReals)
        model.costStorageOpexFixed = pyo.Param(model.STAGE, model.STORAGE_TYPE, within=pyo.NonNegativeReals)

        model.multiPeriodCostStorageTotalInvestment = pyo.Param(model.STAGE, model.STORAGE_TYPE, within=pyo.NonNegativeReals)
        model.multiPeriodCostStorageDecommissioning = pyo.Param(model.STAGE_DECOMMISSIONING, model.STORAGE_TYPE, within=pyo.NonNegativeReals)

        # Decision variables =============================================================================
        # - Bounds rules
        def _generatorThermalCapacity_bounds_rule(model, g, *stg):
            return (0, model.generationThermalPotentialMax[stg[1], g])

        def _generatorThermalNewCapacity_bounds_rule(model, g, stg):
            return (0, model.generationThermalPotentialMax[stg, g])

        def _generationThermal_bounds_rule(model, g, *stg):
            return (0, model.generationThermalPotentialMax[stg[1], g])

        def _generatorRenewableCapacity_bounds_rule(model, g, *stg):
            return (0, model.generationRenewablePotentialMax[stg[1], g])

        def _generatorRenewableNewCapacity_bounds_rule(model, g, stg):
            return (0, model.generationRenewablePotentialMax[stg, g])

        def _generationRenewable_bounds_rule(model, g, *stg):
            return (0, model.generationRenewablePotentialMax[stg[1], g])

        def _electricityStorageCapacity_bounds_rule(model, s, *stg):
            return (0, model.storagePotentialMax[stg[1], s])

        def _electricityStorageNewCapacity_bounds_rule(model, s, stg):
            return (0, model.storagePotentialMax[stg, s])

        def _electricityStorage_bounds_rule(model, g, *stg):
            return (0, model.storagePotentialMax[stg[1], g])

        # - generationThermal capacity: operational, investment, decommissioning
        model.generatorThermalCapacity = pyo.Var(model.GEN_THERMAL, model.STAGE_OPERATIONAL, bounds=_generatorThermalCapacity_bounds_rule, within=pyo.NonNegativeReals)
        model.generatorThermalNewCapacity = pyo.Var(model.GEN_THERMAL, model.STAGE, bounds=_generatorThermalNewCapacity_bounds_rule, within=pyo.NonNegativeReals)
        model.generatorThermalDecommissionedCapacity = pyo.Var(model.GEN_THERMAL, model.STAGE_DECOMMISSIONING, bounds=(0, self.THEORETICAL_CAP_LIMIT), within=pyo.NonNegativeReals)

        # - generationRenewable capacity: operational, investment, decommissioning
        model.generatorRenewableCapacity = pyo.Var(model.GEN_RENEWABLE, model.STAGE_OPERATIONAL, bounds=_generatorRenewableCapacity_bounds_rule, within=pyo.NonNegativeReals)
        model.generatorRenewableNewCapacity = pyo.Var(model.GEN_RENEWABLE, model.STAGE, bounds=_generatorRenewableNewCapacity_bounds_rule, within=pyo.NonNegativeReals)
        model.generatorRenewableDecommissionedCapacity = pyo.Var(model.GEN_RENEWABLE, model.STAGE_DECOMMISSIONING, bounds=(0, self.THEORETICAL_CAP_LIMIT), within=pyo.NonNegativeReals)

        # - Generator output (bounds set by constraint)
        model.generationThermal = pyo.Var(model.GEN_THERMAL, model.STAGE_OPERATIONAL, model.TIME, bounds=_generationThermal_bounds_rule, within=pyo.NonNegativeReals)
        model.generationRenewable = pyo.Var(model.GEN_RENEWABLE, model.STAGE_OPERATIONAL, model.TIME, bounds=_generationRenewable_bounds_rule, within=pyo.NonNegativeReals)
        model.curtailmentRenewable = pyo.Var(model.GEN_RENEWABLE, model.STAGE_OPERATIONAL, model.TIME, bounds=_generationRenewable_bounds_rule, within=pyo.NonNegativeReals)

        # - Load shedding (for every node)
        model.loadShedding = pyo.Var(model.NODE, model.STAGE, model.TIME, bounds=(0, self.THEORETICAL_CAP_LIMIT), within=pyo.NonNegativeReals)  # Upper bound 200 GWel

        if not self.exclude_components["electricity_storage"]:
            # - Electricity storage capacity: operational, investment, decommissioning
            model.electricityStorageCapacity = pyo.Var(model.STORAGE, model.STAGE_OPERATIONAL, bounds=_electricityStorageCapacity_bounds_rule, within=pyo.NonNegativeReals)
            model.electricityStorageNewCapacity = pyo.Var(model.STORAGE, model.STAGE, bounds=_electricityStorageNewCapacity_bounds_rule, within=pyo.NonNegativeReals)
            model.electricityStorageDecommissionedCapacity = pyo.Var(model.STORAGE, model.STAGE_DECOMMISSIONING, within=pyo.NonNegativeReals)

            # - Electricity storage in- and output and level
            model.generationElectricityStorage = pyo.Var(model.STORAGE, model.STAGE_OPERATIONAL, model.TIME, bounds=_electricityStorage_bounds_rule, within=pyo.NonNegativeReals)
            model.consumptionElectricityStorage = pyo.Var(model.STORAGE, model.STAGE_OPERATIONAL, model.TIME, bounds=_electricityStorage_bounds_rule, within=pyo.NonNegativeReals)
            model.storageLevelElectricityStorage = pyo.Var(model.STORAGE, model.STAGE_OPERATIONAL, model.TIME, within=pyo.NonNegativeReals)

        def _flow_bounds_rule(model, n1, n2, stg, t):
            return (0, model.branchExistingCapacity[n1, n2])

        # - Transmission flows
        model.flow1 = pyo.Var(model.BRANCH, model.STAGE, model.TIME, bounds=_flow_bounds_rule, within=pyo.NonNegativeReals)
        model.flow2 = pyo.Var(model.BRANCH, model.STAGE, model.TIME, bounds=_flow_bounds_rule, within=pyo.NonNegativeReals)

        # Expressions #####################################################################################################################
        def generatorThermalCapacityInvestment_rule(model, g, stage_op):
            expr = model.generatorThermalNewCapacity[g, stage_op]  # stage_op == stage_inv
            for stg_inv in range(1, stage_op):
                if model.investmentStageGeneration[(stg_inv, stage_op), model.generationThermalType[g]] == True:
                    expr += model.generatorThermalCapacity[g, (stg_inv, stage_op)]
            return expr

        model.generatorThermalCapacityInvestment = pyo.Expression(model.GEN_THERMAL, model.STAGE, rule=generatorThermalCapacityInvestment_rule)

        def generatorRenewableCapacityInvestment_rule(model, g, stage_op):
            expr = model.generatorRenewableNewCapacity[g, stage_op]  # stage_op == stage_inv
            for stg_inv in range(1, stage_op):
                if model.investmentStageGeneration[(stg_inv, stage_op), model.generationRenewableType[g]] == True:
                    expr += model.generatorRenewableCapacity[g, (stg_inv, stage_op)]
            return expr

        model.generatorRenewableCapacityInvestment = pyo.Expression(model.GEN_RENEWABLE, model.STAGE, rule=generatorRenewableCapacityInvestment_rule)

        if not self.exclude_components["electricity_storage"]:

            def storageCapacityInvestment_rule(model, s, stage_op):
                expr = model.electricityStorageNewCapacity[s, stage_op]  # stage_op == stage_inv
                for stg_inv in range(1, stage_op):
                    if model.investmentStageStorage[(stg_inv, stage_op), model.storageType[s]] == True:
                        expr += model.electricityStorageCapacity[s, (stg_inv, stage_op)]
                return expr

            model.electricityStorageCapacityInvestment = pyo.Expression(model.STORAGE, model.STAGE, rule=storageCapacityInvestment_rule)

        # Constraints =============================================================================
        # - Thermal power generation, capacity expansion and decommissioning
        def maxGeneratorThermalOutput_rule(model, g, t, *stage):
            expr = model.generationThermal[g, stage, t] <= model.generatorThermalCapacity[g, stage]  # GW_el <= GW_el
            return expr

        model.c_maxGenThermal = pyo.Constraint(
            model.GEN_THERMAL,
            model.TIME,
            model.STAGE_OPERATIONAL,
            rule=maxGeneratorThermalOutput_rule,
        )

        def generationThermalCapacity_rule(model, g, *stage):
            if stage[0] == stage[1]:  # TODO: add pre-existing capacity parameter for initial stage, i.e. stage[0] == 1
                expr = model.generatorThermalCapacity[g, stage] == model.generatorThermalNewCapacity[g, stage[0]]
            else:
                expr = model.generatorThermalCapacity[g, stage] == model.generatorThermalCapacity[g, stage[0], stage[1] - 1] - model.generatorThermalDecommissionedCapacity[g, stage]
            return expr

        model.c_generationThermalCapacity = pyo.Constraint(model.GEN_THERMAL, model.STAGE_OPERATIONAL, rule=generationThermalCapacity_rule)

        def generationThermalCapacityDecommissioning_rule(model, g, *stage):
            expr = model.generatorThermalDecommissionedCapacity[g, stage] <= model.generatorThermalCapacity[g, stage[0], stage[1] - 1]
            return expr

        model.c_generationThermalCapacityDecommissioning = pyo.Constraint(model.GEN_THERMAL, model.STAGE_DECOMMISSIONING, rule=generationThermalCapacityDecommissioning_rule)

        def generationThermalCapacityMaximumPotential_rule(model, g, stage):
            expr = 0
            for stage_inv in range(stage):
                expr += model.generatorThermalCapacity[g, stage_inv + 1, stage]
            expr = expr <= model.generationThermalPotentialMax[stage, g]
            return expr

        model.c_generationThermalCapacityMaximumPotential = pyo.Constraint(model.GEN_THERMAL, model.STAGE, rule=generationThermalCapacityMaximumPotential_rule)

        def generatorThermalCapacityInvestmentMaximum_rule(model, g, stage):
            if stage > 1 and model.generationThermalMarketGrowthMax[stage, g] != float("inf"):
                expr = model.generatorThermalCapacityInvestment[g, stage - 1] * (1 + model.generationThermalMarketGrowthMax[stage, g]) ** model.yearsPerStage >= model.generatorThermalCapacityInvestment[g, stage]
            else:
                expr = pyo.Constraint.Skip
            return expr

        model.c_generatorThermalCapacityInvestmentMaximum = pyo.Constraint(model.GEN_THERMAL, model.STAGE, rule=generatorThermalCapacityInvestmentMaximum_rule)

        def generatorThermalCapacityInvestmentMinimum_rule(model, g, stage):
            if stage > 1 and model.generationThermalMarketGrowthMin[stage, g] != 0.0:
                expr = model.generatorThermalCapacityInvestment[g, stage - 1] * (1 + model.generationThermalMarketGrowthMin[stage, g]) ** model.yearsPerStage <= model.generatorThermalCapacityInvestment[g, stage]
            else:
                expr = pyo.Constraint.Skip
            return expr

        model.c_generatorThermalCapacityInvestmentMinimum = pyo.Constraint(model.GEN_THERMAL, model.STAGE, rule=generatorThermalCapacityInvestmentMinimum_rule)

        # - Renewable power generation, capacity expansion and decommissioning
        def maxGeneratorRenewableOutput_rule(model, g, t, *stage):
            tmp_combined_profile = 0.0
            for stg in model.STAGE:
                tmp_combined_profile += model.operationalStageContributionGeneration[stage[0], stage[1], stg, model.generationRenewableType[g]] * model.generationRenewableProfile[stg, g, t]
                if tmp_combined_profile <= 1e-6:
                    tmp_combined_profile = 0
            expr = model.generationRenewable[g, stage, t] + model.curtailmentRenewable[g, stage, t] == model.generatorRenewableCapacity[g, stage] * tmp_combined_profile  # GW_el <= GW_el
            return expr

        model.c_maxGenRenewable = pyo.Constraint(
            model.GEN_RENEWABLE,
            model.TIME,
            model.STAGE_OPERATIONAL,
            rule=maxGeneratorRenewableOutput_rule,
        )

        def generationRenewableCapacity_rule(model, g, *stage):
            if stage[0] == stage[1]:  # TODO: add pre-existing capacity parameter for initial stage, i.e. stage[0] == 1
                expr = model.generatorRenewableCapacity[g, stage] == model.generatorRenewableNewCapacity[g, stage[0]]
            else:
                expr = model.generatorRenewableCapacity[g, stage] == model.generatorRenewableCapacity[g, stage[0], stage[1] - 1] - model.generatorRenewableDecommissionedCapacity[g, stage]
            return expr

        model.c_generationRenewableCapacity = pyo.Constraint(model.GEN_RENEWABLE, model.STAGE_OPERATIONAL, rule=generationRenewableCapacity_rule)

        def generationRenewableCapacityDecommissioning_rule(model, g, *stage):
            expr = model.generatorRenewableDecommissionedCapacity[g, stage] <= model.generatorRenewableCapacity[g, stage[0], stage[1] - 1]
            return expr

        model.c_generationRenewableCapacityDecommissioning = pyo.Constraint(model.GEN_RENEWABLE, model.STAGE_DECOMMISSIONING, rule=generationRenewableCapacityDecommissioning_rule)

        def generationRenewableCapacityMaximumPotential_rule(model, g, stage):
            expr = 0
            for stage_inv in range(stage):
                expr += model.generatorRenewableCapacity[g, stage_inv + 1, stage]
            expr = expr <= model.generationRenewablePotentialMax[stage, g]
            return expr

        model.c_generationRenewableCapacityMaximumPotential = pyo.Constraint(model.GEN_RENEWABLE, model.STAGE, rule=generationRenewableCapacityMaximumPotential_rule)

        def generatorRenewableCapacityInvestmentMaximum_rule(model, g, stage):
            if stage > 1 and model.generationRenewableMarketGrowthMax[stage, g] != float("inf"):
                expr = model.generatorRenewableCapacityInvestment[g, stage - 1] * (1 + model.generationRenewableMarketGrowthMax[stage, g]) ** model.yearsPerStage >= model.generatorRenewableCapacityInvestment[g, stage]
            else:
                expr = pyo.Constraint.Skip
            return expr

        model.c_generatorRenewableCapacityInvestmentMaximum = pyo.Constraint(model.GEN_RENEWABLE, model.STAGE, rule=generatorRenewableCapacityInvestmentMaximum_rule)

        def generatorRenewableCapacityInvestmentMinimum_rule(model, g, stage):
            if stage > 1 and model.generationRenewableMarketGrowthMin[stage, g] != 0.0:
                expr = model.generatorRenewableCapacityInvestment[g, stage - 1] * (1 + model.generationRenewableMarketGrowthMin[stage, g]) ** model.yearsPerStage <= model.generatorRenewableCapacityInvestment[g, stage]
            else:
                expr = pyo.Constraint.Skip
            return expr

        model.c_generatorRenewableCapacityInvestmentMinimum = pyo.Constraint(model.GEN_RENEWABLE, model.STAGE, rule=generatorRenewableCapacityInvestmentMinimum_rule)

        # - Electricity storage generation, consumption, capacity expansion and decommissioning
        if not self.exclude_components["electricity_storage"]:

            def electricityStorageCapacity_rule(model, s, *stage):
                if stage[0] == stage[1]:  # TODO: add pre-existing capacity parameter for initial stage, i.e. stage[0] == 1
                    expr = model.electricityStorageCapacity[s, stage] == model.electricityStorageNewCapacity[s, stage[0]]
                else:
                    expr = model.electricityStorageCapacity[s, stage] == model.electricityStorageCapacity[s, stage[0], stage[1] - 1] - model.electricityStorageDecommissionedCapacity[s, stage]
                return expr

            model.c_electricityStorageCapacity = pyo.Constraint(model.STORAGE, model.STAGE_OPERATIONAL, rule=electricityStorageCapacity_rule)

            def electricityStorageCapacityDecommissioning_rule(model, s, *stage):
                expr = model.electricityStorageDecommissionedCapacity[s, stage] <= model.electricityStorageCapacity[s, stage[0], stage[1] - 1]
                return expr

            model.c_electricityStorageCapacityDecommissioning = pyo.Constraint(model.STORAGE, model.STAGE_DECOMMISSIONING, rule=electricityStorageCapacityDecommissioning_rule)

            def electricityStorageCapacityMaximumPotential_rule(model, s, stage):
                expr = 0
                for stage_inv in range(stage):
                    expr += model.electricityStorageCapacity[s, stage_inv + 1, stage]
                expr = expr <= model.storagePotentialMax[stage, s]
                return expr

            model.c_electricityStorageCapacityMaximumPotential = pyo.Constraint(model.STORAGE, model.STAGE, rule=electricityStorageCapacityMaximumPotential_rule)

            def electricityStorageCapacityInvestmentMaximum_rule(model, s, stage):
                if stage > 1 and model.storageMarketGrowthMax[stage, s] != float("inf"):
                    expr = model.electricityStorageCapacityInvestment[s, stage - 1] * (1 + model.storageMarketGrowthMax[stage, s]) ** model.yearsPerStage >= model.electricityStorageCapacityInvestment[s, stage]
                else:
                    expr = pyo.Constraint.Skip
                return expr

            model.c_electricityStorageCapacityInvestmentMaximum = pyo.Constraint(model.STORAGE, model.STAGE, rule=electricityStorageCapacityInvestmentMaximum_rule)

            def electricityStorageCapacityInvestmentMinimum_rule(model, s, stage):
                if stage > 1 and model.storageMarketGrowthMin[stage, s] != 0.0:
                    expr = model.electricityStorageCapacityInvestment[s, stage - 1] * (1 + model.storageMarketGrowthMin[stage, s]) ** model.yearsPerStage <= model.electricityStorageCapacityInvestment[s, stage]
                else:
                    expr = pyo.Constraint.Skip
                return expr

            model.c_electricityStorageCapacityInvestmentMinimum = pyo.Constraint(model.STORAGE, model.STAGE, rule=electricityStorageCapacityInvestmentMinimum_rule)

            def maxElectricityStorageOutput_rule(model, s, t, *stage):
                expr = model.generationElectricityStorage[s, stage, t] <= model.electricityStorageCapacity[s, stage]  # GW <= GW
                return expr

            model.c_maxElectricityStorageOutput = pyo.Constraint(
                model.STORAGE,
                model.TIME,
                model.STAGE_OPERATIONAL,
                rule=maxElectricityStorageOutput_rule,
            )

            def maxElectricityStorageInput_rule(model, s, t, *stage):
                expr = model.consumptionElectricityStorage[s, stage, t] <= model.electricityStorageCapacity[s, stage]  # GW <= GW
                return expr

            model.c_maxElectricityStorageInput = pyo.Constraint(
                model.STORAGE,
                model.TIME,
                model.STAGE_OPERATIONAL,
                rule=maxElectricityStorageInput_rule,
            )

            def maxElectricityStorageLevel_rule(model, s, t, *stage):
                expr = model.storageLevelElectricityStorage[s, stage, t] <= model.electricityStorageCapacity[s, stage] * model.storageRatioVolume[s]  # GWh <= GWh
                return expr

            model.c_maxElectricityStorageLevel = pyo.Constraint(
                model.STORAGE,
                model.TIME,
                model.STAGE_OPERATIONAL,
                rule=maxElectricityStorageLevel_rule,
            )

            def electricityStorageContinuity_rule(model, s, t, *stage):
                expr = 0
                if t < max(model.TIME):  # (1 - model.storageSelfDischargeRate[s]) *
                    expr = (
                        model.storageLevelElectricityStorage[s, stage, t + 1]
                        == model.storageLevelElectricityStorage[s, stage, t]
                        + model.storageEtaIn[s] * model.consumptionElectricityStorage[s, stage, t]
                        - model.generationElectricityStorage[s, stage, t] * 1 / model.storageEtaOut[s]
                    )  # Storage continuity equation
                else:
                    expr = (
                        model.storageLevelElectricityStorage[s, stage, t]
                        + model.storageEtaIn[s] * model.consumptionElectricityStorage[s, stage, t]
                        - model.generationElectricityStorage[s, stage, t] * 1 / model.storageEtaOut[s]
                        >= model.storageLevelElectricityStorage[s, stage, min(model.TIME)]
                    )  # Final equal to initial storage level of a representative period or the full year
                return expr

            model.c_electricityStorageContinuity = pyo.Constraint(
                model.STORAGE,
                model.TIME,
                model.STAGE_OPERATIONAL,
                rule=electricityStorageContinuity_rule,
            )

        # - Transmission flows
        def maxTransmissionLimit_rule(model, n1, n2, t, stage):
            expr = model.flow1[n1, n2, stage, t] + model.flow2[n1, n2, stage, t] <= model.branchExistingCapacity[n1, n2]
            return expr

        model.c_maxTransmissionLimit = pyo.Constraint(model.BRANCH, model.TIME, model.STAGE, rule=maxTransmissionLimit_rule)

        # - Nodal power balance (market clearing constraint)
        def nodalPowerBalance_rule(model, n, t, stage):
            expr = 0

            # Load shedding
            expr += model.loadShedding[n, stage, t]

            # Thermal generation
            for g in model.GEN_THERMAL:
                if model.generationThermalNode[g] == n:
                    for s in range(1, stage + 1):
                        expr += model.generationThermal[g, s, stage, t]

            # Renewable generation
            for g in model.GEN_RENEWABLE:
                if model.generationRenewableNode[g] == n:
                    for s in range(1, stage + 1):
                        expr += model.generationRenewable[g, s, stage, t]

            # Conventional consumer load
            for c in model.LOAD:
                if model.convLoadNode[c] == n:
                    if model.convLoadAnnualDemand[c] != -1:
                        # expr += -model.convLoadAnnualDemand[c] * model.convLoadProfile[c, t] / sum(model.convLoadProfile[c, t1] for t1 in model.TIME)
                        expr -= model.convLoadProfile[stage, c, t] / 1000 * (1.0 + 0.25 * stage)  # GW
                    else:
                        expr -= model.convLoadProfile[stage, c, t] / 1000 * (1.0 + 0.25 * stage)  # GW

            # Electricity storage
            if not self.exclude_components["electricity_storage"]:
                for su in model.STORAGE:
                    if model.storageNode[su] == n:
                        for s in range(1, stage + 1):
                            expr += model.generationElectricityStorage[su, s, stage, t]
                            expr -= model.consumptionElectricityStorage[su, s, stage, t]

            expr += sum(model.flow1[i, n, stage, t] for i in model.NODE_IN[n])
            expr += -sum(model.flow2[i, n, stage, t] for i in model.NODE_IN[n]) * 1.03093  # TODO: Include transmission loss factors as parameters
            expr += -sum(model.flow1[n, j, stage, t] for j in model.NODE_OUT[n]) * 1.03093  # TODO: Include transmission loss factors as parameters
            expr += sum(model.flow2[n, j, stage, t] for j in model.NODE_OUT[n])

            expr = expr == 0

            if (type(expr) is bool) and (expr == True):
                # Trivial constraint
                expr = pyo.Constraint.Skip
            return expr

        model.c_nodalPowerBalance = pyo.Constraint(model.NODE, model.TIME, model.STAGE, rule=nodalPowerBalance_rule)

        # Objective =============================================================================
        def costInvestment_rule(model, stage):
            """CAPEX and OPEX for investment decisions, ...(NPV)"""
            if stage <= model.numberOfStages:
                expr = self.get_cost_investments(model, stage)
            else:
                expr = 0.0
            return expr  # return model.costInvestment[stage] == expr

        model.costInvestment = pyo.Expression(model.STAGE_MODEL, rule=costInvestment_rule)

        def costSystemOperation_rule(model, stage):
            """System operation costs for power generation, storage, ... (NPV)"""
            if stage > 1:
                expr = self.get_cost_system_operation(model, stage - 1)
            else:
                expr = 0.0
            return expr  # model.costSystemOperation[stage] == expr

        model.costSystemOperation = pyo.Expression(model.STAGE_MODEL, rule=costSystemOperation_rule)

        def costTotal_rule(model, stage):
            """Total operation costs: cost of thermal generationThermal, ... (NPV)"""
            expr = model.costInvestment[stage] + model.costSystemOperation[stage]
            return expr

        model.costTotal = pyo.Expression(model.STAGE_MODEL, rule=costTotal_rule)

        def costTotal_Objective_rule(model):
            expr = pyo.summation(model.costTotal)
            return expr

        model.obj = pyo.Objective(rule=costTotal_Objective_rule, sense=pyo.minimize)

        return model

    def get_cost_investments(self, model, stage, includeRelativeOpex=False, subtractSalvage=True):
        """Investment cost, including lifetime OPEX (NPV)"""
        cost_investment = 0.0

        for g in model.GEN_THERMAL:
            cost_investment += self.get_cost_investment_by_unit_group(model, g, stage, "generationThermal")  # unit_group = "generationThermal"

        for g in model.GEN_RENEWABLE:
            cost_investment += self.get_cost_investment_by_unit_group(model, g, stage, "generationRenewable")  # unit_group = "generationRenewable"

        if not self.exclude_components["electricity_storage"]:
            for s in model.STORAGE:
                cost_investment += self.get_cost_investment_by_unit_group(model, s, stage, "electricityStorage")  # unit_group = "electricityStorage"

        return cost_investment

    def get_cost_investment_by_unit_group(self, model, u, stage, unit_group=None):
        """Expression for investment cost of specified unit group"""
        cost_investment = 0.0
        if unit_group == "generationThermal":
            gen_type = model.generationThermalType[u]
            cost_investment += model.multiPeriodCostGenerationTotalInvestment[stage, gen_type] * model.generatorThermalNewCapacity[u, stage]
            for stg_dec in range(1, stage):  # e.g. investment in 'stage' = 1, corresponding decommissioning in stages [2, 3, 4]
                cost_investment -= model.multiPeriodCostGenerationDecommissioning[stg_dec, stage, gen_type] * model.generatorThermalDecommissionedCapacity[u, stg_dec, stage]
        elif unit_group == "generationRenewable":
            gen_type = model.generationRenewableType[u]
            cost_investment += model.multiPeriodCostGenerationTotalInvestment[stage, gen_type] * model.generatorRenewableNewCapacity[u, stage]
            for stg_dec in range(1, stage):  # e.g. investment in 'stage' = 1, corresponding decommissioning in stages [2, 3, 4]
                cost_investment -= model.multiPeriodCostGenerationDecommissioning[stg_dec, stage, gen_type] * model.generatorRenewableDecommissionedCapacity[u, stg_dec, stage]
        elif unit_group == "electricityStorage" and not self.exclude_components["electricity_storage"]:
            storage_type = model.storageType[u]
            cost_investment += model.multiPeriodCostStorageTotalInvestment[stage, storage_type] * model.electricityStorageNewCapacity[u, stage]
            for stg_dec in range(1, stage):  # e.g. investment in 'stage' = 1, corresponding decommissioning in stages [2, 3, 4]
                cost_investment -= model.multiPeriodCostStorageDecommissioning[stg_dec, stage, storage_type] * model.electricityStorageDecommissionedCapacity[u, stg_dec, stage]
        else:
            print("\n!!! ERROR WRONG OR NO UNIT GROUP SPECIFIED WHEN OBTAINING INVESTMENT COST !!!\n")

        return cost_investment

    def get_cost_system_operation(self, model, stage):
        """Operational costs: cost of gen, load shed (NPV)"""
        cost_system_operation = 0.0

        (discount_factor_sum, discount_factor_perpetuity) = self.get_discount_factor_sum(pyo.value(model.financePresentValueInterestRate), stage, model.yearsPerStage)

        if stage == pyo.value(model.numberOfStages):
            discount_factor_sum = discount_factor_sum + discount_factor_perpetuity
        else:
            del discount_factor_perpetuity
            # print(discount_factor_sum)
        # print("Stage: " + str(stage), "Discount factor sum: " + str(discount_factor_sum), "Period weight factor: " + str(pyo.value(model.periodWeightFactor)))

        # Load shedding
        cost_system_operation += pyo.quicksum(
            model.loadShedding[n, stage, t] * model.willingnessToPay * discount_factor_sum * model.periodWeightFactor for n in model.NODE for t in model.TIME
        )  # willingness-to-pay is e.g. 3 MEUR/GW = 3000 EUR/MW

        for stage_operational_idx in model.STAGE_OPERATIONAL:
            if stage_operational_idx[1] == stage:
                # Thermal generation units

                # TODO: Thermal efficiencies
                # for g in model.GEN_THERMAL
                #     operational_stage = self.get_multi_period_information(
                #         None,
                #         None,
                #         None,
                #         model.financePresentValueInterestRate,
                #         [model.costGenerationDepreciationPeriod[x, model.generationThermalType[g]] for x in model.STAGE],
                #         stage[0],
                #         max(model.STAGE),
                #         model.yearsPerStage,
                #         False,
                #     )[
                #         2
                #     ]  # Only third output relevant
                #     tmp_combined_efficiency = np.multiply(
                #         operational_stage[stage[0] - 1, :], np.asarray([model.generationRenewableProfile[s + 1, g, t] for s in range(operational_stage.shape[1])])
                #     ).sum()  # Weighted sum of renewable availability = sum(Investment period weigths * renewable availability profile)

                cost_system_operation += pyo.quicksum(
                    model.generationThermal[g, stage_operational_idx, t]
                    * model.periodWeightFactor
                    / model.generationThermalEta[g]
                    * (model.costSystemOperationFuel[stage, model.generationThermalFuel[g]] + model.emissionFactor[model.generationThermalFuel[g]] * model.costSystemOperationEmissionPrice[stage])
                    * discount_factor_sum
                    for g in model.GEN_THERMAL
                    for t in model.TIME
                )

                # if t == min(model.TIME):
                #     print(
                #         "thermal sysop",
                #         0,
                #         model.periodWeightFactor
                #         / model.generationThermalEta[0]
                #         * (model.costSystemOperationFuel[stage, model.generationThermalFuel[0]] + model.emissionFactor[model.generationThermalFuel[0]] * model.costSystemOperationEmissionPrice[stage])
                #         * discount_factor_sum,
                #     )

                # Renewable generation units
                cost_system_operation += pyo.quicksum(
                    model.generationRenewable[g, stage_operational_idx, t] * model.costGenerationOpexVariable[stage, model.generationRenewableType[g]] * model.periodWeightFactor * discount_factor_sum
                    for g in model.GEN_RENEWABLE
                    for t in model.TIME
                )

                # Electricity storage units
                if not self.exclude_components["electricity_storage"]:
                    cost_system_operation += pyo.quicksum(
                        (model.generationElectricityStorage[s, t, stage_operational_idx] + model.consumptionElectricityStorage[s, t, stage_operational_idx])
                        * model.costStorageOpexVariable[stage, model.storageType[s]]
                        * model.periodWeightFactor
                        * discount_factor_sum
                        for s in model.STORAGE
                        for t in model.TIME
                    )

        return cost_system_operation

    def create_concrete_model(self, dict_data):
        """Create concrete Pyomo model for the EMPRISE optimization problem instance

        Parameters
        ----------
        dict_data : dictionary
            dictionary containing the model data. This can be created with
            the create_model_data(...) method

        Returns
        -------
            Concrete pyomo model
        """

        concrete_model = self.abstract_model.create_instance(data=dict_data, name="EMPRISE Model", namespace="emprise", report_timing=False)  # namespace important for input data dictionary
        return concrete_model

    def create_model_data(self, input_data):
        """Create model data in dictionary format

        Parameters
        ----------
        input_data : emprise.InputData object
            containing structural and timeseries input data

        Returns
        -------
        dictionary with pyomo data (in pyomo format)
        """

        di = {}

        # Sets =============================================================================
        di["NODE"] = {None: input_data.node["id"].tolist()}
        di["FUEL"] = {None: input_data.fuel["type"].tolist()}
        di["AREA"] = {None: input_data.get_all_areas()}
        di["TIME"] = {None: input_data.time_range}
        di["BRANCH"] = {None: [(row["node_from"], row["node_to"]) for k, row in input_data.branch.iterrows() if row["node_from"] in di["NODE"][None] and row["node_to"] in di["NODE"][None]]}

        # General =============================================================================
        di["numberOfStages"] = {None: input_data.number_of_stages}  # -
        di["willingnessToPay"] = {None: input_data.general["willingnessToPay"]}  # EUR/MWh
        di["financePresentValueInterestRate"] = {None: input_data.general["finance"]["interestRate"]}  # -

        di["yearsPerStage"] = {None: input_data.general["yearsPerStage"]}  # -
        di["periodWeightFactor"] = {None: input_data.ts_period_weight_factor}  # -

        # Generation thermal =============================================================================
        di["generationThermalNode"] = {}
        di["generationThermalType"] = {}
        di["generationThermalFuel"] = {}
        di["generationThermalEta"] = {}
        di["generationThermalPotentialMax"] = {}
        di["generationThermalMarketGrowthMin"] = {}
        di["generationThermalMarketGrowthMax"] = {}
        k = 0
        for it, row in input_data.generator_thermal.iterrows():
            if row["node"] in di["NODE"][None]:
                di["generationThermalNode"][k] = row["node"]
                di["generationThermalType"][k] = row["type"]
                di["generationThermalFuel"][k] = row["fuel"]
                di["generationThermalEta"][k] = row["eta"]
                for stg in range(1, input_data.number_of_stages + 1):
                    di["generationThermalPotentialMax"][(stg, k)] = float(row["potential_max"].split("::")[stg - 1])
                    di["generationThermalMarketGrowthMin"][(stg, k)] = float(row["market_growth_min"].split("::")[stg - 1])
                    di["generationThermalMarketGrowthMax"][(stg, k)] = float(row["market_growth_max"].split("::")[stg - 1])
                k = k + 1
        di["GEN_THERMAL"] = {None: range(k)}
        di["GEN_THERMAL_TYPE"] = {None: sorted(set(di["generationThermalType"].values()))}
        del k

        # Generation renewable =============================================================================
        di["generationRenewableNode"] = {}
        di["generationRenewableType"] = {}
        di["generationRenewableIec"] = {}
        di["generationRenewableLcoe"] = {}
        di["generationRenewablePotentialMax"] = {}
        di["generationRenewableMarketGrowthMin"] = {}
        di["generationRenewableMarketGrowthMax"] = {}
        di["generationRenewableProfile"] = {}
        k = 0
        for it, row in input_data.generator_renewable.iterrows():
            if row["node"] in di["NODE"][None]:
                di["generationRenewableNode"][k] = row["node"]
                di["generationRenewableType"][k] = row["type"]
                di["generationRenewableIec"][k] = row["iec"]
                di["generationRenewableLcoe"][k] = "LCOE_" + str(row["lcoe"])
                ref = row["node"]
                for stg in range(1, input_data.number_of_stages + 1):
                    di["generationRenewablePotentialMax"][(stg, k)] = float(row["potential_max"].split("::")[stg - 1])
                    di["generationRenewableMarketGrowthMin"][(stg, k)] = float(row["market_growth_min"].split("::")[stg - 1])
                    di["generationRenewableMarketGrowthMax"][(stg, k)] = float(row["market_growth_max"].split("::")[stg - 1])
                    # print(stg, stg - 1)
                    for i, t in enumerate(input_data.time_range):
                        if row["type"].startswith("SOLAR"):
                            tmp_tuple = (
                                row["node"],
                                row["type"],
                                "LCOE_" + str(row["lcoe"]),
                                str(input_data.reference_years[stg - 1]),
                            )
                            di["generationRenewableProfile"][(stg, k, t)] = input_data.ts_generator_solar[tmp_tuple][i]
                            del tmp_tuple
                        elif row["type"].startswith("ONSHORE") or row["type"].startswith("OFFSHORE"):
                            tmp_tuple = (
                                row["node"],
                                row["type"],
                                "LCOE_" + str(row["lcoe"]),
                                str(input_data.reference_years[stg - 1]),
                            )
                            di["generationRenewableProfile"][(stg, k, t)] = input_data.ts_generator_wind[tmp_tuple][i]
                            del tmp_tuple
                        else:
                            print(
                                "\n\n !!! WARNING: no timeseries data found for type {} !!!\n\n",
                                row["type"],
                            )
                k = k + 1
        di["GEN_RENEWABLE"] = {None: range(k)}
        di["GEN_RENEWABLE_TYPE"] = {None: sorted(set(di["generationRenewableType"].values()))}
        del k

        # Conventional load =============================================================================
        di["convLoadAnnualDemand"] = {}
        di["convLoadProfile"] = {}
        di["convLoadNode"] = {}
        k = 0
        for _, row in input_data.consumer_conventional.iterrows():
            if row["node"] in di["NODE"][None]:
                di["convLoadNode"][k] = row["node"]
                di["convLoadAnnualDemand"][k] = row["annualDemand"]
                ref = row["node"]
                for stg in range(1, input_data.number_of_stages + 1):
                    for i, t in enumerate(input_data.time_range):
                        di["convLoadProfile"][(stg, k, t)] = input_data.ts_consumer_conventional[(ref, str(input_data.reference_years[stg - 1]))][i]
                k = k + 1
        di["LOAD"] = {None: range(k)}
        del k

        # Storage =============================================================================
        di["storageNode"] = {}
        di["storageType"] = {}
        di["storageEtaOut"] = {}
        di["storageEtaIn"] = {}
        di["storageRatioVolume"] = {}
        di["storageSelfDischargeRate"] = {}
        di["storageDepthOfDischarge"] = {}
        di["storagePotentialMax"] = {}
        di["storageMarketGrowthMin"] = {}
        di["storageMarketGrowthMax"] = {}
        k = 0
        for it, row in input_data.storage.iterrows():
            if row["node"] in di["NODE"][None] and row["include"]:
                di["storageNode"][k] = row["node"]
                di["storageType"][k] = row["type"]
                di["storageEtaOut"][k] = row["eta_out"]
                di["storageEtaIn"][k] = row["eta_in"]
                di["storageRatioVolume"][k] = row["ratio_volume"]
                di["storageSelfDischargeRate"][k] = row["sdr"]
                di["storageDepthOfDischarge"][k] = row["dod"]
                for stg in range(1, input_data.number_of_stages + 1):
                    di["storagePotentialMax"][(stg, k)] = float(row["potential_max"].split("::")[stg - 1])
                    di["storageMarketGrowthMin"][(stg, k)] = float(row["market_growth_min"].split("::")[stg - 1])
                    di["storageMarketGrowthMax"][(stg, k)] = float(row["market_growth_max"].split("::")[stg - 1])
                k = k + 1
        if k == 0:
            di["STORAGE"] = {None: pyo.Set.Skip}
            di["STORAGE_TYPE"] = {None: []}
        else:
            di["STORAGE"] = {None: range(k)}
            di["STORAGE_TYPE"] = {None: sorted(set(di["storageType"].values()))}
        del k

        # Cross-border exchange =============================================================================
        di["branchDistance"] = {}
        di["branchType"] = {}
        di["branchExistingCapacity"] = {}
        di["branchExistingExpand"] = {}
        for k, row in input_data.branch.iterrows():
            if row["node_from"] in di["NODE"][None] and row["node_to"] in di["NODE"][None]:
                di["branchDistance"][(row["node_from"], row["node_to"])] = row["distance"]
                di["branchType"][(row["node_from"], row["node_to"])] = row["type"]
                di["branchExistingCapacity"][(row["node_from"], row["node_to"])] = row["capacity"]
                di["branchExistingExpand"][(row["node_from"], row["node_to"])] = row["expand"]

        # System operation costs =============================================================================
        # di["costSystemOperationFuel2"] = input_data.cost["systemOperation"]["fuel"]  # MEUR/GWh_th
        di["costSystemOperationEmissionPrice"] = input_data.cost["systemOperation"]["emissionPrice"]  # MEUR/tCO2eq

        di["emissionFactor"] = {}
        di["costSystemOperationFuel"] = {}
        for k, row in input_data.fuel.iterrows():
            di["emissionFactor"][row["type"]] = row["emission_factor"]  # tCO2eq/GWh_th
            for stg in range(1, input_data.number_of_stages + 1):
                di["costSystemOperationFuel"][(stg, row["type"])] = float(row["cost"].split("::")[stg - 1])  # MEUR/GWh_th

        di["costGenerationCapex"] = {}
        di["costGenerationDepreciationPeriod"] = {}
        di["costGenerationInterestRate"] = {}
        di["costGenerationOpexVariable"] = {}
        di["costGenerationOpexFixed"] = {}

        di["costStorageCapex"] = {}
        di["costStorageDepreciationPeriod"] = {}
        di["costStorageInterestRate"] = {}
        di["costStorageOpexVariable"] = {}
        di["costStorageOpexFixed"] = {}

        for stg in range(1, input_data.number_of_stages + 1):
            for k, row in input_data.cost_generation.iterrows():
                gen_type = row["type"]
                if gen_type in di["GEN_THERMAL_TYPE"][None] + di["GEN_RENEWABLE_TYPE"][None]:
                    di["costGenerationCapex"][(stg, gen_type)] = float(row["capex"].split("::")[stg - 1])  # EUR/MW
                    di["costGenerationDepreciationPeriod"][(stg, gen_type)] = int(row["depreciation"].split("::")[stg - 1])  # a
                    di["costGenerationInterestRate"][(stg, gen_type)] = float(row["interest_rate"].split("::")[stg - 1])  # p.u.
                    di["costGenerationOpexVariable"][(stg, gen_type)] = float(row["opex_variable"].split("::")[stg - 1])  # EUR/MWh
                    di["costGenerationOpexFixed"][(stg, gen_type)] = float(row["opex_fixed"].split("::")[stg - 1])  # EUR/MW/a
            for k, row in input_data.cost_storage.iterrows():
                storage_type = row["type"]
                if storage_type in di["STORAGE_TYPE"][None]:
                    di["costStorageCapex"][(stg, storage_type)] = float(row["capex"].split("::")[stg - 1])  # EUR/MW
                    di["costStorageDepreciationPeriod"][(stg, storage_type)] = int(row["depreciation"].split("::")[stg - 1])  # a
                    di["costStorageInterestRate"][(stg, storage_type)] = float(row["interest_rate"].split("::")[stg - 1])  # p.u.
                    di["costStorageOpexVariable"][(stg, storage_type)] = float(row["opex_variable"].split("::")[stg - 1])  # EUR/MWh
                    di["costStorageOpexFixed"][(stg, storage_type)] = float(row["opex_fixed"].split("::")[stg - 1])  # EUR/MW/a

        tmp_stage = range(1, 1 + di["numberOfStages"][None])
        tmp_stage_operational = [(x, y) for x in tmp_stage for y in tmp_stage if x <= y]
        # tmp_stage_decommissioning = [(x, y) for x in tmp_stage for y in tmp_stage if x < y]

        di["operationalStageContributionGeneration"] = {}
        di["multiPeriodCostGenerationTotalInvestment"] = {}
        di["multiPeriodCostGenerationDecommissioning"] = {}
        di["investmentStageGeneration"] = {}

        di["operationalStageContributionStorage"] = {}
        di["multiPeriodCostStorageTotalInvestment"] = {}
        di["multiPeriodCostStorageDecommissioning"] = {}
        di["investmentStageStorage"] = {}
        for stg_op in tmp_stage_operational:
            for gen_type in di["GEN_THERMAL_TYPE"][None] + di["GEN_RENEWABLE_TYPE"][None]:
                (total_cost, decommissioning_redemption, operational_stage_contribution_matrix, investment_stages) = self.get_multi_period_information(
                    [di["costGenerationCapex"][(x, gen_type)] for x in tmp_stage],
                    [di["costGenerationOpexFixed"][(x, gen_type)] for x in tmp_stage],
                    [di["costGenerationInterestRate"][(x, gen_type)] for x in tmp_stage],
                    di["financePresentValueInterestRate"][None],
                    [di["costGenerationDepreciationPeriod"][(x, gen_type)] for x in tmp_stage],
                    stg_op[0],
                    di["numberOfStages"][None],
                    di["yearsPerStage"][None],
                    False,
                )

                # print(stg_op, gen_type, investment_stages)
                # print(total_cost, decommissioning_redemption, operational_stage_contribution_matrix)

                if stg_op[0] == stg_op[1]:
                    di["multiPeriodCostGenerationTotalInvestment"][(stg_op[0], gen_type)] = total_cost
                else:
                    di["multiPeriodCostGenerationDecommissioning"][(stg_op[0], stg_op[1], gen_type)] = decommissioning_redemption[stg_op[1] - stg_op[0] - 1]

                for stg in tmp_stage:
                    # print((stg_op[0], stg_op[1], stg, gen_type), operational_stage_contribution_matrix[stg_op[1] - 1, stg - 1])
                    di["operationalStageContributionGeneration"][(stg_op[0], stg_op[1], stg, gen_type)] = operational_stage_contribution_matrix[stg_op[1] - 1, stg - 1]

                if stg_op[1] in investment_stages:
                    di["investmentStageGeneration"][(stg_op[0], stg_op[1], gen_type)] = 1
                else:
                    di["investmentStageGeneration"][(stg_op[0], stg_op[1], gen_type)] = 0

            for storage_type in di["STORAGE_TYPE"][None]:
                (total_cost, decommissioning_redemption, operational_stage_contribution_matrix, investment_stages) = self.get_multi_period_information(
                    [di["costStorageCapex"][(x, storage_type)] for x in tmp_stage],
                    [di["costStorageOpexFixed"][(x, storage_type)] for x in tmp_stage],
                    [di["costStorageInterestRate"][(x, storage_type)] for x in tmp_stage],
                    di["financePresentValueInterestRate"][None],
                    [di["costStorageDepreciationPeriod"][(x, storage_type)] for x in tmp_stage],
                    stg_op[0],
                    di["numberOfStages"][None],
                    di["yearsPerStage"][None],
                    False,
                )

                if stg_op[0] == stg_op[1]:
                    di["multiPeriodCostStorageTotalInvestment"][(stg_op[0], storage_type)] = total_cost
                else:
                    di["multiPeriodCostStorageDecommissioning"][(stg_op[0], stg_op[1], storage_type)] = decommissioning_redemption[stg_op[1] - stg_op[0] - 1]

                for stg in tmp_stage:
                    di["operationalStageContributionStorage"][(stg_op[0], stg_op[1], stg, storage_type)] = operational_stage_contribution_matrix[stg_op[1] - 1, stg - 1]

                if stg_op[1] in investment_stages:
                    di["investmentStageStorage"][(stg_op[0], stg_op[1], storage_type)] = 1
                else:
                    di["investmentStageStorage"][(stg_op[0], stg_op[1], storage_type)] = 0

        return {"emprise": di}  # see namespace definition in the create_concrete_model function

    def update_scenario_model_data(self, dict_data_sc=None, emprise_config=None, inv_sc=None):
        """Updates scenario-specific model data to represent stage-dependent uncertainty setup"""
        if dict_data_sc is None:
            raise ValueError("!!! EMPRISE update_scenario_model_data requires 'dict_data_sc' argument !!!")
        if emprise_config is None:
            raise ValueError("!!! EMPRISE update_scenario_model_data requires 'emprise_config' argument !!!")
        if inv_sc is None:
            raise ValueError("!!! EMPRISE update_scenario_model_data requires 'inv_sc' argument !!!")

        if emprise_config["uncertainty_emission_price_scenario"]:
            dict_data_sc["emprise"]["costSystemOperationEmissionPrice"] = {
                stage: emprise_config["uncertainty_emission_price_scenario"][inv_sc[stage - 1]][stage - 1] / 1000 / 1000 for stage in range(1, dict_data_sc["emprise"]["numberOfStages"][None] + 1)  # MEUR/tCO2eq
            }  # EUR/tCO2eq
        print("Emission scenario in MEUR/tCO2eq: ", dict_data_sc["emprise"]["costSystemOperationEmissionPrice"])  # MEUR/tCO2eq
        return dict_data_sc

    @staticmethod
    def get_capital_recovery_factor(interest_rate, n_periods):
        """Repeating payment factor (annuity)"""
        if interest_rate == 0.0:
            rpf = 1.0
        else:
            rpf = interest_rate * ((1 + interest_rate) ** n_periods) / (((1 + interest_rate) ** n_periods) - 1)
        return rpf

    @staticmethod
    def get_multi_period_information(
        capex: list,
        opex_fix: list,
        wacc: float,
        present_value_interest: float,
        depreciation_period: int,
        investment_period: int,
        number_of_investment_periods: int,
        investment_period_length: int,
        create_plot: bool = False,
    ):
        """Calculates multi-period specific investment cost and decommissioning redemption values

        For multi-period investment planning, this function calculates the total costs (discounted investment and fixed operation)
        starting from an investment period (assuming reinvestments after each depreciation period).
        The function further calculates redemption payments resulting from potential decommissioning decisions in the investment periods following the investment decision.

        Parameters
        ----------
        capex : list (of non-negative floats)
            The capital expenditures for all investment period (in specfic monetary unit, e.g. EUR/MW), len(capex) needs to equal number_of_investment_periods, e.g. [1000.0, 900.0, 750.0, 700.0]
        opex_fix : list (of non-negative floats)
            The fixed operational expenditures for all investment periods (in specfic monetary unit/time period unit, e.g. EUR/MW/yr), len(opex_fix) needs to equal number_of_investment_periods, e.g. [50.0, 50.0, 50.0, 50.0]
        wacc : float (in the interval [0,1])
            Weighted average cost of capital (in 1) to calculate equivalent annual cost, e.g. 0.06
        present_value_interest : float (in the interval [0,1])
            Present value interest rate (in 1) to calculate the discount factors representing the time value of money including the perpetuity, e.g. 0.02
        depreciation_period : float (non-negative)
            Depreciation period (in years) to calculate equivalent annual cost, e.g. 12
        investment_period : int (non-negative and smaller or equal to number_of_investment_periods)
            Investment period for which the first investment decision is assumed, e.g. 2
        number_of_investment_periods : int (non-negative)
            Total number of investment periods including perpetuity investment period, e.g. 4
        investment_period_length : int (non-negative)
            Length of the equal-length investment periods (in time units, e.g. years), e.g. 10
        create_plot : bool
            Plot total cost results over planning horizon

        Output
        ------
        total_cost : float
            Total (discounted) investment and fixed operation cost for investment decision in investment_period until the end of time (perpetuity in last considered period)
        decommissioning_redemption : list (of floats)
            Redemption payments for potential future decommissioning decisions starting after the investment_period to compensate the total cost paid for the entirety of the planning horizon
        operational_stage_contribution_matrix : two-dimensional numpy array (of floats)
            Share of technology development (investment) period (col) contribution for every considered operation period (row), including discount factor information
        investment_stages : list (of integers)
            Planning stages (i.e. periods) in which (re)investments take place depending on lengths of deprecation periods and overall planning horizon

        # CURRENTLY NOT USED: operational_stage_idx : two-dimensional numpy array (of floats)
            Share of technology development (investment) period (col) contribution for every considered operation period (row), excluding discount factor information. NOTE: Perpetuity currently counts as a single year

        Example
        -------
        get_multi_period_information([1000.0, 900.0, 750.0, 700.0], [50.0, 50.0, 50.0, 50.0], [0.06, 0.06, 0.06, 0.06], 0.02, [15, 13, 12, 10], 2, 4, 10, False)
        -> total_cost = 5951.8285
        -> decommissioning_redemption = [4591.0051, 3708.4673]
        -> operational_stage_contribution_matrix = [[0.         0.         0.         0.        ]
                                                    [0.         1.         0.         0.        ]
                                                    [0.         0.35372757 0.64627243 0.        ]
                                                    [0.         0.         0.0960792  0.9039208 ]]
        -> investment_stages = [2, 3, 4]
        """

        # print(capex, opex_fix, wacc, present_value_interest, depreciation_period, investment_period, number_of_investment_periods, investment_period_length, create_plot)
        # print(
        #     type(capex), type(opex_fix), type(wacc), type(present_value_interest), type(depreciation_period), type(investment_period), type(number_of_investment_periods), type(investment_period_length), type(create_plot)
        # )

        import numpy as np
        import math

        if capex is None:
            capex = [1.0 for i in range(number_of_investment_periods)]
        if opex_fix is None:
            opex_fix = [1.0 for i in range(number_of_investment_periods)]
        if wacc is None:
            wacc = [0.05 for i in range(number_of_investment_periods)]

        if create_plot:
            import matplotlib.pyplot as plt

        equivalent_annual_cost_factor = [wacc[x] / (1 - (1 + wacc[x]) ** -depreciation_period[x]) for x in range(len(capex))]
        perpetuity_factor = 1.0 / present_value_interest
        discount_factors = (1 - present_value_interest) ** np.arange(2 * (number_of_investment_periods) * investment_period_length)

        # Determine (re)investment periods
        start_index = [
            x for x in range(investment_period - 1, len(depreciation_period)) if 1 + sum(depreciation_period[investment_period - 1 : x]) <= investment_period_length * (number_of_investment_periods - investment_period)
        ]
        investment_start = [((investment_period - 1) * investment_period_length) + 1 + sum(depreciation_period[investment_period - 1 : x]) for x in start_index]
        if not start_index:  # list is empty (implicit booleanness)
            investment_start_perpetuity = ((investment_period - 1) * investment_period_length) + 1
        else:
            investment_start_perpetuity = ((investment_period - 1) * investment_period_length) + 1 + sum(depreciation_period[investment_period - 1 : start_index[-1] + 1])
        investment_start_all = investment_start + [investment_start_perpetuity]
        investment_stages = [math.floor(x / investment_period_length) + 1 for x in investment_start_all]

        # Retrieve investment and fixed operation cost
        investment_cost_matrix = np.zeros((len(investment_start) + 1, len(discount_factors)))
        fixed_operation_cost_matrix = np.zeros((len(investment_start) + 1, len(discount_factors)))
        operational_stage_matrix = np.zeros((len(investment_start) + 1, len(discount_factors)))
        operational_stage_idx_matrix = np.zeros((len(investment_start) + 1, len(discount_factors)))
        operational_stage_discount_factor_matrix = np.zeros((len(investment_start) + 1, len(discount_factors)))
        for inv in range(len(investment_start)):
            tmp_investment_period = math.floor(investment_start[inv] / investment_period_length)
            investment_cost_matrix[inv, (investment_start[inv] - 1) : (investment_start[inv] + depreciation_period[tmp_investment_period] - 1)] = (
                capex[tmp_investment_period]
                * equivalent_annual_cost_factor[tmp_investment_period]
                * discount_factors[(investment_start[inv] - 1) : (investment_start[inv] + depreciation_period[tmp_investment_period] - 1)]
            )
            fixed_operation_cost_matrix[inv, (investment_start[inv] - 1) : (investment_start[inv] + depreciation_period[tmp_investment_period] - 1)] = (
                opex_fix[tmp_investment_period] * discount_factors[(investment_start[inv] - 1) : (investment_start[inv] + depreciation_period[tmp_investment_period] - 1)]
            )
            operational_stage_matrix[inv, (investment_start[inv] - 1) : (investment_start[inv] + depreciation_period[tmp_investment_period] - 1)] = tmp_investment_period + 1
            operational_stage_discount_factor_matrix[inv, (investment_start[inv] - 1) : (investment_start[inv] + depreciation_period[tmp_investment_period] - 1)] = discount_factors[
                (investment_start[inv] - 1) : (investment_start[inv] + depreciation_period[tmp_investment_period] - 1)
            ]
            operational_stage_idx_matrix[inv, (investment_start[inv] - 1) : (investment_start[inv] + depreciation_period[tmp_investment_period] - 1)] = 1.0
            del tmp_investment_period
            if create_plot:
                plt.bar(range(len(discount_factors)), investment_cost_matrix[inv, :])
                plt.bar(range(len(discount_factors)), fixed_operation_cost_matrix[inv, :], bottom=investment_cost_matrix[inv, :])

        tmp_investment_period = math.floor(investment_start_perpetuity / investment_period_length)
        if tmp_investment_period > len(capex) - 1:
            tmp_investment_period = len(capex) - 1
        investment_cost_matrix[-1, investment_start_perpetuity - 1] = (
            capex[tmp_investment_period] * equivalent_annual_cost_factor[tmp_investment_period] * perpetuity_factor * discount_factors[investment_start_perpetuity - 1]
        )
        fixed_operation_cost_matrix[-1, investment_start_perpetuity - 1] = opex_fix[tmp_investment_period] * perpetuity_factor * discount_factors[investment_start_perpetuity - 1]
        operational_stage_matrix[-1, investment_start_perpetuity - 1] = tmp_investment_period + 1
        operational_stage_discount_factor_matrix[-1, investment_start_perpetuity - 1] = perpetuity_factor * discount_factors[investment_start_perpetuity - 1]
        operational_stage_idx_matrix[-1, investment_start_perpetuity - 1] = 1.0  # TODO: Needs to be corrected, current value does not make much sense
        del tmp_investment_period

        # Create plot if chosen by the user
        if create_plot:
            plt.bar(range(len(discount_factors)), investment_cost_matrix[-1, :])
            plt.bar(range(len(discount_factors)), fixed_operation_cost_matrix[-1, :], bottom=investment_cost_matrix[-1, :])
            plt.xlabel("Planning horizon")
            plt.ylabel("Cost in monetary unit/time unit")
            plt.title("Annual investment and fixed operation cost")
            plt.show()

        # Total cost as the sum of all (discounted) investment and fixed operation cost
        total_cost = sum(sum(investment_cost_matrix)) + sum(sum(fixed_operation_cost_matrix))

        # Compile decommissioning redemption values for each relevant decision period
        decommissioning_redemption_investment = []
        decommissioning_redemption_operation = []
        for ip in range(investment_period, number_of_investment_periods):
            idx1 = next(x for x, val in enumerate(investment_start_all) if val >= ip * investment_period_length)
            decommissioning_redemption_investment.append(sum(sum(investment_cost_matrix[idx1:, :])))  # Redemption payments from omitted reinvestments may only come from
            tmp_redemption_operation = [sum(sum(fixed_operation_cost_matrix[:, idx2 * investment_period_length : (idx2 + 1) * investment_period_length - 1])) for idx2 in range(ip, number_of_investment_periods)]
            decommissioning_redemption_operation.append(sum(tmp_redemption_operation))
            del tmp_redemption_operation, idx1

        # Compile weigths for operational stages
        operational_stage_contribution = np.zeros((number_of_investment_periods, number_of_investment_periods))
        operational_stage_idx = np.zeros((number_of_investment_periods, number_of_investment_periods))
        for ip in range(0, number_of_investment_periods):
            for idx3 in range(0, number_of_investment_periods):
                tmp_operational_stage_matrix = operational_stage_matrix[:, ip * investment_period_length : (ip + 1) * investment_period_length - 1].astype(int)
                tmp_operational_stage_discount_factor_matrix = operational_stage_discount_factor_matrix[:, ip * investment_period_length : (ip + 1) * investment_period_length - 1]
                tmp_operational_stage_idx_matrix = operational_stage_idx_matrix[:, ip * investment_period_length : (ip + 1) * investment_period_length - 1]
                operational_stage_contribution[ip, idx3] = tmp_operational_stage_discount_factor_matrix[tmp_operational_stage_matrix == idx3 + 1].sum()
                operational_stage_idx[ip, idx3] = tmp_operational_stage_idx_matrix[tmp_operational_stage_matrix == idx3 + 1].sum()
            if operational_stage_contribution[ip, :].sum() > 0.0:
                operational_stage_contribution[ip, :] = operational_stage_contribution[ip, :] / operational_stage_contribution[ip, :].sum()
                operational_stage_idx[ip, :] = operational_stage_idx[ip, :] / operational_stage_idx[ip, :].sum()

        # print(decommissioning_redemption_investment, decommissioning_redemption_operation)

        # Total decommissioning redemption payments
        decommissioning_redemption = [sum(x) for x in zip(decommissioning_redemption_investment, decommissioning_redemption_operation)]

        return (total_cost, decommissioning_redemption, operational_stage_contribution, investment_stages)  # operational_stage_idx currently not used

    @staticmethod
    def get_discount_factor_sum(interest_rate, stage, years_per_stage, allocation_method=""):
        """Calculates the stage-specific discount factor as a sum over all years belonging to the decision stage.
        The allocation method determines how the summation window is configured, i.e. does the given decision stage refer to the starting year or the centre of the years
        """
        import numpy as np

        relevant_years = np.arange((stage - 1) * years_per_stage, stage * years_per_stage)
        if allocation_method == "center":
            relevant_years = relevant_years - int(years_per_stage / 2)
        discount_factor = 1 / (1 + interest_rate) ** relevant_years
        # discount_factor[discount_factor > 1] = 1  # relevant for years before current year (years_per_stage/2), only for center method
        discount_factor_sum = sum(discount_factor)
        discount_factor_perpetuity = discount_factor[-1] / interest_rate
        return (discount_factor_sum, discount_factor_perpetuity)

    @staticmethod
    def parse_scenario_information(stage, node, n_stages, n_inv_sc, n_sysop_sc):
        """Get investment and system operation scenario information (given a balanced/symmetric branching structure)"""
        from math import ceil

        if stage > 1:
            if stage == n_stages:
                number_of_branches_per_node = n_sysop_sc
                sysop_sc = node - (ceil(node / number_of_branches_per_node) - 1) * number_of_branches_per_node
                inv_sc = 1  # arbitrary since not relevant for last stage of scenario tree
            else:
                number_of_branches_per_node = n_inv_sc * n_sysop_sc
                inv_sc_help = node - (ceil(node / number_of_branches_per_node) - 1) * number_of_branches_per_node
                sysop_sc = inv_sc_help - (ceil(inv_sc_help / n_sysop_sc) - 1) * n_sysop_sc
                inv_sc = ceil(inv_sc_help / n_sysop_sc)

            (sysop_sc_prev, inv_sc_prev) = EmpriseModel.parse_scenario_information(
                stage - 1,
                ceil(node / number_of_branches_per_node),
                n_stages,
                n_inv_sc,
                n_sysop_sc,
            )
            inv_sc = inv_sc_prev + [inv_sc]
            sysop_sc = sysop_sc_prev + [sysop_sc]
        else:
            sysop_sc = [1]
            inv_sc = [1]
        return (sysop_sc, inv_sc)

    @staticmethod
    def truncate(x, precision=4):
        return x  # int(x * 10 ** precision) / 10 ** precision


# Confirm pyomo version that is used (especially relevant in distributed environments)
print("\n### Using python version {} ###".format(sys.version))
print("### Using pyomo version {} ###\n".format(pyomo.__version__))


# =============================================================================
def make_nodes_for_scenario(model, branching_factors, scen_num, scen_prob, exclude_component_list=[]):
    """Make just those scenario tree nodes needed by a scenario.
    Return them as a list.
    NOTE: the nodes depend on the scenario model and are, in some sense,
          local to it.
    Args:
        model (EmpriseModel): Concrete EmpriseModel instance
        branching_factors (list of int): branching factors
        scen_num (int): Scenario number
        scen_prob (list of floats): Stage-specific conditional probabilities
        exclude_component_list (list of strings): EMPRISE modules not to be included
    Output: ret_val (list of ScenarioNodes): Scenario tree nodes and their relationship for the
        given scenario

    """
    from functools import reduce

    ret_val = []
    branching_factors_ = branching_factors + [1]
    denominators = [reduce(lambda x, y: x * y, branching_factors_[stg:]) for stg in range(len(branching_factors_))]
    for stg in range(len(branching_factors) + 1):  # Leaf node stage is also accessed!
        if stg == 0:
            var_list = [model.generatorThermalCapacity[:, :, stg + 1], model.generatorRenewableCapacity[:, :, stg + 1]]
            if "electricity_storage" not in exclude_component_list:
                var_list.append(model.electricityStorageCapacity[:, :, stg + 1])
            node_no_last_stage = [0]
            current_node = "ROOT"
            ret_val.append(scenario_tree.ScenarioNode(current_node, scen_prob[stg], stg + 1, model.costTotal[stg + 1], None, var_list, model))
            current_parent = current_node
        else:  # intermediate node
            if stg == len(branching_factors):  # leaf node (child)
                var_list = []
            else:
                var_list = [model.generatorThermalCapacity[:, :, stg + 1], model.generatorRenewableCapacity[:, :, stg + 1]]
                if "electricity_storage" not in exclude_component_list:
                    var_list.append(model.electricityStorageCapacity[:, :, stg + 1])

                node_no_last_stage.append(int((scen_num - 1 - sum([node_no_last_stage[stg_] * denominators[stg_] for stg_ in range(stg)])) / denominators[stg]))
                current_node = current_node + "_" + str(node_no_last_stage[-1])
                ret_val.append(
                    scenario_tree.ScenarioNode(
                        current_node,
                        scen_prob[stg],
                        stg + 1,
                        model.costTotal[stg + 1],
                        None,
                        var_list,
                        model,
                        parent_name=current_parent,
                    )
                )
                current_parent = current_node

    return ret_val


# =============================================================================
def scenario_creator(scenario_name, branching_factors=None, emprise_config=None, data_path=None):
    """The callback needs to create an instance and then attach
    the PySP nodes to it in a list _mpisppy_node_list ordered by stages.
    Optionally attach _PHrho.
    Args:
        scenario_name (str): root name of the scenario data file
        branching_factors (list of ints): the branching factors
        emprise_config (dict of various data types): configuration data for creating the multi-stage stochastic EMPRISE instances (e.g. number of stage, input data file paths)
        data_path (str, optional): Path to the Hydro data.
    """

    from functools import reduce

    if branching_factors is None:
        raise ValueError("!!! EMPRISE scenario_creator requires 'branching_factors' argument !!!")

    if emprise_config is None:
        raise ValueError("!!! EMPRISE scenario_creator requires 'emprise_config' argument !!!")

    print("### Building of EMPRISE instance for scenario '{}' started! ###".format(scenario_name))

    # Obtain scenario-specific information for strategic and operational uncertainty scenarios
    sc_num = sputils.extract_num(scenario_name)  # e.g. 1 from "Scen1"
    (sysop_sc, inv_sc) = EmpriseModel.parse_scenario_information(
        emprise_config["number_of_stages"] + 1,  # Actual stages are planning periods + 1, Focus only on full number of stages (differs from node-specific calls in previous PySP implementation)
        sc_num,
        emprise_config["number_of_stages"] + 1,  # Actual stages are planning periods + 1
        emprise_config["number_of_investment_scenarios"],
        emprise_config["number_of_system_operation_scenarios"],
    )

    scenario_specific_reference_years = [emprise_config["reference_years"][sc - 1] for sc in sysop_sc[1:]]

    for stg in range(len(branching_factors) + 1):
        if stg == 0:  # root node probability
            scenario_specific_probability = [1.0]
        elif stg < len(branching_factors):  # intermediate node probability
            scenario_specific_probability.append(emprise_config["scenario_probability"]["inv"][stg + 1][inv_sc[stg] - 1] * emprise_config["scenario_probability"]["sysop"][stg][sysop_sc[stg] - 1])
        else:  # leaf node probability
            scenario_specific_probability.append(emprise_config["scenario_probability"]["sysop"][stg][sysop_sc[stg] - 1])

    # Create scenario-specific EMPRISE input data object
    from . import InputData

    scenario_input_data = InputData(emprise_config["number_of_stages"], scenario_specific_reference_years)
    scenario_input_data.read_structural_data(file_path=emprise_config["file_names_structural"])

    scenario_input_data.read_time_series_data(
        file_names=emprise_config["file_names_timeseries"],
        n_header_rows=emprise_config["n_header_rows_timeseries"],
        reference_years=scenario_specific_reference_years,
        timerange=range(emprise_config["sample_offset"], emprise_config["sample_size"] + emprise_config["sample_offset"]),
        timedelta=1.0,
    )

    # Create abstract EMPRISE model
    emprise_model = EmpriseModel(exclude_component_list=emprise_config["exclude_component_list"])

    # Create scenario-specific EMPRISE dictionary data
    scenario_dict_data = emprise_model.create_model_data(scenario_input_data)
    scenario_dict_data = emprise_model.update_scenario_model_data(scenario_dict_data, emprise_config, inv_sc)

    # Instantiate scenario-specific concrete EMPRISE model
    instance = emprise_model.create_concrete_model(dict_data=scenario_dict_data)
    instance.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)

    # # Solve individual instance (only used for local debugging)
    # solver = pyo.SolverFactory("cplex")
    # results = solver.solve(instance, tee=True)

    instance._mpisppy_node_list = make_nodes_for_scenario(instance, branching_factors, sc_num, scenario_specific_probability, emprise_config["exclude_component_list"])

    instance._mpisppy_probability = reduce(lambda x, y: x * y, scenario_specific_probability)

    # if sc_num == 1:
    #     import sys
    #     import os

    #     f = open(os.path.join(emprise_config["path_result_dir"], scenario_name + "_instance.txt"), "w")
    #     sys.stdout = f
    #     instance.pprint()
    #     f.close()

    # print(scenario_name, [i.name for i in instance._mpisppy_node_list], scenario_specific_probability, instance._mpisppy_probability)

    print("### Building of EMPRISE instance for scenario '{}' finished! ###".format(scenario_name))

    return instance


# =============================================================================
def scenario_denouement(rank, scenario_name, scenario):
    pass


# if __name__ == "__main__":
