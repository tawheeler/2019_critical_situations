module AutoMobius

using AutomotiveDrivingModels
# using AutoRisk
using AutoViz
using Records
using BayesNets
using Discretizers
using StatsBase
using Parameters
using OnlineStats
using HDF5, JLD

export
    MobiusEntity,
    MobiusScene,
    MobiusRoadway,

    BayesNets1DSceneModel,
    discretize_dataset,
    sample_continuous_value,

    IDM_AGGRESSIVE,
    IDM_TIMID,
    CorrelatedIDMBehaviorGenerator,
    UncorrelatedIDMBehaviorGenerator,
    get_correlated_IDM,
    get_timidness,

    MobiusCollisionCallback,
    SimTickCounter,
    DriverstateRecCallback,
    simulate_until_collision,
    does_sim_produce_collision,
    get_collision_estimation,

    AttentiveDriverState,
    AttentiveLaneFollowingDriver,
    ThresholdedRangeRatePerceptionState,
    ThresholdedRangeRatePerceptionLaneFollowingDriver,
    AttentionOverlay,

    DelayedDriverState,
    DelayedDriver,

    IDMDef,
    PackedListRecord,
    pack_output_list_record,
    pack_attentive_mobius_frame,
    unpack!,

    ErrorableIDMBehaviorGenerator,

    AttentiveEntity,
    AttentiveMobiusState,
    AttentiveMobiusDef,
    AttentiveMobiusEntity,
    AttentiveMobiusScene,
    get_attentive_driver_state,
    pull_driverstate_frame,
    set_driverstate_frame!,

    export_betas_array,
    import_betas_array,

    assign_blame,
    findfirst_frame_above_risk_threshold,
    unpack_critical_scene!

const MobiusEntity = Entity{PosSpeed1D, BoundingBoxDef, Int}
const MobiusScene = EntityFrame{PosSpeed1D, BoundingBoxDef, Int}
const MobiusRoadway = Wraparound{Straight1DRoadway}
MobiusScene(n::Int=100) = Frame(MobiusEntity, n)

include("mobius_bayes_net_scene_model.jl")
include("idm_behavior_generator.jl")
include("errorable_drivers.jl")
include("delayed_drivers.jl")
include("mobius_simulation.jl")
include("export.jl")
include("risk_evaluation.jl")
include("errorable_idm_behavior_generator.jl")
include("critical_cars.jl")


end