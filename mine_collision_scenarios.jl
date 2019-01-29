using AutomotiveDrivingModels
using AutoViz
using AutoScenes
using Records
using Parameters
using Mocha
using HDF5, JLD
import StatsBase: sample
import DataStructures: CircularBuffer

include("AutoMobius/AutoMobius.jl")
using AutoMobius

#=
Number of ticks in which no collision may occur and
at the end of which criticality must be below criticality threshold
=#
const TIMESTEP = 0.1
const BURNIN = 1000
const RISK_THRESHOLD = 0.15
const MAX_COLLISION_COUNT = 10 # number of collisions to harvest
const BUFFER_TICK_COUNT = 50 # number of additional ticks to store before criticality transition
const N_BENIGN_SCENES = 2000 # number of benign scenes to keep (uniformly, via resevoir sampling)
const MAX_SIM_COUNT = typemax(Int)
const MAX_TOTAL_TICK_COUNT = typemax(Int)
const MAX_TICK_COUNT_PER_SIM = 1000000
const MAX_RECORD_HISTORY = 1000 # max length a critical scenario record can be (this is much bigger than it needs to be, on purpose)

const DATA_DIR = "/media/tim/DATAPART1/PublicationData/2017_adas_validation/"
const OUTPUT_FILE_SUFFIX = "1019"
const COLLISION_SCENARIO_DIR = joinpath(DATA_DIR, "collision_scenarios_$OUTPUT_FILE_SUFFIX")
isdir(COLLISION_SCENARIO_DIR) || mkdir(COLLISION_SCENARIO_DIR)

include("risk_estimator.jl")

###################################################
#               NORMAL
###################################################

scene = MobiusScene()
ngsim_scenes = open(joinpath(DATA_DIR, "scenes_NGSIM.txt")) do io
     read(io, MIME"text/plain"(), Vector{MobiusScene})
end
filter!(s->length(s) ≤ MAX_N_CARS, ngsim_scenes)

roadway = Wraparound(Straight1DRoadway(200.0))
models = Dict{Int, LaneFollowingDriver}()
behgen = ErrorableIDMBehaviorGenerator(TIMESTEP)
rec = QueueRecord(MobiusEntity, 2, TIMESTEP)
driverstate_rec = QueueRecord(AttentiveEntity, MAX_RECORD_HISTORY, TIMESTEP)

"""
Used so that other things have access to the current risk
"""
mutable struct CurrentRisk
    r::Float64
end
const CURRENT_RISK = CurrentRisk(NaN)

mutable struct CurrentTick
    i::Int
end
const CURRENT_TICK = CurrentTick(0)

"""
    Risk Estimation Callback

Halts the simulation once the estimated risk exceeds threshold
"""
struct RiskEstCallback
    est::RiskEstimator
    current_risk::CurrentRisk
    risk_threshold::Float64
end
function AutomotiveDrivingModels.run_callback{S,Def,I,D<:DriverModel}(
    callback::RiskEstCallback,
    rec::QueueRecord{Entity{S,Def,I}},
    roadway::MobiusRoadway,
    models::Dict{I,D},
    tick::Int,
    )

    callback.current_risk.r = r = predict_risk(callback.est, rec[0], models, roadway)
    return r > callback.risk_threshold # stop if we think it will lead to a collision
end
risk_est_callback = RiskEstCallback(
                        RISK_ESTIMATOR,
                        CURRENT_RISK,
                        RISK_THRESHOLD,
                    )

#=
Risks contains the risk for scenes vs. time.
Very simple file - just two columns with estimated risk
Appends r for every tick after burnout
=#
mutable struct RiskLoggerCallback
    io::IO
    est::RiskEstimator
    risk_threshold::Float64
    current_risk::CurrentRisk
    current_tick::CurrentTick
end
function AutomotiveDrivingModels.run_callback{S,Def,I,D<:DriverModel}(
    callback::RiskLoggerCallback,
    rec::QueueRecord{Entity{S,Def,I}},
    roadway::MobiusRoadway,
    models::Dict{I,D},
    tick::Int,
    )

    callback.current_risk.r = predict_risk(callback.est, rec[0], models, roadway)
    write(callback.io, convert(Float32, callback.current_risk.r))
    callback.current_tick.i += 1
    return false # never halt the simulation
end
risk_logger_callback = RiskLoggerCallback(
                        open(joinpath(COLLISION_SCENARIO_DIR, "risks.dat"), "w"),
                        RISK_ESTIMATOR,
                        RISK_THRESHOLD,
                        CURRENT_RISK,
                        CURRENT_TICK,
                       )

#=
Uses resevoir sampling to pull benign scenes
Assumes that current_risk is updated by another callback before this one
Assumes that driverstate_rec is updated by another callback before this one
Assumes that tick_source is updated to provide the risk estimation tick
=#
mutable struct BenignSceneSamplingCallback
    current_risk::CurrentRisk
    current_tick::CurrentTick
    risk_threshold::Float64
    benign_scenes::Vector{AttentiveMobiusScene}
    scene_ticks::Vector{Int}
    resevoir_sampling_count::Int
end
function AutomotiveDrivingModels.run_callback{S,Def,I,D<:DriverModel}(
    callback::BenignSceneSamplingCallback,
    rec::QueueRecord{Entity{S,Def,I}},
    roadway::MobiusRoadway,
    models::Dict{I,D},
    tick::Int,
    )

    risk = callback.current_risk.r
    if risk < callback.risk_threshold
        callback.resevoir_sampling_count += 1

        r = callback.resevoir_sampling_count ≤ length(callback.benign_scenes) ?
                callback.resevoir_sampling_count :
                rand(1:callback.resevoir_sampling_count)

        if r ≤ length(callback.benign_scenes)
            callback.benign_scenes[r] = pack_attentive_mobius_frame(rec[0], models)
            callback.scene_ticks[r] = callback.current_tick.i
        end
    end
    return false # never halt the simulation
end
benign_scene_sampling_callback = BenignSceneSamplingCallback(
                                    CURRENT_RISK,
                                    CURRENT_TICK,
                                    RISK_THRESHOLD,
                                    Array{AttentiveMobiusScene}(N_BENIGN_SCENES),
                                    Array{Int}(N_BENIGN_SCENES),
                                    0,
                                )

struct CriticalScenarioCallback
    current_risk::CurrentRisk
    current_tick::CurrentTick
    risk_threshold::Float64
    rec::QueueRecord{AttentiveMobiusEntity}
    risk_log::CircularBuffer{Float64}
    buffer_tick_count::Int # number of additional ticks before crossing risk threshold to store
end
function export_latest_crit_scenario(callback::CriticalScenarioCallback)
    i = findprev(x->x < callback.risk_threshold, callback.risk_log, length(callback.risk_log)-1)
    if i == 0
        i = 1
    end

    i = max(1, i - callback.buffer_tick_count)

    record_length = length(callback.risk_log) - i + 1 # number of frames to log
    start_of_record = callback.current_tick.i - record_length + 1 # tick corresponding to start of record

    @assert callback.rec.nframes ≥ record_length
    callback.rec.nframes = record_length # drop the old frames
    output_record = convert(ListRecord{AttentiveMobiusState, AttentiveMobiusDef, Int}, callback.rec)

    filepath = joinpath(COLLISION_SCENARIO_DIR, "collision_listrecord_tick_$(start_of_record)_datetime_" * Dates.format(now(), "YYYYmmdd_HHMMSS_sss") * ".txt")
    println("saving critical trace...")
    open(filepath, "w") do io
        write(io, MIME"text/plain"(), output_record)
    end
    println("\tdone")
end
function AutomotiveDrivingModels.run_callback{S,Def,I,D<:DriverModel}(
    callback::CriticalScenarioCallback,
    rec::QueueRecord{Entity{S,Def,I}},
    roadway::MobiusRoadway,
    models::Dict{I,D},
    tick::Int,
    )

    update!(callback.rec, pack_attentive_mobius_frame(rec[0], models))

    risk = callback.current_risk.r
    push!(callback.risk_log, risk)

    if length(callback.risk_log) > 1 && callback.risk_log[end-1] > callback.risk_threshold && callback.risk_log[end] < callback.risk_threshold
        # went from critical to benign once more
        # export to disk

        println("risk went below! ", callback.risk_log[end-1:end])
        export_latest_crit_scenario(callback)
    end
    return false # never halt the simulation
end
crit_scenario_callback = CriticalScenarioCallback(
                            CURRENT_RISK,
                            CURRENT_TICK,
                            RISK_THRESHOLD,
                            QueueRecord(AttentiveMobiusEntity, MAX_RECORD_HISTORY, TIMESTEP),
                            CircularBuffer{Float64}(MAX_RECORD_HISTORY),
                            BUFFER_TICK_COUNT,
                         )


collision_count = 0
tick_count = 0
sim_count = 0

sim_tick_counter = SimTickCounter(0)

srand(0)
tic()
while collision_count < MAX_COLLISION_COUNT && tick_count < MAX_TOTAL_TICK_COUNT && sim_count < MAX_SIM_COUNT

    sim_count += 1
    println("sim ", sim_count)
    tic()

    copy!(scene, sample(ngsim_scenes)) # draw scene from NGSIM scenes
    rand!(models, behgen, scene) # assign models (draw IDM, initialize attentiveness)

    # run sim for burnin period
    # break if we exceed criticality threshold
    empty!(rec)
    reset_hidden_states!(models)
    sim_tick_counter.tick = 0
    callbacks = (MobiusCollisionCallback(), sim_tick_counter, risk_est_callback)
    simulate!(rec, scene, roadway, models, BURNIN, callbacks)
    if sim_tick_counter.tick == BURNIN

        # run sim until collision
        # log the estimated risk at each step
        # store some of the benign frames
        # store critical traces + 1s lead time

        copy!(scene, rec[0])
        rec, driverstate_rec, nticks = simulate_until_collision(scene, roadway, models, rec, driverstate_rec,
                                                                (risk_logger_callback,benign_scene_sampling_callback,crit_scenario_callback),
                                                                nticks=MAX_TICK_COUNT_PER_SIM)

        had_collision = run_callback(MobiusCollisionCallback(), rec, roadway, models, 1)
        collision_count += had_collision
        tick_count += nticks

        println("\ttick_count ", tick_count)
        println("\tcollision_count ", collision_count)
    end
    toc()
end
print("total "); toc()

# output benign scenes
tic()
println("SAVING BENIGN SCENE DATA")
benign_scene_range = 1:min(length(benign_scene_sampling_callback.benign_scenes),
                           benign_scene_sampling_callback.resevoir_sampling_count)
open(joinpath(COLLISION_SCENARIO_DIR, "benign_scenes.txt"), "w") do io
    write(io, MIME"text/plain"(), benign_scene_sampling_callback.benign_scenes[1:maximum(benign_scene_range)])
end
open(joinpath(COLLISION_SCENARIO_DIR, "benign_scene_ticks.txt"), "w") do io
    for i in benign_scene_range
        println(io, benign_scene_sampling_callback.scene_ticks[i])
    end
end
println("[DONE]")
toc()

# open(joinpath(output_dir, "general_stats.txt"), "w") do io
#     println(io, "burnin: ", burnin)
#     println(io, "timestep: ", timestep)
#     println(io, "max_tot_collisions: ", max_tot_collisions)
#     println(io, "max_sim_ticks_per_sim: ", max_sim_ticks_per_sim)
#     println(io, "record_history: ", record_history)
#     println(io, "collision_count: ", collision_count)
#     println(io, "frame_count: ", frame_count)
#     println(io, "sim_count: ", sim_count)
# end


close(risk_logger_callback.io)

println("DONE")