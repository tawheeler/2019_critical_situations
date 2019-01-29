using AutomotiveDrivingModels
using AutoViz
using AutoScenes
using Records
using Parameters
using HDF5, JLD
using Mocha
import StatsBase: sample

include("AutoMobius/AutoMobius.jl")
using AutoMobius

include("crit_cluster_sampling/sample_factor_graph.jl")
include("safety_validation/safety_system.jl")

# const DATA_DIR = "/media/tim/DATAPART1/PublicationData/2017_adas_validation/"
const DATA_DIR = "/media/tim/Tim 1500 GB/2017_adas_validation/"

const TIMESTEP = 0.1
const BURNIN = 1000
const SAFETY_SYSTEM = true

const RISK_THRESHOLD = 0.1

const TOT_SIM_TIME = 1e9 # [s] # elapsed time: 25845.280343621 seconds
const TOT_SIM_TICKS = round(Int, TOT_SIM_TIME/TIMESTEP)
const MAX_SIM_TICKS_PER_SIM = 1000000

####

include("risk_estimator.jl")

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

mutable struct AEBActivationCountCallback
    ego_id::Int
    count::Int
end
AEBActivationCountCallback(ego_id::Int) = AEBActivationCountCallback(ego_id, 0)
function AutomotiveDrivingModels.run_callback{S,Def,I,D<:DriverModel}(
    callback::AEBActivationCountCallback,
    rec::QueueRecord{Entity{S,Def,I}},
    roadway::MobiusRoadway,
    models::Dict{I,D},
    tick::Int,
    )

    callback.count += (models[callback.ego_id].ticks_since_activation == 1)

    return false
end

####

srand(RUNNUM)
scene = MobiusScene()
ngsim_scenes = open(joinpath(DATA_DIR, "scenes_NGSIM.txt")) do io
     read(io, MIME"text/plain"(), Vector{MobiusScene})
end
filter!(s->length(s) ≤ MAX_N_CARS, ngsim_scenes)

roadway = Wraparound(Straight1DRoadway(200.0))
models = Dict{Int, LaneFollowingDriver}()
behgen = ErrorableIDMBehaviorGenerator(TIMESTEP)
rec = QueueRecord(MobiusEntity, 2, TIMESTEP)
driverstate_rec = QueueRecord(AttentiveEntity, 2, TIMESTEP)

aeb_activation_callback = AEBActivationCountCallback(0)
activated_aeb_count = 0
collision_count = 0
sim_count = 0
tick_count = 0
sim_tick_counter = SimTickCounter(0)

outfile = SAFETY_SYSTEM ? "mc_eval_$RUNNUM.txt" : "mc_eval_no_safety_sys_$RUNNUM.txt"

t_start = now()
tic()
io = open(joinpath(DATA_DIR, outfile), "w")
    println(io, "burnin: ", BURNIN)
    println(io, "model: ", "NONE")
    println(io, "timestep: ", TIMESTEP)
    println(io, "tot_sim_time: ", TOT_SIM_TIME)
    println(io, "tot_sim_ticks: ", TOT_SIM_TICKS)
    println(io, "max_sim_ticks_per_sim: ", MAX_SIM_TICKS_PER_SIM)
    while tick_count < TOT_SIM_TICKS

        sim_count += 1
        println("sim ", sim_count)
        println(io, "sim ", sim_count)

        copy!(scene, sample(ngsim_scenes)) # draw scene from NGSIM scenes
        rand!(models, behgen, scene) # assign models (draw IDM, initialize attentiveness)

        # run sim for burnin period
        # break if we exceed criticality threshold
        empty!(rec)
        reset_hidden_states!(models)
        sim_tick_counter.tick = 0
        callbacks = (AutoMobius.MobiusCollisionCallback(), sim_tick_counter, risk_est_callback)
        simulate!(rec, scene, roadway, models, BURNIN, callbacks)
        if sim_tick_counter.tick == BURNIN

            # now the sim actually starts

            # assign one car to be ego
            ego_index = rand(1:length(scene))
            ego_id = scene[ego_index].id
            models2 = Dict{Int,DriverModel}()
            for (id,val) in models
                models2[id] = val
            end
            if SAFETY_SYSTEM
                models2[ego_id] = VolvoAEB(WrappedLaneFollowingDriver{StoppingAccel,typeof(models[ego_id])}(models[ego_id]))
            end

            empty!(rec)
            if SAFETY_SYSTEM
                aeb_activation_callback.count = 0
                aeb_activation_callback.ego_id = ego_id
                rec, driverstate_rec, nticks = simulate_until_collision(scene, roadway, models2, rec, driverstate_rec, (aeb_activation_callback,), nticks=MAX_SIM_TICKS_PER_SIM)
            else
                rec, driverstate_rec, nticks = simulate_until_collision(scene, roadway, models2, rec, driverstate_rec, nticks=MAX_SIM_TICKS_PER_SIM)
            end

            had_collision = run_callback(MobiusCollisionCallback(), rec, roadway, models2, 1)
            collision_count += had_collision
            if SAFETY_SYSTEM
                activated_aeb_count += aeb_activation_callback.count
            end
            tick_count += nticks
            if had_collision
                final_scene = rec[0]
                i, j = get_first_collision(final_scene, roadway)
                deltav = abs(final_scene[i].state.v - final_scene[j].state.v) # [m/s]
                is_ego_a = ego_id == final_scene[i].id
                is_ego_b = ego_id == final_scene[j].id
                str = "\tcollision Δv  $deltav  $(is_ego_a ? "ego" : "oth")  $(is_ego_b ? "ego" : "oth")"
                println(str)
                println(io, str)
            end

            elapsed_time = now() - t_start
            println("\ttick_count ", tick_count, "  ", round(tick_count/TOT_SIM_TICKS*100, 3), "%")
            println("\tcollision_count ", collision_count)
            println("\tactivated_aeb_count ", activated_aeb_count)
            println("\telapsed time ", elapsed_time)
            println("\testimated remaining ", Dates.Millisecond(round(Int, elapsed_time.value/tick_count * (TOT_SIM_TICKS - tick_count))))
            println(io, "\ttick_count ", tick_count)
            println(io, "\tcollision_count ", collision_count)
            println(io, "\tactivated_aeb_count ", activated_aeb_count)
            println(io, "\telapsed time ", elapsed_time)
        end
    end
close(io)
toc()

println("DONE")