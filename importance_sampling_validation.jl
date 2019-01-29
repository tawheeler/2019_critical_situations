using AutomotiveDrivingModels
using AutoViz
using AutoScenes
using Records
using Parameters
using Distributions
using HDF5, JLD
import StatsBase: sample

include("AutoMobius/AutoMobius.jl")
using AutoMobius

include("clustering/clustering.jl")
include("crit_cluster_sampling/sample_factor_graph_with_criticality.jl")
include("safety_validation/safety_system.jl")

dicts = load_collision_scenarios()
crit_entries = load_crit_entries()
clusters = map_cluster_names_to_indices(crit_entries)

for (s,inds) in clusters
    println(s, " => ", length(inds), " critical entries")
end

const RISK_THRESOLD = 0.5
crit_scenes, crit_models = load_crit_scenes_and_models(crit_entries, dicts, risk_threshold=RISK_THRESOLD)
benign_scenes = load_benign_scenes()

speed_lo = Inf
speed_hi = -Inf
for scene in vcat(collect(values(crit_scenes)), benign_scenes)
    for veh in scene
        speed_lo = min(speed_lo, veh.state.v)
        speed_hi = max(speed_hi, veh.state.v)
    end
end
@show speed_lo
@show speed_hi

### Load Factor Graphs

const TOTAL_FRAMECOUNTS = 420121939
const MAX_SIM_FRAME_BUFFER = 10

clusterids = ["R1", "LR1"]
factormodels = Dict{String, FactorModel}()
cluster_keys = Dict{String, Vector{String}}()
cluster_factorgraphs = Dict{String, Vector{FactorGraph}}()
cluster_framecounts = Dict{String, Int}()
cluster_max_sim_frames = Dict{String, Int}()
for clusterid in clusterids
    model = @AutoScenes.load_factor_model("data/1d_factorgraph_model_with_criticality_$clusterid.txt")
    model = create_penalty_model_from_global(model)
    factormodels[clusterid] = model
end
for clusterid in clusterids
    cluster_keys[clusterid] = [entry.key for entry in crit_entries if entry.assignment == clusterid]

    features = factormodels[clusterid].features
    cluster_factorgraphs[clusterid] = [get_factorgraph(crit_scenes[k], roadway, get_crit_cars(dicts, k), features) for k in cluster_keys[clusterid]]

    cluster_framecounts[clusterid] = 0
    cluster_max_sim_frames[clusterid] = 0
    for key in cluster_keys[clusterid]
        betas = dicts.risk_ests[key]
        start_index = findlast(β -> mean(β) ≥ RISK_THRESOLD, betas)
        n_crit_frames = length(betas) - start_index + 1
        cluster_framecounts[clusterid] += n_crit_frames
        cluster_max_sim_frames[clusterid] = max(cluster_max_sim_frames[clusterid], n_crit_frames)
    end
    cluster_max_sim_frames[clusterid] += MAX_SIM_FRAME_BUFFER
end
let
    model = @AutoScenes.load_factor_model("data/1d_factorgraph_model_NGSIM_0804.txt")
    model = create_penalty_model_from_global(model)
    factormodels["benign"] = model
    push!(clusterids, "benign")

    cluster_factorgraphs["benign"] = [get_factorgraph(scene, roadway, Int[], model.features) for scene in benign_scenes]
    cluster_framecounts["benign"] = TOTAL_FRAMECOUNTS - sum(values(cluster_framecounts))
    cluster_max_sim_frames["benign"] = 1000000
end

# TODO: store this in a file and then load it
proposal_probs = Dict{String,Float64}(
        "R1" => 0.45,
        "LR1" => 0.45,
        "benign" => 0.10,
    )

P = Categorical([cluster_framecounts[id] for id in clusterids]./TOTAL_FRAMECOUNTS) # true distribution
Q = Categorical([proposal_probs[id] for id in clusterids]) # proposal distribution

@show P
@show Q
@show cluster_max_sim_frames

T = Dict(:s => Normal(0.0,5.0), :v => Normal(0.0,1.0))
burnin = 1000
gens = Dict{String, FactorGraphSceneGenerator}()
for clusterid in clusterids
    gens[clusterid] = FactorGraphSceneGenerator(factormodels[clusterid], T, burnin)
end

####

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

srand(0)

scene = MobiusScene()
timestep = 0.1
roadway = Wraparound(Straight1DRoadway(200.0))
behgen = ErrorableIDMBehaviorGenerator(timestep)

max_sims = Inf
tot_sim_time = 1e8 #1e9 # [s]
tot_sim_ticks = round(Int, tot_sim_time/timestep)

record_history = 2
rec = QueueRecord(MobiusEntity, record_history, timestep)
driverstate_rec = QueueRecord(AttentiveEntity, record_history, timestep)

# ttc_crit = Float64[]

col_count = 0
sim_count = 0
tick_count = 0
t_start = now()
safety_system = true
tic()
io = open(safety_system ? "data/is_eval.txt" : "data/is_eval_no_safety_sys.txt", "w")
    println(io, "burnin: ", burnin)
    println(io, "T: ", T)
    println(io, "timestep: ", timestep)
    println(io, "tot_sim_time: ", tot_sim_time)
    println(io, "tot_sim_ticks: ", tot_sim_ticks)
    println(io, "clusters: ", length(clusterids))
    for clusterid in clusterids
        println(io, "\t", clusterid)
        println(io, "\tframe_count: ", cluster_framecounts[clusterid])
        println(io, "\tproposal_prob: ", proposal_probs[clusterid])
        println(io, "\tcluster_max_sim_frames: ", cluster_max_sim_frames[clusterid])
    end

    while tick_count < tot_sim_ticks && sim_count < max_sims

        sim_count += 1
        println("sim ", sim_count)
        println(io, "sim ", sim_count)

        # sample new start scene
        clusterid = clusterids[rand(Q)]
        # clusterid = "R1"
        # clusterid = "LR1"
        # clusterid = "benign"
        println("\t", clusterid)

        models = Dict{Int, LaneFollowingDriver}()
        ego_index = 0
        max_sim_ticks = 100 # cuz is this what was used during risk est (10s)
        if clusterid != "benign"
            j = rand(1:length(cluster_keys[clusterid]))
            key = cluster_keys[clusterid][j]
            copy!(scene, crit_scenes[key])
            for (id, val) in crit_models[key]
                models[id] = deepcopy(val)
            end
            ego_index = dicts.crit_rears[key]
            factorgraph = cluster_factorgraphs[clusterid][j]
        else
            j = rand(1:length(benign_scenes))
            copy!(scene, benign_scenes[j])
            rand!(models, behgen, scene)
            max_sim_ticks = cluster_max_sim_frames[clusterid]
            ego_index = rand(1:length(scene))
            factorgraph = cluster_factorgraphs[clusterid][j]
        end
        Δ = metropolis_hastings!(gens[clusterid], factorgraph)
        # fill!(Δ, 0.0) # DEBUG
        sampled_scene = shift_scene!(MobiusScene(), scene, Δ, factorgraph)

        # assign one car to be ego
        if clusterid != "benign"
            # neighbor_fore = get_neighbor_fore(sampled_scene, ego_index, roadway)
            # oth_index = neighbor_fore.ind
            # headway = get_headway(scene[ego_index], scene[oth_index], roadway)
            # v_ego = scene[ego_index].state.v
            # v_oth = scene[oth_index].state.v
            # deltav = v_oth - v_ego
            # ttc = (deltav < 0 ? min(-headway / deltav, 30.0) : 40.0)
            # println("ego_attentiveness: ", models[ego_index].state.is_attentive)
            # println("v_ego: ", v_ego)
            # println("v_oth: ", v_oth)
            # println("deltav: ", deltav)
            # println("headway: ", headway)
            # println("ttc: ", ttc)
            # push!(ttc_crit, ttc)
        end
        ego_id = sampled_scene[ego_index].id

        models2 = Dict{Int, DriverModel}()
        for (id, m) in models
            models2[id] = m
        end
        if safety_system
            models2[ego_id] = VolvoAEB(human=WrappedLaneFollowingDriver{StoppingAccel,typeof(models[ego_id])}(models2[ego_id]))
        end

        # @show ego_id
        # for veh in sampled_scene
        #     println(veh.state, "  ", models[veh.id].state)
        # end

        empty!(rec)
        if safety_system
            aeb_activation_callback = AEBActivationCountCallback(ego_id)
            rec, driverstate_rec, nticks = simulate_until_collision(sampled_scene, roadway, models2, rec, driverstate_rec, (aeb_activation_callback,), nticks=max_sim_ticks)
        else
            rec, driverstate_rec, nticks = simulate_until_collision(sampled_scene, roadway, models2, rec, driverstate_rec, nticks=max_sim_ticks)
        end
        had_collision = run_callback(MobiusCollisionCallback(), rec, roadway, models2, 1)
        aeb_activated = safety_system ? aeb_activation_callback.count > 0 : false
        tick_count += nticks
        col_count += had_collision

        elapsed_time = now() - t_start
        println("\ttick_count ", tick_count, "  ", round(tick_count/tot_sim_ticks*100, 3), "%")
        println("\tcol_rate ", col_count / sim_count)
        println("\telapsed time ", elapsed_time)
        println("\testimated remaining ", Dates.Millisecond(round(Int, elapsed_time.value/tick_count * (tot_sim_ticks - tick_count))))
        println(io, "\t", clusterid)
        println(io, "\tΔ: ", Δ)
        println(io, "\tego_index: ", ego_index)
        println(io, "\tnticks ", nticks)
        println(io, "\thad_collision ", had_collision)
        println(io, "\taeb_activated ", aeb_activated)
        println(io, "\telapsed time ", elapsed_time)

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
    end
close(io)
toc()

# println(ttc_crit)
# using PGFPlots
# p = Axis([
#         Plots.Histogram(filter!(v->v < 30.0, ttc_crit), density=true, style="forget plot"),
#         ], xlabel=L"ttc_{crit}", ylabel="probability", ymin=0, xmin=0, width="12cm", style="legend pos=outer north east")
# PGFPlots.save("plot.pdf", p)

println("DONE")