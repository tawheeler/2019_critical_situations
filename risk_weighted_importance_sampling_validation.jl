using AutomotiveDrivingModels
using AutoScenes
using AutoViz
using Records
using Parameters
using Distributions
using HDF5, JLD
using Mocha
import StatsBase: sample

include("AutoMobius/AutoMobius.jl")
using AutoMobius

include("clustering/clustering.jl")
include("crit_cluster_sampling/sample_factor_graph_with_attentiveness.jl")
include("safety_validation/safety_system.jl")

const RISK_THRESHOLD = 0.15
const RISK_EPSILON = 0.025
tot_sim_time = 1e8 #1e9 # [s]

const DATA_DIR = "/media/tim/DATAPART1/PublicationData/2017_adas_validation/"
const OUTPUT_FILE_SUFFIX = "1019"
const COLLISION_SCENARIO_DIR = joinpath(DATA_DIR, "collision_scenarios_$OUTPUT_FILE_SUFFIX")

###################################################
#               MOCHA
###################################################

include("risk_estimator.jl")

###################################################
#               END MOCHA
###################################################

println("LOADING DATA"); tic()

packedscenes = open(joinpath(COLLISION_SCENARIO_DIR, "crit_transition_scenes.txt"), "r") do io
    read(io, MIME"text/plain"(), Vector{AttentiveMobiusScene})
end
struct CritSceneBlame
    rear::Int
    fore::Int
    local_tick::Int
    global_tick::Int
    filename::String
end
blames = open(joinpath(COLLISION_SCENARIO_DIR, "crit_transition_blames.txt"), "r") do io
    retval = Array{CritSceneBlame}(length(packedscenes))
    lines = readlines(io)
    for i in 1 : length(retval)
        line = lines[i]
        tokens = split(strip(line), ",")
        filename=strip(tokens[1])
        rear = parse(Int, tokens[2])
        fore = parse(Int, tokens[3])
        local_tick = parse(Int, tokens[4])
        global_tick = parse(Int, tokens[5])
        retval[i] = CritSceneBlame(rear, fore, local_tick, global_tick,filename)
    end
    retval
end

indices = [b.rear != 0 for b in blames]
blames = blames[indices]
packedscenes = packedscenes[indices]

timestep = 0.1
roadway = Wraparound(Straight1DRoadway(200.0))

cluster_assignments = open(joinpath(COLLISION_SCENARIO_DIR, "crit_cluster_assignments.txt"), "r") do io
    retval = Array{Int}(length(blames))
    for (i,line) in enumerate(readlines(io))
        retval[i] = parse(Int, line)
    end
    retval
end
toc()

clusterids = unique(cluster_assignments)
println("clusterids: ", clusterids)

### Load Factor Graphs
factormodels = Dict{Int, FactorModel}()
cluster_inds = Dict{Int, Vector{Int}}()
cluster_factorgraphs = Dict{Int, Vector{FactorGraph}}()
cluster_counts = Dict{Int, Int}()
for clusterid in clusterids
    model = @AutoScenes.load_factor_model(joinpath(COLLISION_SCENARIO_DIR, "factorgraph_model_fast_$clusterid.txt"))
    model = create_penalty_model_from_global(model)
    factormodels[clusterid] = model
    cluster_inds[clusterid] = find(cluster_assignments .== clusterid)
    cluster_counts[clusterid] = sum(cluster_assignments .== clusterid)

    cluster_factorgraphs[clusterid] = FactorGraph[]
    for (scene_index,a) in enumerate(cluster_assignments)
        if a == clusterid
            crit_cars = [blames[scene_index].rear, blames[scene_index].fore]
            fg = get_factorgraph(packedscenes[scene_index], roadway, crit_cars, model.features)
            push!(cluster_factorgraphs[clusterid], fg)
        end
    end
end

# TODO: store this in a file and then load it
proposal_probs = Dict{Int,Float64}(clusterid => 1.0 for clusterid in clusterids)

P = Categorical(normalize([cluster_counts[id] for id in clusterids],1)) # true distribution

@show P

T = Dict(:s => Normal(0.0,5.0), :v => Normal(0.0,1.0), :t => Normal(0.0,0.1))
burnin = 1000
gens = Dict{Int, FactorGraphSceneGenerator}()
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

function risk_enforced_metropolis_hastings_step!{F,R}(
    gen::FactorGraphSceneGenerator{F},
    factorgraph::FactorGraph{R},
    a::Vector{Float64}, # a set of deviations from factorgraph.vars.values
    b::Vector{Float64}, # candidate deviations from factorgraph.vars.values
    logPtilde_a::Float64, # precomputed
    r_target::Float64, # target risk
    ϵ::Float64, # risk epsilon
    est::RiskEstimator,
    scene::MobiusScene,
    models::Dict{Int, LaneFollowingDriver},
    riskest_scene::MobiusScene,
    riskest_packedscene::AttentiveMobiusScene,
    riskest_packedscene_out::AttentiveMobiusScene,
    )

    vars = factorgraph.vars

    # compute logP(a→b) and logP(b→a)
    # use TruncatedNormals to help enforce bounds
    logP_a2b = 0.0
    logP_b2a = 0.0
    for i in 1 : length(vars)
        sym = vars.symbols[i]
        bounds = vars.bounds[i]

        if bounds != ZERO_BOUND # todo: precompute this
            Pa2b = Truncated(gen.Ts[sym], bounds.Δlo - a[i], bounds.Δhi - a[i])
            Δ = rand(Pa2b) # proposed transition for this variable
            logP_a2b += logpdf(Pa2b, Δ)
            b[i] = a[i] + Δ
            Pb2a = Truncated(gen.Ts[sym], bounds.Δlo - b[i], bounds.Δhi - b[i])
            logP_b2a += logpdf(Pb2a, -Δ)
            @assert !isinf(logP_a2b)
            @assert !isinf(logP_b2a)
        end
    end

    # TODO: ensure that new state is acceptable to the graph

    # risk enforcement
    shift_scene!(riskest_packedscene_out, riskest_packedscene, b, factorgraph)
    unpack!(riskest_scene, models, riskest_packedscene_out, timestep)
    r = predict_risk(est, riskest_scene, models, factorgraph.roadway)
    if abs(r - r_target) > ϵ
        return (a, logPtilde_a)
    end

    # calc acceptance probability
    # - you need to add b into vars.values and then recover
    vars.values .+= b
    logPtilde_b = log_ptilde(gen.model.features, gen.model.weights, vars,
                             factorgraph.assignments, factorgraph.roadway)
    vars.values .-= b

    logA = logPtilde_b - logPtilde_a + logP_b2a - logP_a2b
    A = exp(logA)

    # if has_collision(riskest_scene, roadway)
    #     @printf("\tprob accept %.3f\n", A)
    #     println("\tlogA: ", logA)
    #     println("\tlogPtilde_a: ", logPtilde_a)
    #     println("\tlogPtilde_b: ", logPtilde_b)
    #     println("\tlogP_b2a: ", logP_b2a)
    #     println("\tlogP_a2b: ", logP_a2b)
    # end

    # see whether we accept
    if rand() ≤ A
        # @show "accepted"
        @assert !has_collision(riskest_scene, roadway)
        copy!(a, b)
        logPtilde_a = logPtilde_b
    end

    return (a, logPtilde_a)
end
function risk_enforced_metropolis_hastings!{F,R}(
    gen::FactorGraphSceneGenerator{F},
    factorgraph::FactorGraph{R},
    est::RiskEstimator,
    scene::MobiusScene,
    models::Dict{Int, LaneFollowingDriver},
    riskest_scene::MobiusScene,
    riskest_packedscene::AttentiveMobiusScene,
    riskest_packedscene_out::AttentiveMobiusScene,
    ;

    r_target::Float64 = RISK_THRESHOLD,
    ϵ::Float64 = RISK_EPSILON,
    n_steps::Int = gen.burnin,
    a::Vector{Float64} = zeros(length(factorgraph.vars)),
    b::Vector{Float64} = zeros(length(factorgraph.vars)),
    logPtilde_a::Float64 = begin
        factorgraph.vars.values .+= a
        logPtilde_a = log_ptilde(gen.model.features, gen.model.weights, factorgraph.vars,
            factorgraph.assignments, factorgraph.roadway)
        factorgraph.vars.values .-= a
        logPtilde_a
    end,
    )

    for i in 1 : n_steps
        (a, logPtilde_a) = risk_enforced_metropolis_hastings_step!(gen, factorgraph, a, b, logPtilde_a, r_target, ϵ, est, scene, models, riskest_scene, riskest_packedscene, riskest_packedscene_out)
    end

    return a
end

mutable struct CurrentRisk
    r::Float64
end
const CURRENT_RISK = CurrentRisk(NaN)

"""
    Risk Estimation Callback

Halts the simulation once the estimated risk goes below threshold
"""
struct RiskEstCallback
    est::RiskEstimator
    current_risk::CurrentRisk
    risk_threshold::Float64
    models::Dict{Int,LaneFollowingDriver}
end
function AutomotiveDrivingModels.run_callback{S,Def,I,D<:DriverModel}(
    callback::RiskEstCallback,
    rec::QueueRecord{Entity{S,Def,I}},
    roadway::MobiusRoadway,
    models::Dict{I,D},
    tick::Int,
    )

    for (id, m) in models
        if isa(m, LaneFollowingDriver)
            callback.models[id] = m
        else
            callback.models[id] = m.human.submodel
        end
    end

    r = predict_risk(callback.est, rec[0], callback.models, roadway)
    retval = callback.current_risk.r > callback.risk_threshold && r < callback.risk_threshold
    callback.current_risk.r = r
    return retval
end
risk_est_callback = RiskEstCallback(
                        RISK_ESTIMATOR,
                        CURRENT_RISK,
                        RISK_THRESHOLD - RISK_EPSILON,
                        Dict{Int,LaneFollowingDriver}(),
                    )

srand(RUNNUM)

scene = MobiusScene()
timestep = 0.1
roadway = Wraparound(Straight1DRoadway(200.0))
behgen = AutoMobius.ErrorableIDMBehaviorGenerator(timestep)

max_sims = Inf
tot_sim_ticks = round(Int, tot_sim_time/timestep)
max_sim_ticks = 100000

record_history = 2
rec = QueueRecord(MobiusEntity, record_history, timestep)
driverstate_rec = QueueRecord(AttentiveEntity, record_history, timestep)

riskest_scene = MobiusScene()
riskest_packedscene = AttentiveMobiusScene()
riskest_packedscene_out = AttentiveMobiusScene()

aeb_activation_callback = AEBActivationCountCallback(0)

col_count = 0
sim_count = 0
tick_count = 0
t_start = now()
safety_system = true
tic()
io = open(joinpath(COLLISION_SCENARIO_DIR, safety_system ? "is_eval_$RUNNUM.txt" : "is_eval_no_safety_sys_$RUNNUM.txt"), "w")
    println(io, "burnin: ", burnin)
    println(io, "T: ", T)
    println(io, "timestep: ", timestep)
    println(io, "tot_sim_time: ", tot_sim_time)
    println(io, "tot_sim_ticks: ", tot_sim_ticks)
    println(io, "clusters: ", length(clusterids))
    for clusterid in clusterids
        println(io, "\t", clusterid)
        println(io, "\tframe_count: ", cluster_counts[clusterid])
        println(io, "\tproposal_prob: ", proposal_probs[clusterid])
    end

    while tick_count < tot_sim_ticks && sim_count < max_sims

        sim_count += 1
        println("sim ", sim_count)
        println(io, "sim ", sim_count)

        # sample new start scene
        clusterid = clusterids[rand(P)]
        println("\t", clusterid)

        models = Dict{Int, LaneFollowingDriver}()
        ego_index = 0

        j = sample(1:length(cluster_inds[clusterid]))
        packedscene_index = cluster_inds[clusterid][j]
        packedscene = packedscenes[packedscene_index]
        unpack!(scene, models, packedscene, timestep)
        copy!(riskest_packedscene, packedscene)
        ego_index = blames[packedscene_index].rear
        factorgraph = cluster_factorgraphs[clusterid][j]

        Δ = risk_enforced_metropolis_hastings!(gens[clusterid], factorgraph, RISK_ESTIMATOR, scene, models, riskest_scene, riskest_packedscene, riskest_packedscene_out)
        # fill!(Δ, 0.0) # DEBUG
        # Δ = zeros(length(factorgraph.vars)) # DEBUG
        shift_scene!(riskest_packedscene, packedscene, Δ, factorgraph)

        # assign ego car at random (as there is no guarantee that ego car is the crit car!)
        ego_id = riskest_packedscene[rand(1:length(riskest_packedscene))].id
        unpack!(scene, models, riskest_packedscene, timestep)
        models2 = Dict{Int, DriverModel}()
        for (id, m) in models
            models2[id] = m
        end
        if safety_system
            models2[ego_id] = VolvoAEB(WrappedLaneFollowingDriver{StoppingAccel,typeof(models[ego_id])}(models[ego_id]))
        end

        empty!(rec)
        if safety_system
            aeb_activation_callback.count = 0
            aeb_activation_callback.ego_id = ego_id
            rec, driverstate_rec, nticks = simulate_until_collision(scene, roadway, models2, rec, driverstate_rec, (aeb_activation_callback, risk_est_callback), nticks=max_sim_ticks)
        else
            rec, driverstate_rec, nticks = simulate_until_collision(scene, roadway, models2, rec, driverstate_rec, (risk_est_callback,), nticks=max_sim_ticks)
        end

        had_collision = run_callback(MobiusCollisionCallback(), rec, roadway, models2, 1)
        col_count += had_collision
        activated_aeb_count = safety_system ? aeb_activation_callback.count : 0
        tick_count += nticks

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
        println(io, "\taeb_activated ", activated_aeb_count)
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

println("DONE")