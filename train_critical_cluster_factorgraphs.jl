using AutomotiveDrivingModels
using AutoScenes
using Records
using Parameters
using Clustering

include("AutoMobius/AutoMobius.jl")
using AutoMobius

const DATA_DIR = "/media/tim/DATAPART1/PublicationData/2017_adas_validation/"
const OUTPUT_FILE_SUFFIX = "1019"
const COLLISION_SCENARIO_DIR = joinpath(DATA_DIR, "collision_scenarios_$OUTPUT_FILE_SUFFIX")

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

counts = Dict{Int,Int}()
for clusterid in clusterids
    println("clusterid: ", sum(cluster_assignments .== clusterid))
end
println("total: ", length(cluster_assignments))

speed_lo = Inf
speed_hi = -Inf
for scene in packedscenes
    for entity in scene
        speed_lo = min(speed_lo, entity.state[1].v)
        speed_hi = max(speed_hi, entity.state[1].v)
    end
end
@show (speed_lo, speed_hi)

include("crit_cluster_sampling/train_factor_graph_with_attentiveness.jl")

timestep = 0.1
roadway = Wraparound(Straight1DRoadway(200.0))

features = (v,v²,v³,Δv,Δv²,Δv³,Δs,Δs²,Δs³,vᵣΔv,vᵣΔs,ΔvΔs,
            tᵣΔv, tᵣΔv², tᵣΔs, tᵣΔs², aᵣΔv, aᵣΔv², aᵣΔs, aᵣΔs²,
            v_crit, v_crit², Δv_crit, Δv_crit², Δs_crit, Δs_crit²,
            Δv_crit_tame, Δs_crit_tame, Δv_tame_crit, Δs_tame_crit)

gradient_mask = ones(Float64, length(features))
ngsim_model = @AutoScenes.load_factor_model(joinpath(DATA_DIR, "1d_factorgraph_model_NGSIM_0804.txt"))

scene_index = 1
crit_cars = [blames[scene_index].rear, blames[scene_index].fore]
fg = get_factorgraph(packedscenes[scene_index], roadway, crit_cars, features)

factorgraphs_all = Array{typeof(fg)}(length(packedscenes))
for scene_index in 1 : length(factorgraphs_all)
    crit_cars = [blames[scene_index].rear, blames[scene_index].fore]
    fg = get_factorgraph(packedscenes[scene_index], roadway, crit_cars, features)
    factorgraphs_all[scene_index] = fg
end

# θ = ones(length(features))
# ∇ = Array{Float64}(length(θ))
# srand(0)
# factorgraphs = [fg]
# log_pseudolikelihood_gradient!(∇, features, θ, factorgraphs)

# println(log_pseudolikelihood(features, θ, factorgraphs))
# θ += 0.01*∇
# println(log_pseudolikelihood(features, θ, factorgraphs))

const α = 0.01 # learning rate
const λ = 0.01 # regularization term
const N_ITERATIONS = 1000 # 2000 # number of iterations
const ∇max = 1.0 # maximum element-wise gradient value
const SKIP = 20

clusterθs = Dict{Int, Vector{Vector{Float64}}}()
clusterlogPLs = Dict{Int, Vector{Float64}}()
clusteriters = Dict{Int, Vector{Int}}()

for clusterid in clusterids
    @show clusterid

    cluster_inds = cluster_assignments .== clusterid
    factorgraphs = factorgraphs_all[cluster_inds]

    epoch = length(factorgraphs)
    println("\tn factorgraphs: ", length(factorgraphs))

    # initialize weights to match originally learned weights
    θ = ones(length(features)) .* 0.01 # decrease other by default
    for (i,f) in enumerate(features)
        j = findfirst(ngsim_model.features, f)
        if j != 0
            θ[i] = ngsim_model.weights[j]
        end
    end

    ∇ = Array{Float64}(length(θ))
    sampler = BatchIterator(epoch)
    batch = Array{eltype(factorgraphs)}(1)
    b = i -> clamp(round(Int, sqrt(i)), 1, epoch)
    logPLs = Float64[]
    iters = Int[]
    θs = Vector{Float64}[]
    iter = 0

    push!(logPLs, log_pseudolikelihood(features, θ, factorgraphs))
    push!(iters, 1)

    tic()
    for i in 1 : N_ITERATIONS
        iter += 1
        resize!(batch, b(iter))
        pull_batch!(batch, factorgraphs, sampler)
        log_pseudolikelihood_gradient!(∇, features, θ, batch)
        ∇ -= 2λ*θ
        ∇ .*= gradient_mask
        θ += (x->clamp(x, -∇max, ∇max)).(α*∇)
        if mod(iter, SKIP) == 0
            push!(logPLs, log_pseudolikelihood(features, θ, factorgraphs) / length(factorgraphs))
            push!(iters, iter)
            push!(θs, deepcopy(θ))
        end
    end
    toc()

    clusterθs[clusterid] = θs
    clusterlogPLs[clusterid] = logPLs
    clusteriters[clusterid] = iters
end

println("saving models"); tic()
for clusterid in clusterids
    model = FactorModel(features, clusterθs[clusterid][end])
    open(joinpath(COLLISION_SCENARIO_DIR, "factorgraph_model_fast_$clusterid.txt"), "w") do io
        write(io, MIME"text/plain"(), model)
    end
    open(joinpath(COLLISION_SCENARIO_DIR, "factorgraph_model_fast_$(clusterid)_training.txt"), "w") do io
        for (iter, logPL) in zip(clusteriters[clusterid], clusterlogPLs[clusterid])
            @printf(io, "%d,%.16e\n", iter, logPL)
        end
    end
end
toc()

using PGFPlots
println("saving plot"); tic()
p = Plots.Plot[]
for (clusterid, logPLs) in clusterlogPLs
    push!(p, Plots.Linear(clusteriters[clusterid][2:end], logPLs[2:end], style="solid, thick, mark=none", legendentry=clusterid))
end
ax = Axis(p,  xlabel="iteration", ylabel="log pseudolikelihood loss", style="legend pos=outer north east")
save(joinpath(COLLISION_SCENARIO_DIR, "factorgraph_model_training_plot_fast.tex"), ax)
toc()

println("DONE")