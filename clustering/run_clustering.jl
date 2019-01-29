using AutomotiveDrivingModels
using AutoScenes
using Records
using MultivariateStats
using Clustering

include("../AutoMobius/AutoMobius.jl")
using AutoMobius

const DATA_DIR = "/media/tim/DATAPART1/PublicationData/2017_adas_validation/"
# const OUTPUT_FILE_SUFFIX = "1010"
const OUTPUT_FILE_SUFFIX = "1019"
const COLLISION_SCENARIO_DIR = joinpath(DATA_DIR, "collision_scenarios_$OUTPUT_FILE_SUFFIX")

const N_KMEANS_TRIES = 100
const KVALS = collect(2:10)

# - load critical situations
struct CritSceneBlame
    rear::Int
    fore::Int
    local_tick::Int
    global_tick::Int
    filename::String
end
blames = open(joinpath(COLLISION_SCENARIO_DIR, "crit_transition_blames.txt"), "r") do io
    retval = CritSceneBlame[]
    lines = readlines(io)
    for line in lines
        tokens = split(strip(line), ",")
        filename=strip(tokens[1])
        rear = parse(Int, tokens[2])
        fore = parse(Int, tokens[3])
        local_tick = parse(Int, tokens[4])
        global_tick = parse(Int, tokens[5])
        push!(retval, CritSceneBlame(rear, fore, local_tick, global_tick,filename))
    end
    retval
end

# remove blames that don't find collisions (should be very few)
filter!(b->b.rear != 0, blames)

# - extract features
m = length(blames)
@show m
n = 15
@show n
#=
v_rear, # speed
v_fore,
a_rear, # attentiveness
a_fore,
t_rear, # timidness
t_fore,
Δv_des_fore_rear, # relative desired speed
Δv_fore_rear, # relative speed
Δs_fore_rear, # headway
ttc_fore_rear, # time to collision
δv_rear, # speed change between past and present
δa_rear, # attentiveness change for rear car
δΔv_fore_rear, # change in Δv_fore_rear between past and present
δΔs_fore_rear, # change in Δv_fore_rear between past and present
δttc_fore_rear, # change in ttc between past and present
=#

X = Array{Float64}(n,m) # each column is an observation

scene_crit = MobiusScene()
scene_past = MobiusScene()
models_crit = Dict{Int, LaneFollowingDriver}()
models_past = Dict{Int, LaneFollowingDriver}()
roadway = Wraparound(Straight1DRoadway(200.0))
for (j,b) in enumerate(blames)

    lrec = open(joinpath(COLLISION_SCENARIO_DIR, b.filename), "r") do io
        read(io, MIME"text/plain"(), PackedListRecord)
    end

    tick_crit = b.local_tick
    empty!(models_crit)
    unpack!(lrec, tick_crit, scene_crit, models_crit)

    # make sure past scene had the same cars
    ids = sort!(collect(keys(models_crit)))
    tick_past = tick_crit
    for i in 1 : 10 # up to a second before
        if tick_past > 1
            tick_past -= 1
            empty!(models_past)
            unpack!(lrec, tick_past, scene_past, models_past)
            ids_past = sort!(collect(keys(models_past)))
            if ids_past != ids
                tick_past += 1
                break
            end
        end
    end

    empty!(models_past)
    unpack!(lrec, tick_past, scene_past, models_past)

    ind_rear = b.rear
    ind_fore = b.fore
    veh_rear_crit = scene_crit[ind_rear]
    veh_fore_crit = scene_crit[ind_fore]
    veh_rear_past = scene_past[ind_rear]
    veh_fore_past = scene_past[ind_fore]
    id_rear = veh_rear_crit.id
    id_fore = veh_fore_crit.id

    X[1,j] = veh_rear_crit.state.v
    X[2,j] = veh_fore_crit.state.v
    X[3,j] = models_crit[id_rear].state.is_attentive
    X[4,j] = models_crit[id_fore].state.is_attentive
    X[5,j] = get_timidness(models_crit[id_rear].submodel)
    X[6,j] = get_timidness(models_crit[id_fore].submodel)
    X[7,j] = models_crit[id_fore].submodel.submodel.v_des - models_crit[id_rear].submodel.submodel.v_des
    X[8,j] = veh_fore_crit.state.v - veh_rear_crit.state.v
    X[9,j] = get_headway(veh_rear_crit, veh_fore_crit, roadway)
    X[10,j] = X[8,j] / (X[9,j] + 1.0) # NOTE: not exactly ttc but similar (and can go negative). More useful?

    X[11,j] = veh_rear_crit.state.v - veh_rear_past.state.v
    X[12,j] = models_crit[id_rear].state.is_attentive - models_past[id_rear].state.is_attentive
    X[13,j] = X[8,j] - (veh_fore_past.state.v - veh_rear_past.state.v)
    X[14,j] = X[9,j] - get_headway(veh_rear_past, veh_fore_past, roadway)
    X[15,j] = X[10,j] - ((veh_fore_past.state.v - veh_rear_past.state.v) / ((get_headway(veh_rear_past, veh_rear_past, roadway)) + 1.0))
    # X[16,j] = length(scene_crit)

    for l in 1 : n
        if isnan(X[l,j]) || isinf(X[l,j])
            println("$l  $m: ", X[l,j])
        end
    end
end
@assert !any(x->isnan(x) || isinf(x), X)

# # - run PCA
# pca = fit(PCA, X)
# Xreduced = transform(pca, X)
# @show size(Xreduced)

μ = mean(X, 2)
σ = std(X, 2)
Xreduced = (X .- μ) ./ σ
@show size(Xreduced)

# - cluster
n, m = size(Xreduced)
dists = Array{Float64}(m,m)
for i in 1 : m
    dists[i,i] = 0
    for j in i+1 : m
        dists[i,j] = dists[j,i] = norm(Xreduced[:,i] - Xreduced[:,j],2)
    end
end

srand(0)
best_clustering = kmeans(Xreduced, KVALS[1], init=:rand)
best_s = mean(silhouettes(best_clustering, dists))
@show KVALS[1]
@show best_s
for k in KVALS
    for i in 1 : N_KMEANS_TRIES # number of times to run
        res = kmeans(Xreduced, k, init=:rand)
        if res.converged
            s = mean(silhouettes(res, dists)) # use silhouettes to decide what level k to use. s[i] close to one indicates that the i-th point lies will within its own cluster
                                              # use the average silhouette result to assess how good the clustering level is
            if s > best_s
                best_s = s
                best_clustering = res
                @show k
                @show best_s
            end
        else
            warn("$k number $i did not converge")
        end
    end
end

# output cluster assignments

R = best_clustering
a = assignments(R)
c = counts(R) # number of samples in each cluster
@show nclusters(R)
@show c

for k in 1 : nclusters(R)
    inds = a .== k
    cluster_mean = mean(X[:,inds], 2)

    @show k
    println("counts: ", c[k])
    println("v_rear:           ", cluster_mean[1])
    println("v_fore:           ", cluster_mean[2])
    println("a_rear:           ", cluster_mean[3])
    println("a_fore:           ", cluster_mean[4])
    println("t_rear:           ", cluster_mean[5])
    println("t_fore:           ", cluster_mean[6])
    println("Δv_des_fore_rear: ", cluster_mean[7])
    println("Δv_fore_rear:     ", cluster_mean[8])
    println("Δs_fore_rear:     ", cluster_mean[9])
    println("ttc_fore_rear:    ", cluster_mean[10])
    println("δv_rear:          ", cluster_mean[11])
    println("δa_rear:          ", cluster_mean[12])
    println("δΔv_fore_rear:    ", cluster_mean[13])
    println("δΔs_fore_rear:    ", cluster_mean[14])
    println("δttc_fore_rear:   ", cluster_mean[15])
    # println("n_vehicles:       ", cluster_mean[16])
end

open(joinpath(COLLISION_SCENARIO_DIR, "crit_cluster_assignments.txt"), "w") do io
    for i in a
        println(io, i)
    end
end

println("DONE")

#=
nclusters(R) = 5
c = [1237, 200, 161, 2269, 49]
k = 1
counts: 1237
v_rear:           13.665689073130428
v_fore:           14.189596050847452
a_rear:           0.7857720291026677
a_fore:           0.9991915925626516
t_rear:           0.6032471728550863
t_fore:           0.502729194944563
Δv_des_fore_rear: -1.115749554806802
Δv_fore_rear:     0.5239069777170021
Δs_fore_rear:     26.219212350973123
ttc_fore_rear:    0.0207775483463094
δv_rear:          0.2957036403033734
δa_rear:          0.0024252223120452706
δΔv_fore_rear:    -0.1352369977523782
δΔs_fore_rear:    0.6281997520309202
δttc_fore_rear:   0.01742324546749285
k = 2
counts: 200
v_rear:           16.357957359249077
v_fore:           15.157019162965046
a_rear:           0.08
a_fore:           0.0
t_rear:           0.6295289013982953
t_fore:           0.4553547233933311
Δv_des_fore_rear: -1.9333333758551043
Δv_fore_rear:     -1.2009381962840333
Δs_fore_rear:     19.895689759063288
ttc_fore_rear:    -0.06109702574556249
δv_rear:          0.19699540775299607
δa_rear:          -0.005
δΔv_fore_rear:    -0.22849745329549834
δΔs_fore_rear:    -1.0773845149006183
δttc_fore_rear:   -0.05615082761090915
k = 3
counts: 161
v_rear:           13.957394515545138
v_fore:           14.118034472627853
a_rear:           0.0
a_fore:           0.9751552795031055
t_rear:           0.5533923048507431
t_fore:           0.47613710255902625
Δv_des_fore_rear: -0.8575327454380632
Δv_fore_rear:     0.16063995708272039
Δs_fore_rear:     27.512247149679087
ttc_fore_rear:    0.007940378600885792
δv_rear:          0.2771612107830532
δa_rear:          -0.9999999999999999
δΔv_fore_rear:    -0.4290194891985799
δΔs_fore_rear:    0.4904432035187189
δttc_fore_rear:   0.004936089401516371
k = 4
counts: 2269
v_rear:           17.347605758610563
v_fore:           15.825513814040885
a_rear:           0.0035257822829440283
a_fore:           1.0
t_rear:           0.6199339354419736
t_fore:           0.4761084173258895
Δv_des_fore_rear: -1.5964632510885564
Δv_fore_rear:     -1.5220919445697016
Δs_fore_rear:     18.20063887192794
ttc_fore_rear:    -0.07915980063444573
δv_rear:          0.18224741887622642
δa_rear:          0.0
δΔv_fore_rear:    -0.16874166643328045
δΔs_fore_rear:    -1.4196700816543877
δttc_fore_rear:   -0.07227041797417154
k = 5
counts: 49
v_rear:           5.690900788557871
v_fore:           4.5103898652419225
a_rear:           0.9183673469387754
a_fore:           0.4693877551020408
t_rear:           0.5703964288590757
t_fore:           0.6340939477023154
Δv_des_fore_rear: 0.70704245915996
Δv_fore_rear:     -1.1805109233159485
Δs_fore_rear:     10.63227865653597
ttc_fore_rear:    -0.12908610111815577
δv_rear:          -0.5267403226177534
δa_rear:          0.14285714285714285
δΔv_fore_rear:    -0.18360691524706996
δΔs_fore_rear:    -1.0282728689660383
δttc_fore_rear:   -0.12401996792861675
DONE
=#