using AutomotiveDrivingModels
include("AutoMobius/AutoMobius.jl")
using AutoMobius
using Distributions

# const DATA_DIR = "/media/tim/DATAPART1/PublicationData/2017_adas_validation/"
const DATA_DIR = "/media/tim/Tim 1500 GB/2017_adas_validation/"
const OUTPUT_FILE_SUFFIX = "1005"
const COLLISION_SCENARIO_DIR = joinpath(DATA_DIR, "collision_scenarios_$OUTPUT_FILE_SUFFIX")

const RISK_THRESOLD = 0.1
const TIMESTEP = 0.1

timestep = 0.1
roadway = Wraparound(Straight1DRoadway(200.0))

struct CritSceneBlame
    rear::Int
    fore::Int
    local_tick::Int
    global_tick::Int
    filename::String
end
blames = open(joinpath(COLLISION_SCENARIO_DIR, "crit_transition_blames.txt"), "r") do io
    lines = readlines(io)
    retval = Array{CritSceneBlame}(length(lines))
    for (i,line) in enumerate(lines)
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

struct MCRiskAssessment
    nsims::Int
    nticks::Int
    ncol::Int # number of collisions observed
end
get_risk_beta(a::MCRiskAssessment, prior::Float64=1.0) = Beta(prior + a.ncol, prior + a.nsims - a.ncol)
get_risk(a::MCRiskAssessment, prior::Float64=1.0) = mean(get_risk_beta(a, prior))
function monte_carlo_risk_assessment(
    packedscene::AttentiveMobiusScene,
    roadway::MobiusRoadway;
    nsims::Int = 100,
    nticks::Int = 100, # simulation horizon
    scene::MobiusScene = MobiusScene(),
    models::Dict{Int, LaneFollowingDriver} = Dict{Int, LaneFollowingDriver}(),
    rec::QueueRecord{MobiusEntity} = QueueRecord(MobiusEntity, 2, TIMESTEP),
    collision_callback::MobiusCollisionCallback = MobiusCollisionCallback(),
    )

    ncol = 0
    for i in 1 : nsims
        unpack!(scene, models, packedscene, get_timestep(rec))
        if has_collision(scene, roadway)
            ncol += 1
        else
            empty!(rec)
            simulate!(rec, scene, roadway, models, nticks, (collision_callback,))
            @assert nframes(rec) > 1
            ncol += has_collision(rec[0], roadway)
        end
    end

    return MCRiskAssessment(nsims, nticks, ncol)
end

struct RiskAssessment
    nsims::Int
    nticks::Int
    ncol::Int
    est_risk::Float64
end
risk_curves = Array{Vector{RiskAssessment}}(length(blames))
list_records = Array{PackedListRecord}(length(blames))
for (j,b) in enumerate(blames)
    MC_risk_path = splitext(b.filename)[1] * "_risks.txt"
    lines = readlines(joinpath(COLLISION_SCENARIO_DIR, MC_risk_path))
    mc_risks = Array{RiskAssessment}(length(lines))
    for (i,line) in enumerate(lines)
        tokens = split(strip(line), ",")
        mc_risks[i] = RiskAssessment(
            parse(Int, tokens[1]),
            parse(Int, tokens[2]),
            parse(Int, tokens[3]),
            parse(Float64, tokens[4]),
        )
    end
    risk_curves[j] = mc_risks
    list_records[j] = open(joinpath(COLLISION_SCENARIO_DIR, b.filename), "r") do io
        read(io, MIME"text/plain"(), PackedListRecord)
    end
end

max_n_cars = 0
for lrec in list_records
    for frame in 1 : nframes(lrec)
        max_n_cars = max(max_n_cars, n_objects_in_frame(lrec, frame))
    end
end

tot_n_scenes_above_risk_threshold = 0
tot_n_scenes = 0
for risk_curve in risk_curves
    for r in risk_curve
        risk = mean(Beta(1 + r.ncol, 1 + r.nsims - r.ncol))
        tot_n_scenes += 1
        tot_n_scenes_above_risk_threshold += risk > RISK_THRESOLD
    end
end
@show tot_n_scenes
@show tot_n_scenes_above_risk_threshold
@show max_n_cars

n_per_feature = 15 # {ismissing, v, Δv_fore, Δv_rear, Δs_fore, Δs_rear, v_des, Δv_des_fore, inv_ttc_fore, attentiveness,
                   #  steps_since_attention_swap, v_ego_log, v_oth_log, headway_log, timidness}
width = n_per_feature
height = max_n_cars

X = zeros(Float64, width, height, 1, 2tot_n_scenes) .* NaN # (width,height,channels,number)
y = Array{Float64}(2tot_n_scenes)

scene = MobiusScene()
scene_extract = Frame(AttentiveMobiusEntity)
models = Dict{Int, LaneFollowingDriver}()
roadway = Wraparound(Straight1DRoadway(200.0))

n_infractions = 0

b = 0
for (i,(lrec,risk_curve)) in enumerate(zip(list_records, risk_curves))
    for (frame, r) in enumerate(risk_curve)

        risk = mean(Beta(1 + r.ncol, 1 + r.nsims - r.ncol)) # based on MC eval
        unpack!(lrec, frame, scene, models, scene_extract=scene_extract)
        lead_follow = LeadFollowRelationships(scene, roadway)

        b += 1
        y[b] = risk
        for (vehicle_index, ego) in enumerate(scene)
            i = vehicle_index

            ind_rear = lead_follow.index_rear[vehicle_index]
            ind_fore = lead_follow.index_fore[vehicle_index]
            veh_rear = scene[ind_rear]
            veh_fore = scene[ind_fore]

            X[1, i,1,b] = 0 # ismissing
            X[2, i,1,b] = ego.state.v
            X[3, i,1,b] = veh_fore.state.v - ego.state.v
            X[4, i,1,b] = ego.state.v - veh_rear.state.v
            X[5, i,1,b] = get_headway(scene[vehicle_index], scene[ind_fore], roadway)
            X[6, i,1,b] = get_headway(scene[ind_rear], scene[vehicle_index], roadway)
            X[7, i,1,b] = models[ego.id].submodel.submodel.v_des
            X[8, i,1,b] = models[veh_fore.id].submodel.submodel.v_des - models[ego.id].submodel.submodel.v_des
            X[9, i,1,b] = models[ego.id].state.is_attentive
            X[10,i,1,b] = get_timidness(models[ego.id].submodel)
        end
        for i in length(scene)+1 : max_n_cars
            X[:, i, 1,b] = 0
            X[1, i, 1,b] = 1 # ismissing
        end
    end
end
X = X[:,:,:,1:b]
y = y[1:b]
@assert !any(x->isnan(x) || isinf(x), X)
@assert !any(x->isnan(x) || isinf(x), y)

@show n_infractions
@show tot_n_scenes

println("DONE")

# {ismissing, v, Δv_fore, Δv_rear, Δs_fore, Δs_rear, v_des, Δv_des_fore, attentiveness, steps_since_attention_swap, v_ego_log, v_oth_log, headway_log, timidness}
μ = zeros(Float64, n_per_feature, max_n_cars, 1)
σ = ones(Float64, n_per_feature, max_n_cars, 1)
for i in [2,5,6,7,9,11,12,13,14,15]
    for j in 1 : max_n_cars
        μ[i,j,1] = mean(X[i,j,1,:])
    end
end
for i in [2,3,4,5,6,7,8,9,11,12,13,14,15]
    for j in 1 : max_n_cars
        σ[i,j,1] = std(X[i,j,1,:])
    end
end

Xtrans = (X .- μ)./σ # standardize

@assert !any(x->isnan(x) || isinf(x), Xtrans)
@assert !any(x->isnan(x) || isinf(x), y)

train = 1 : round(Int, tot_n_scenes*0.75)
test = train[end]+1 : tot_n_scenes

println("μ: ", μ)
println("σ: ", σ)

using HDF5
h5open(joinpath(DATA_DIR, "risk_estimator_data_train.h5"), "w") do io
    write(io, "mean", μ)
    write(io, "std", σ)
    write(io, "X", Xtrans[:,:,:,train])
    write(io, "y", y[train])
end
h5open(joinpath(DATA_DIR, "risk_estimator_data_test.h5"), "w") do io
    write(io, "mean", μ)
    write(io, "std", σ)
    write(io, "X", Xtrans[:,:,:,test])
    write(io, "y", y[test])
end

println("DONE")