

function infer_aggressiveness(model::DriverModel, passive_v_des::Float64 = 33.3, aggressive_v_des::Float64 = 38.9)
    return (model.submodel.v_des - passive_v_des) / (aggressive_v_des - passive_v_des)
end

function infer_attentiveness(model::DriverModel)
    return model.state.is_attentive
end

function get_ttc(veh_rear::Vehicle1D, veh_fore::Vehicle1D, censor_hi::Float64=10.)
    Δs = get_rear(veh_fore) - get_front(veh_rear)
    Δv = veh_rear.state.v - veh_fore.state.v
    if Δv <= 0
        ttc = 0.
    else
        ttc = max(min(Δs / Δv, censor_hi), 0.)
    end
    return ttc
end

type ScenarioFeatureExtractor <: AbstractFeatureExtractor
    features::Vector{Float64}
    num_features::Int64
    function ScenarioFeatureExtractor()
        num_features = 9
        return new(zeros(Float64, num_features), num_features)
    end
end
Base.length(ext::ScenarioFeatureExtractor) = ext.num_features
function feature_names(ext::ScenarioFeatureExtractor)
    return String["num_veh", 
        "mean_headway", 
        "min_headway", 
        "mean_ttc", 
        "mean_velocity", 
        "std_dev_velocity", 
        "mean_aggressiveness", 
        "std_dev_aggressiveness", 
        "mean_attentive"]
end

function AutomotiveDrivingModels.pull_features!(
        ext::ScenarioFeatureExtractor, 
        scene::Scene1D,
        roadway::StraightRoadway,   
        models::Dict{Int, LaneFollowingDriver}
    )
    veh_features = Dict{String, Array{Float64}}()
    names = ["fore_headway", "fore_ttc", "velocity", "aggressiveness", "attentiveness"]
    for name in names
        veh_features[name] = Float64[]
    end
    for (veh_idx, veh) in enumerate(scene)
        # track neighbor fore
        neigh_fore = get_neighbor_fore(scene, veh_idx, roadway)

        # extract
        headway = 30. # censor high
        ttc = 10. # censor high
        if neigh_fore.ind != 0
            headway = get_headway(veh, scene[neigh_fore.ind], roadway)
            ttc = get_ttc(veh, scene[neigh_fore.ind])
        end
        headway = min(headway, 30.)
        ttc = min(ttc, 10.)

        # collect
        push!(veh_features["fore_headway"], headway)
        push!(veh_features["fore_ttc"], ttc)
        push!(veh_features["velocity"], veh.state.v)
        aggressiveness = infer_aggressiveness(models[veh.id])
        push!(veh_features["aggressiveness"], aggressiveness)
        attentiveness = infer_attentiveness(models[veh.id])
        push!(veh_features["attentiveness"], attentiveness)
    end
    idx = 0
    ext.features[idx+=1] = length(scene)
    ext.features[idx+=1] = mean(veh_features["fore_headway"])
    ext.features[idx+=1] = minimum(veh_features["fore_headway"])
    ext.features[idx+=1] = mean(veh_features["fore_ttc"])
    ext.features[idx+=1] = mean(veh_features["velocity"])
    ext.features[idx+=1] = std(veh_features["velocity"])
    ext.features[idx+=1] = mean(veh_features["aggressiveness"])
    ext.features[idx+=1] = std(veh_features["aggressiveness"])
    ext.features[idx+=1] = mean(veh_features["attentiveness"])
    return ext.features
end