const BATCH_SIZE = 1
const N_RISK_EST_FEATURES = 14
const MAX_N_CARS = 12

backend = DefaultBackend()
init(backend)

struct RiskEstimator
    net::Net
    input::Array{Float64,4}
    output::Matrix{Float64}
    μ::Array{Float64, 3} # (n_per_feature, max_n_cars, 1)
    σ::Array{Float64, 3} # (n_per_feature, max_n_cars, 1)
end
const RISK_ESTIMATOR = let

    input = zeros(N_RISK_EST_FEATURES,MAX_N_CARS,1,BATCH_SIZE)

    layers = [
        MemoryDataLayer(name="data", tops=[:X], batch_size=BATCH_SIZE, data = Array[input]),
        ConvolutionLayer(name="conv1", n_filter=16, kernel=(N_RISK_EST_FEATURES,1), bottoms=[:X], tops=[:conv1], neuron=Neurons.ReLU()),
        ConvolutionLayer(name="conv2", n_filter=16, kernel=(1,1), bottoms=[:conv1], tops=[:conv2], neuron=Neurons.ReLU()),
        InnerProductLayer(name="ip1", output_dim=32, neuron=Neurons.ReLU(), bottoms=[:conv2], tops=[:ip1]), # -> (32,N)
        InnerProductLayer(name="ip2", output_dim=16, neuron=Neurons.ReLU(), bottoms=[:ip1], tops=[:ip2]), # -> (16,N)
        InnerProductLayer(name="ip3", output_dim=1, bottoms=[:ip2], neuron=Neurons.Sigmoid(), tops=[:ip3]), # -> (1,N) ∈ [0,1]
    ]
    net = Net("risk-est", backend, layers)

    jldopen(joinpath(DATA_DIR, "mocha_snapshots/snapshot-100000.jld"), "r") do io
      load_network(io, net)
    end
    init(net)

    my_μ = h5open(joinpath(DATA_DIR, "risk_estimator_data_train.h5"), "r") do io
        read(io, "mean")
    end
    my_σ = h5open(joinpath(DATA_DIR, "risk_estimator_data_train.h5"), "r") do io
        read(io, "std")
    end

    RiskEstimator(net, input, Array{Float64}(1,1), my_μ, my_σ)
end

function predict_risk(est::RiskEstimator, scene::MobiusScene, models::Dict{Int, LaneFollowingDriver}, roadway::MobiusRoadway; b::Int=1)::Float64

    lead_follow = LeadFollowRelationships(scene, roadway)
    for (vehicle_index, ego) in enumerate(scene)
        i = vehicle_index

        ind_rear = lead_follow.index_rear[vehicle_index]
        ind_fore = lead_follow.index_fore[vehicle_index]
        veh_rear = scene[ind_rear]
        veh_fore = scene[ind_fore]

        est.input[1, i,1,b] = 0 # ismissing
        est.input[2, i,1,b] = ego.state.v
        est.input[3, i,1,b] = veh_fore.state.v - ego.state.v
        est.input[4, i,1,b] = ego.state.v - veh_rear.state.v
        est.input[5, i,1,b] = get_headway(scene[vehicle_index], scene[ind_fore], roadway)
        est.input[6, i,1,b] = get_headway(scene[ind_rear], scene[vehicle_index], roadway)
        est.input[7, i,1,b] = models[ego.id].submodel.submodel.v_des
        est.input[8, i,1,b] = models[veh_fore.id].submodel.submodel.v_des - models[ego.id].submodel.submodel.v_des
        est.input[9, i,1,b] = models[ego.id].state.is_attentive
        est.input[10,i,1,b] = models[ego.id].state.steps_since_attention_swap
        est.input[11,i,1,b] = models[ego.id].state.v_ego_log
        est.input[12,i,1,b] = models[ego.id].state.v_oth_log
        est.input[13,i,1,b] = models[ego.id].state.headway_log
        est.input[14,i,1,b] = get_timidness(models[ego.id].submodel)
    end
    for i in length(scene)+1 : size(est.input,2)
        est.input[:, i, 1,b] = 0
        est.input[1, i, 1,b] = 1 # ismissing
    end

    # standardize
    est.input .-= est.μ
    est.input ./= est.σ

    forward(est.net)
    copy!(est.output, est.net.output_blobs[:ip3])

    return est.output[1] # ∈ [0,1]
end