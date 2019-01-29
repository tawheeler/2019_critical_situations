const AttentiveEntity = Entity{AttentiveDriverState, Void, Int}

get_attentive_driver_state(model::LaneFollowingDriver) = model.state
function pull_driverstate_frame{D<:DriverModel}(models::Dict{Int, D})
    Frame([AttentiveEntity(get_attentive_driver_state(model), nothing, id) for (id, model) in models])
end
function set_driverstate_frame!{D<:LaneFollowingDriver}(models::Dict{Int, D}, frame::Frame{AttentiveEntity})
    for (state, _, id) in frame
         models[id].state = state
    end
    return models
end

mutable struct DriverstateRecCallback
    driverstate_rec::QueueRecord{AttentiveEntity}
end
function AutomotiveDrivingModels.run_callback{S,Def,I,D<:DriverModel}(
    callback::DriverstateRecCallback,
    rec::QueueRecord{Entity{S,Def,I}},
    roadway::Any,
    models::Dict{I,D},
    tick::Int,
    )

    frame = pull_driverstate_frame(models)
    Records.update!(callback.driverstate_rec, frame)
    return false # do not terminate
end

