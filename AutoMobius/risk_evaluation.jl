struct MobiusCollisionCallback end
function AutomotiveDrivingModels.run_callback{S,Def,I,D<:DriverModel}(
    callback::MobiusCollisionCallback,
    rec::QueueRecord{Entity{S,Def,I}},
    roadway::MobiusRoadway,
    models::Dict{I,D},
    tick::Int,
    )

    return has_collision(rec[0], roadway)
end

mutable struct SimTickCounter
    tick::Int
end
function AutomotiveDrivingModels.run_callback{S,Def,I,D<:DriverModel}(
    callback::SimTickCounter,
    rec::QueueRecord{Entity{S,Def,I}},
    roadway::Any,
    models::Dict{I,D},
    tick::Int,
    )

    callback.tick = tick
    return false # do not terminate
end

function simulate_until_collision{D<:DriverModel}(
        scene::MobiusScene,
        roadway::MobiusRoadway,
        models::Dict{Int, D},
        rec::QueueRecord{MobiusEntity},
        driverstate_rec::QueueRecord{AttentiveEntity},
        ;
        nticks::Int = typemax(Int),
    )

    sim_tick_counter = SimTickCounter(0)
    reset_hidden_states!(models)
    simulate!(rec, scene, roadway, models, nticks, (DriverstateRecCallback(driverstate_rec), MobiusCollisionCallback(), sim_tick_counter))
    return (rec, driverstate_rec, sim_tick_counter.tick)
end
function simulate_until_collision{D<:DriverModel}(
        scene::MobiusScene,
        roadway::MobiusRoadway,
        models::Dict{Int, D},
        rec::QueueRecord{MobiusEntity},
        driverstate_rec::QueueRecord{AttentiveEntity},
        callbacks,
        ;
        nticks::Int = typemax(Int),
    )

    sim_tick_counter = SimTickCounter(0)
    reset_hidden_states!(models)
    callbacks = (DriverstateRecCallback(driverstate_rec), MobiusCollisionCallback(), sim_tick_counter, callbacks...)
    simulate!(rec, scene, roadway, models, nticks, callbacks)
    return (rec, driverstate_rec, sim_tick_counter.tick)
end

function does_sim_produce_collision{D<:LaneFollowingDriver}(
    scene::MobiusScene,
    roadway::MobiusRoadway,
    models::Dict{Int,D},
    rec::QueueRecord{MobiusEntity},
    nticks::Int,
    )

    collision_callback = MobiusCollisionCallback()
    simulate!(rec, scene, roadway, models, nticks, (collision_callback,))
    return run_callback(collision_callback, rec, roadway, models, 1)
end

function get_collision_estimation{D<:DriverModel}(
    rec::PackedListRecord,
    frame_index::Int,
    roadway::MobiusRoadway,
    prior::Float64;
    scene::MobiusScene = MobiusScene(),
    models::Dict{Int,D} = Dict{Int, LaneFollowingDriver}(),
    simrec::QueueRecord{MobiusEntity} = QueueRecord(MobiusEntity, 1, get_timestep(rec)),
    nticks::Int = nframes(rec) - frame_index,
    nsimulations::Int = 10,
    )

    α = prior
    β = prior + nsimulations
    for i in 1 : nsimulations

        unpack!(rec, frame_index, scene, models)
        if does_sim_produce_collision(scene, roadway, models, simrec, nticks)
            α += 1
            β -= 1
        end
    end

    return Beta(α, β)
end