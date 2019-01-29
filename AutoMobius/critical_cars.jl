"""
Identify all cars that make the scene critical.
For 1D I assume this is just the set of cars that actually collide - in 2D that is not necessarily the case.
"""
function assign_blame(rec::PackedListRecord, roadway::MobiusRoadway;
    scene::MobiusScene = MobiusScene(),
    )

    unpack!(rec, nframes(rec), scene) # unpack most recent scene, which should contain the collision.
    i, j = get_first_collision(scene, roadway) # in any order
    return [i,j]
end

function findfirst_frame_above_risk_threshold(
    collision_risk::Vector{Beta},
    threshold::Float64, # risk threhsold
    )

    for (frame,β) in enumerate(collision_risk)
        if mean(β) > threshold
            return frame
        end
    end
    return 0 # not found
end

function unpack_critical_scene!{D}(
    scene::MobiusScene,
    rec::PackedListRecord,
    roadway::MobiusRoadway,
    collision_risk::Vector{Beta},
    risk_threshold::Float64;
    models::Dict{Int,D} = Dict{Int, LaneFollowingDriver}(),
    )

    frame_index = findfirst_frame_above_risk_threshold(collision_risk, risk_threshold)
    unpack!(rec, frame_index, scene, models)
    return scene
end

function unpack_critical_scene!(
    scene::AttentiveMobiusScene,
    rec::PackedListRecord,
    roadway::MobiusRoadway,
    collision_risk::Vector{Beta},
    risk_threshold::Float64,
    )

    frame_index = findfirst_frame_above_risk_threshold(collision_risk, risk_threshold)
    get!(scene, rec, frame_index)
    return scene
end