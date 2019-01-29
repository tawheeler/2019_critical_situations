using Vec

immutable PosSpeedCritical
    s::Float64
    v::Float64
    iscrit::Bool
end

const MobiusCritEntity = Entity{PosSpeedCritical, BoundingBoxDef, Int}
const MobiusCritScene = EntityFrame{PosSpeedCritical, BoundingBoxDef, Int}
MobiusCritScene(n::Int=100) = Frame(MobiusCritEntity, n)

AutomotiveDrivingModels.get_center(veh::MobiusCritEntity) = veh.state.s
AutomotiveDrivingModels.get_footpoint(veh::MobiusCritEntity) = veh.state.s
AutomotiveDrivingModels.get_front(veh::MobiusCritEntity) = veh.state.s + veh.def.len/2
AutomotiveDrivingModels.get_rear(veh::MobiusCritEntity) = veh.state.s - veh.def.len/2

function AutomotiveDrivingModels.get_headway{R}(veh_rear::MobiusCritEntity, veh_fore::MobiusCritEntity, roadway::R)
    return get_headway(get_front(veh_rear), get_rear(veh_fore), roadway)
end
function AutomotiveDrivingModels.get_neighbor_fore{R}(scene::MobiusCritScene, vehicle_index::Int, roadway::R)
    ego = scene[vehicle_index]
    best_ind = 0
    best_gap = Inf
    for (i,veh) in enumerate(scene)
        if i != vehicle_index
            Δs = get_headway(ego, veh, roadway)
            if Δs > 0 && Δs < best_gap
                best_gap, best_ind = Δs, i
            end
        end
    end
    return NeighborLongitudinalResult(best_ind, best_gap)
end
function AutomotiveDrivingModels.get_neighbor_rear{R}(
    scene::MobiusCritScene,
    vehicle_index::Int,
    roadway::R,
    )

    ego = scene[vehicle_index]
    best_ind = 0
    best_gap = Inf
    for (i,veh) in enumerate(scene)
        if i != vehicle_index
            Δs = get_headway(veh, ego, roadway)
            if Δs > 0 && Δs < best_gap
                best_gap, best_ind = Δs, i
            end
        end
    end
    return NeighborLongitudinalResult(best_ind, best_gap)
end


function MobiusCritScene(scene::MobiusScene, critical_cars::Vector{Int})
    retval = MobiusCritScene(length(scene))
    for (i,veh) in enumerate(scene)
        state = PosSpeedCritical(veh.state.s, veh.state.v, i ∈ critical_cars)
        push!(retval, MobiusCritEntity(state, veh.def, veh.id))
    end
    return retval
end


v(scene::MobiusCritScene, vehicle_indices::Assignment, roadway::MobiusRoadway) = FeatureValue(scene[vehicle_indices[1]].state.v)

function Δv(scene::MobiusCritScene, vehicle_indices::Assignment, roadway::MobiusRoadway)
    v_rear = scene[vehicle_indices[1]].state.v
    v_fore = scene[vehicle_indices[2]].state.v
    return FeatureValue(v_fore - v_rear)
end
Δv_crit(scene::MobiusCritScene, vehicle_indices::Assignment, roadway::MobiusRoadway) = Δv(scene, vehicle_indices, roadway)


function Δs(scene::MobiusCritScene, vehicle_indices::Assignment, roadway::MobiusRoadway)
    vehA = scene[vehicle_indices[1]]
    vehB = scene[vehicle_indices[2]]
    return FeatureValue(get_headway(vehA, vehB, roadway))
end
Δs_crit(scene::MobiusCritScene, vehicle_indices::Assignment, roadway::MobiusRoadway) = Δs(scene, vehicle_indices, roadway)

function AutoScenes.assign_metric{F <: typeof(v)}(f::F, scene::MobiusCritScene, roadway::MobiusRoadway)
    return Assignment[(i,) for i in 1: length(scene)]
end
function AutoScenes.assign_metric{F <: Union{typeof(Δv), typeof(Δs)}}(f::F, scene::MobiusCritScene, roadway::MobiusRoadway)

    lead_follow = LeadFollowRelationships(scene, roadway)

    assignments = Assignment[]
    for (i, rear) in enumerate(scene)
        j = lead_follow.index_fore[i]
        fore = scene[j]
        if !rear.state.iscrit && !fore.state.iscrit
            push!(assignments, (i,j))
        end
    end

    return assignments
end
function AutoScenes.assign_metric{F <: Union{typeof(Δv_crit), typeof(Δs_crit)}}(f::F, scene::MobiusCritScene, roadway::MobiusRoadway)

    lead_follow = LeadFollowRelationships(scene, roadway)

    assignments = Assignment[]
    for (i, rear) in enumerate(scene)
        j = lead_follow.index_fore[i]
        fore = scene[j]
        if rear.state.iscrit && fore.state.iscrit
            push!(assignments, (i,j))
        end
    end

    return assignments
end