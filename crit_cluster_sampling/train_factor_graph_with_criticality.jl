const SPEED_LO = -1.5
const SPEED_HI = 41.5

#########################################
#     Global Parameterization
#
# used for training
#########################################


"""
Assign the vars. We use position, speed, length, and criticality of each vehicle.
"""
function AutoScenes.Vars(scene::MobiusScene, roadway::MobiusRoadway, critical_cars::Vector{Int}=Int[])

    vars = Vars(0)
    l_bound = ZERO_BOUND
    c_bound = ZERO_BOUND
    lead_follow = LeadFollowRelationships(scene, roadway)

    for (vehicle_index, veh) in enumerate(scene)

        rear = scene[lead_follow.index_rear[vehicle_index]]
        fore = scene[lead_follow.index_fore[vehicle_index]]

        Δs_fore = get_headway(veh, fore, roadway)
        Δs_rear = get_headway(rear, veh, roadway)

        s_bound = StateBounds(-Δs_rear, Δs_fore)
        v_bound = StateBounds(SPEED_LO - veh.state.v, SPEED_HI - veh.state.v)

        push!(vars, veh.state.s, s_bound, :s, vehicle_index)
        push!(vars, veh.state.v, v_bound, :v, vehicle_index)
        push!(vars, veh.def.len, l_bound, :l, vehicle_index)
        push!(vars, vehicle_index ∈ critical_cars, c_bound, :c, vehicle_index)
    end

    return vars
end

function get_factorgraph(scene::MobiusScene, roadway::MobiusRoadway, critical_cars::Vector{Int}, features)
    vars = Vars(scene, roadway, critical_cars)
    assignments = assign_features(features, scene, roadway, vars)
    return FactorGraph(vars, assignments, roadway)
end
function get_factorgraph(scene::MobiusScene, roadway::MobiusRoadway, features)
    vars = Vars(scene, roadway)
    assignments = assign_features(features, scene, roadway, vars)
    return FactorGraph(vars, assignments, roadway)
end

invlerp(lo::Real, val::Real, hi::Real, a::Real=0, b::Real=1)::Float64 = lerp(a, b, (val - lo) / (hi - lo))

#=
The features we want are reverse lerped versions of:

 v,  v², v³
Δv, Δv², Δv³
Δs, Δs², Δs³
vᵣ⋅Δv, vᵣ⋅Δs, Δv⋅Δs
=#
v(vars::Vars, a::Assignment, roadway::MobiusRoadway) = invlerp(SPEED_LO, vars.values[a[1]], SPEED_HI)
v²(vars::Vars, a::Assignment, roadway::MobiusRoadway) = v(vars, a, roadway)^2
v³(vars::Vars, a::Assignment, roadway::MobiusRoadway) = v(vars, a, roadway)^3
function Δv(vars::Vars, a::Assignment, roadway::MobiusRoadway)
    v_rear = vars.values[a[1]]
    v_fore = vars.values[a[2]]
    Δv_max = SPEED_HI - SPEED_LO
    return invlerp(-Δv_max, v_fore - v_rear, Δv_max, -1, 1)
end
Δv²(vars::Vars, a::Assignment, roadway::MobiusRoadway) = Δv(vars, a, roadway)^2
Δv³(vars::Vars, a::Assignment, roadway::MobiusRoadway) = Δv(vars, a, roadway)^3
function Δs(vars::Vars, a::Assignment, roadway::MobiusRoadway)
    sa = vars.values[a[1]]
    sb = vars.values[a[2]]
    la = vars.values[a[3]]
    lb = vars.values[a[4]]

    vehA = MobiusEntity(PosSpeed1D(sa, NaN), BoundingBoxDef(AgentClass.CAR, la, NaN), 1)
    vehB = MobiusEntity(PosSpeed1D(sb, NaN), BoundingBoxDef(AgentClass.CAR, lb, NaN), 1)
    ab = get_headway(vehA, vehB, roadway)
    # @printf("%10.6f  %10.6f  %10.6f\n", ab, log(ab + 1), invlerp(log(1.0), log(ab + 1), log(50.0+1)))
    return invlerp(log(1.0), log(ab + 1), log(50.0+1))
end
Δs²(vars::Vars, a::Assignment, roadway::MobiusRoadway) = Δs(vars, a, roadway)^2
Δs³(vars::Vars, a::Assignment, roadway::MobiusRoadway) = Δs(vars, a, roadway)^3
vᵣΔv(vars::Vars, a::Assignment, roadway::MobiusRoadway) = v(vars, (a[1],), roadway)*Δv(vars, a, roadway)
vᵣΔs(vars::Vars, a::Assignment, roadway::MobiusRoadway) = v(vars, (a[5],), roadway)*Δs(vars, a[[1,2,3,4]], roadway)
ΔvΔs(vars::Vars, a::Assignment, roadway::MobiusRoadway) = Δv(vars, a[[1,2]], roadway)*Δs(vars, a[[3,4,5,6]], roadway)
fₔ(vars::Vars, a::Assignment, roadway::MobiusRoadway) = vars.values[a[1]]
fₔΔv(vars::Vars, a::Assignment, roadway::MobiusRoadway) = fₐ(vars, (a[1],), roadway)*Δv(vars, a[[2,3]], roadway)
fₔΔv²(vars::Vars, a::Assignment, roadway::MobiusRoadway) = fₐ(vars, (a[1],), roadway)*Δv(vars, a[[2,3]], roadway)^2
fₔΔs(vars::Vars, a::Assignment, roadway::MobiusRoadway) = fₐ(vars, (a[1],), roadway)*Δs(vars, a[[2,3,4,5]], roadway)
fₔΔs²(vars::Vars, a::Assignment, roadway::MobiusRoadway) = fₐ(vars, (a[1],), roadway)*Δs(vars, a[[2,3,4,5]], roadway)^2

function AutoScenes.assign_feature{F <: Union{typeof(v), typeof(v²), typeof(v³)}}(
    f::F,
    scene::MobiusScene,
    roadway::MobiusRoadway,
    vars::Vars,
    )
    assignments = Assignment[]
    for (i, sym) in enumerate(vars.symbols)
        if sym == :v
            push!(assignments, (i,))
        end
    end
    return assignments
end
function AutoScenes.assign_feature{F <: Union{typeof(Δv), typeof(Δv²), typeof(Δv³), typeof(vᵣΔv)}}(
    f::F,
    scene::MobiusScene,
    roadway::MobiusRoadway,
    vars::Vars,
    )

    lead_follow = LeadFollowRelationships(scene, roadway)

    assignments = Assignment[]
    for rear in 1 : length(scene)
        fore = lead_follow.index_fore[rear]
        push!(assignments, (findfirst(vars, rear, :v), findfirst(vars, fore, :v)))
    end

    return assignments
end
function AutoScenes.assign_feature{F <: Union{typeof(Δs), typeof(Δs²), typeof(Δs³)}}(
    f::F,
    scene::MobiusScene,
    roadway::MobiusRoadway,
    vars::Vars,
    )

    lead_follow = LeadFollowRelationships(scene, roadway)

    assignments = Assignment[]
    for rear in 1 : length(scene)
        fore = lead_follow.index_fore[rear]
        push!(assignments, (findfirst(vars, rear, :s), findfirst(vars, fore, :s),
                            findfirst(vars, rear, :l), findfirst(vars, fore, :l)))
    end

    return assignments
end
function AutoScenes.assign_feature{F <: typeof(vᵣΔs)}(
    f::F,
    scene::MobiusScene,
    roadway::MobiusRoadway,
    vars::Vars,
    )

    lead_follow = LeadFollowRelationships(scene, roadway)

    assignments = Assignment[]
    for rear in 1 : length(scene)
        fore = lead_follow.index_fore[rear]
        push!(assignments, (findfirst(vars, rear, :s), findfirst(vars, fore, :s),
                            findfirst(vars, rear, :l), findfirst(vars, fore, :l),
                            findfirst(vars, rear, :v)))
    end

    return assignments
end
function AutoScenes.assign_feature{F <: typeof(ΔvΔs)}(
    f::F,
    scene::MobiusScene,
    roadway::MobiusRoadway,
    vars::Vars,
    )

    lead_follow = LeadFollowRelationships(scene, roadway)

    assignments = Assignment[]
    for rear in 1 : length(scene)
        fore = lead_follow.index_fore[rear]
        push!(assignments, (findfirst(vars, rear, :v), findfirst(vars, fore, :v),
                            findfirst(vars, rear, :s), findfirst(vars, fore, :s),
                            findfirst(vars, rear, :l), findfirst(vars, fore, :l)))
    end

    return assignments
end

#=
For critical cars we also want:

 v_crit,  v_crit²
Δv_crit, Δv_crit²
Δs_crit, Δs_crit²
Δv_crit_tame,
Δs_crit_tame,
Δv_tame_crit,
Δs_tame_crit,

TODO: try pdf(Gamma, ttc) and that squared instead, only for critical car
=#

v_crit(vars::Vars, a::Assignment, roadway::MobiusRoadway) = v(vars, a, roadway)
v_crit²(vars::Vars, a::Assignment, roadway::MobiusRoadway) = v²(vars, a, roadway)
Δv_crit(vars::Vars, a::Assignment, roadway::MobiusRoadway) = Δv(vars, a, roadway)
Δv_crit²(vars::Vars, a::Assignment, roadway::MobiusRoadway) = Δv²(vars, a, roadway)
Δs_crit(vars::Vars, a::Assignment, roadway::MobiusRoadway) = Δs(vars, a, roadway)
Δs_crit²(vars::Vars, a::Assignment, roadway::MobiusRoadway) = Δs²(vars, a, roadway)
Δv_crit_tame(vars::Vars, a::Assignment, roadway::MobiusRoadway) = Δv(vars, a, roadway)
Δs_crit_tame(vars::Vars, a::Assignment, roadway::MobiusRoadway) = Δs(vars, a, roadway)
Δv_tame_crit(vars::Vars, a::Assignment, roadway::MobiusRoadway) = Δv(vars, a, roadway)
Δs_tame_crit(vars::Vars, a::Assignment, roadway::MobiusRoadway) = Δs(vars, a, roadway)

function AutoScenes.assign_feature{F <: Union{typeof(v_crit), typeof(v_crit²)}}(
    f::F,
    scene::MobiusScene,
    roadway::MobiusRoadway,
    vars::Vars,
    )

    assignments = Assignment[]
    for (i, sym) in enumerate(vars.symbols)
        if sym == :v
            if vars.values[findfirst(vars, vars.vehicle_indices[i], :c)] == 1 # is critical
                push!(assignments, (i,))
            end
        end
    end
    return assignments
end
function AutoScenes.assign_feature{F <: Union{typeof(Δv_crit), typeof(Δv_crit²)}}(
    f::F,
    scene::MobiusScene,
    roadway::MobiusRoadway,
    vars::Vars,
    )

    lead_follow = LeadFollowRelationships(scene, roadway)

    assignments = Assignment[]
    for rear in 1 : length(scene)
        fore = lead_follow.index_fore[rear]
        if vars.values[findfirst(vars, rear, :c)] == 1 &&
           vars.values[findfirst(vars, fore, :c)] == 1
            push!(assignments, (findfirst(vars, rear, :v), findfirst(vars, fore, :v)))
        end
    end

    return assignments
end
function AutoScenes.assign_feature{F <: Union{typeof(Δs_crit), typeof(Δs_crit²)}}(
    f::F,
    scene::MobiusScene,
    roadway::MobiusRoadway,
    vars::Vars,
    )

    lead_follow = LeadFollowRelationships(scene, roadway)

    assignments = Assignment[]
    for rear in 1 : length(scene)
        fore = lead_follow.index_fore[rear]
        if vars.values[findfirst(vars, rear, :c)] == 1 &&
           vars.values[findfirst(vars, fore, :c)] == 1

            push!(assignments, (findfirst(vars, rear, :s), findfirst(vars, fore, :s),
                                findfirst(vars, rear, :l), findfirst(vars, fore, :l)))
        end
    end

    return assignments
end
function AutoScenes.assign_feature{F <: typeof(Δv_crit_tame)}(
    f::F,
    scene::MobiusScene,
    roadway::MobiusRoadway,
    vars::Vars,
    )

    lead_follow = LeadFollowRelationships(scene, roadway)

    assignments = Assignment[]
    for rear in 1 : length(scene)
        fore = lead_follow.index_fore[rear]
        if vars.values[findfirst(vars, rear, :c)] == 1 &&
           vars.values[findfirst(vars, fore, :c)] == 0
            push!(assignments, (findfirst(vars, rear, :v), findfirst(vars, fore, :v)))
        end
    end

    return assignments
end
function AutoScenes.assign_feature{F <: typeof(Δs_crit_tame)}(
    f::F,
    scene::MobiusScene,
    roadway::MobiusRoadway,
    vars::Vars,
    )

    lead_follow = LeadFollowRelationships(scene, roadway)

    assignments = Assignment[]
    for rear in 1 : length(scene)
        fore = lead_follow.index_fore[rear]
        if vars.values[findfirst(vars, rear, :c)] == 1 &&
           vars.values[findfirst(vars, fore, :c)] == 0

            push!(assignments, (findfirst(vars, rear, :s), findfirst(vars, fore, :s),
                                findfirst(vars, rear, :l), findfirst(vars, fore, :l)))
        end
    end

    return assignments
end
function AutoScenes.assign_feature{F <: typeof(Δv_tame_crit)}(
    f::F,
    scene::MobiusScene,
    roadway::MobiusRoadway,
    vars::Vars,
    )

    lead_follow = LeadFollowRelationships(scene, roadway)

    assignments = Assignment[]
    for rear in 1 : length(scene)
        fore = lead_follow.index_fore[rear]
        if vars.values[findfirst(vars, rear, :c)] == 0 &&
           vars.values[findfirst(vars, fore, :c)] == 1
            push!(assignments, (findfirst(vars, rear, :v), findfirst(vars, fore, :v)))
        end
    end

    return assignments
end
function AutoScenes.assign_feature{F <: typeof(Δs_tame_crit)}(
    f::F,
    scene::MobiusScene,
    roadway::MobiusRoadway,
    vars::Vars,
    )

    lead_follow = LeadFollowRelationships(scene, roadway)

    assignments = Assignment[]
    for rear in 1 : length(scene)
        fore = lead_follow.index_fore[rear]
        if vars.values[findfirst(vars, rear, :c)] == 0 &&
           vars.values[findfirst(vars, fore, :c)] == 1

            push!(assignments, (findfirst(vars, rear, :s), findfirst(vars, fore, :s),
                                findfirst(vars, rear, :l), findfirst(vars, fore, :l)))
        end
    end

    return assignments
end
