const SPEED_LO =  0.0
const SPEED_HI = 30.0

#########################################
#     Global Parameterization
#
# used for training
#########################################

"""
Assign the vars. We use position, speed, and length of each vehicle.
"""
function AutoScenes.Vars(scene::MobiusScene, roadway::Straight1DRoadway)

    vars = Vars(0)

    lead_follow = LeadFollowRelationships(scene, roadway)

    j = 0
    for (vehicle_index, veh) in enumerate(scene)

        rear = lead_follow.index_rear[vehicle_index]
        fore = lead_follow.index_fore[vehicle_index]


        # only allow vehicles with leading and trailing counterparts to move
        if rear != 0 && fore != 0
            Δs_fore = get_headway(veh, scene[fore], roadway)
            Δs_rear = get_headway(scene[rear], veh, roadway)
            bounds_s = StateBounds(-Δs_rear, Δs_fore)
        else
            bounds_s = ZERO_BOUND
        end

        bounds_v = StateBounds(SPEED_LO - veh.state.v, SPEED_HI - veh.state.v)
        bounds_l = ZERO_BOUND

        push!(vars, veh.state.s, bounds_s, :s, vehicle_index)
        push!(vars, veh.state.v, bounds_v, :v, vehicle_index)
        push!(vars, veh.def.len, bounds_l, :l, vehicle_index)
    end

    return vars
end

invlerp(lo::Real, val::Real, hi::Real, a::Real=0, b::Real=1)::Float64 = lerp(a, b, (val - lo) / (hi - lo))

#=
The features we want are reverse lerped versions of:

 v,  v², v³
Δv, Δv², Δv³
Δs, Δs², Δs³
vᵣ⋅Δv, vᵣ⋅Δs, Δv⋅Δs
=#
v(vars::Vars, a::Assignment, roadway::Straight1DRoadway) = invlerp(SPEED_LO, vars.values[a[1]], SPEED_HI)
v²(vars::Vars, a::Assignment, roadway::Straight1DRoadway) = v(vars, a, roadway)^2
v³(vars::Vars, a::Assignment, roadway::Straight1DRoadway) = v(vars, a, roadway)^3
function Δv(vars::Vars, a::Assignment, roadway::Straight1DRoadway)
    v_rear = vars.values[a[1]]
    v_fore = vars.values[a[2]]
    Δv_max = SPEED_HI - SPEED_LO
    return invlerp(-Δv_max, v_fore - v_rear, Δv_max, -1, 1)
end
Δv²(vars::Vars, a::Assignment, roadway::Straight1DRoadway) = Δv(vars, a, roadway)^2
Δv³(vars::Vars, a::Assignment, roadway::Straight1DRoadway) = Δv(vars, a, roadway)^3
function Δs(vars::Vars, a::Assignment, roadway::Straight1DRoadway)
    sa = vars.values[a[1]]
    sb = vars.values[a[2]]
    la = vars.values[a[3]]
    lb = vars.values[a[4]]

    vehA = MobiusEntity(PosSpeed1D(sa, NaN), BoundingBoxDef(AgentClass.CAR, la, NaN), 1)
    vehB = MobiusEntity(PosSpeed1D(sb, NaN), BoundingBoxDef(AgentClass.CAR, lb, NaN), 1)
    ab = get_headway(vehA, vehB, roadway)
    # @printf("%10.6f  %10.6f  %10.6f  %10.6f  %10.6f  %10.6f  %10.6f  %10.6f  %10.6f\n", sa, sb, la, lb, sa+la/2, sb-lb/2, ab, log(ab + 1), invlerp(log(1.0), log(ab + 1), log(50.0+1)))

    return invlerp(log(1.0), log(ab + 1), log(50.0+1))
end
Δs²(vars::Vars, a::Assignment, roadway::Straight1DRoadway) = Δs(vars, a, roadway)^2
Δs³(vars::Vars, a::Assignment, roadway::Straight1DRoadway) = Δs(vars, a, roadway)^3
vᵣΔv(vars::Vars, a::Assignment, roadway::Straight1DRoadway) = v(vars, (a[1],), roadway)*Δv(vars, a, roadway)
vᵣΔs(vars::Vars, a::Assignment, roadway::Straight1DRoadway) = v(vars, (a[5],), roadway)*Δs(vars, a[[1,2,3,4]], roadway)
ΔvΔs(vars::Vars, a::Assignment, roadway::Straight1DRoadway) = Δv(vars, a[[1,2]], roadway)*Δs(vars, a[[3,4,5,6]], roadway)

function AutoScenes.assign_feature{F <: Union{typeof(v), typeof(v²), typeof(v³)}}(
    f::F,
    scene::MobiusScene,
    roadway::Straight1DRoadway,
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
    roadway::Straight1DRoadway,
    vars::Vars,
    )

    lead_follow = LeadFollowRelationships(scene, roadway)

    assignments = Assignment[]
    for rear in 1 : length(scene)
        fore = lead_follow.index_fore[rear]
        if fore != 0
            push!(assignments, (findfirst(vars, rear, :v), findfirst(vars, fore, :v)))
        end
    end

    return assignments
end
function AutoScenes.assign_feature{F <: Union{typeof(Δs), typeof(Δs²), typeof(Δs³)}}(
    f::F,
    scene::MobiusScene,
    roadway::Straight1DRoadway,
    vars::Vars,
    )

    lead_follow = LeadFollowRelationships(scene, roadway)

    assignments = Assignment[]
    for rear in 1 : length(scene)
        fore = lead_follow.index_fore[rear]
        if fore != 0
            push!(assignments, (findfirst(vars, rear, :s), findfirst(vars, fore, :s),
                                findfirst(vars, rear, :l), findfirst(vars, fore, :l)))
        end
    end

    return assignments
end
function AutoScenes.assign_feature{F <: typeof(vᵣΔs)}(
    f::F,
    scene::MobiusScene,
    roadway::Straight1DRoadway,
    vars::Vars,
    )

    lead_follow = LeadFollowRelationships(scene, roadway)

    assignments = Assignment[]
    for rear in 1 : length(scene)
        fore = lead_follow.index_fore[rear]
        if fore != 0
            push!(assignments, (findfirst(vars, rear, :s), findfirst(vars, fore, :s),
                                findfirst(vars, rear, :l), findfirst(vars, fore, :l),
                                findfirst(vars, rear, :v)))
        end
    end

    return assignments
end
function AutoScenes.assign_feature{F <: typeof(ΔvΔs)}(
    f::F,
    scene::MobiusScene,
    roadway::Straight1DRoadway,
    vars::Vars,
    )

    lead_follow = LeadFollowRelationships(scene, roadway)

    assignments = Assignment[]
    for rear in 1 : length(scene)
        fore = lead_follow.index_fore[rear]
        if fore != 0
            push!(assignments, (findfirst(vars, rear, :v), findfirst(vars, fore, :v),
                                findfirst(vars, rear, :s), findfirst(vars, fore, :s),
                                findfirst(vars, rear, :l), findfirst(vars, fore, :l)))
        end
    end

    return assignments
end