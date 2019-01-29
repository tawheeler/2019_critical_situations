const SPEED_LO =  0.0
const SPEED_HI = 30.0

#########################################
#     Penalty Parameterization
#
# used for sampling
#########################################

"""
Assign the vars. We use position, speed, and length of each vehicle.
In the penalty parameterization, the s-position can vary by allowing each
vehicle to move at s ± L/2. We penalize overlap / out of order vehicles.
"""
function AutoScenes.Vars(scene::MobiusScene, roadway::MobiusRoadway)

    vars = Vars(0)
    L = get_s_max(roadway)
    s_bound = StateBounds(-L/2, L/2)
    l_bound = ZERO_BOUND

    for (vehicle_index, veh) in enumerate(scene)

        v_bound = StateBounds(SPEED_LO - veh.state.v, SPEED_HI - veh.state.v)

        push!(vars, veh.state.s, s_bound, :s, vehicle_index)
        push!(vars, veh.state.v, v_bound, :v, vehicle_index)
        push!(vars, veh.def.len, l_bound, :l, vehicle_index)
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

plus the penalty function

out_of_order

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
function out_of_order(vars::Vars, a::Assignment, roadway::MobiusRoadway)

    # a < b < c

    sa = vars.values[a[1]]
    sb = vars.values[a[2]]
    sc = vars.values[a[3]]
    la = vars.values[a[4]]
    lb = vars.values[a[5]]
    lc = vars.values[a[6]]

    # true if in violation

    vehA = MobiusEntity(PosSpeed1D(sa, NaN), BoundingBoxDef(AgentClass.CAR, la, NaN), 1)
    vehB = MobiusEntity(PosSpeed1D(sb, NaN), BoundingBoxDef(AgentClass.CAR, lb, NaN), 1)
    vehC = MobiusEntity(PosSpeed1D(sc, NaN), BoundingBoxDef(AgentClass.CAR, lc, NaN), 1)

    ab = get_headway(vehA, vehB, roadway)
    ac = get_headway(vehA, vehC, roadway)
    return ac < ab || is_colliding(vehA, vehB, roadway) ||
                      is_colliding(vehA, vehC, roadway) ||
                      is_colliding(vehB, vehC, roadway)
end

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
function AutoScenes.assign_feature{F <: typeof(out_of_order)}(
    f::F,
    scene::MobiusScene,
    roadway::MobiusRoadway,
    vars::Vars,
    )

    lead_follow = LeadFollowRelationships(scene, roadway)

    assignments = Assignment[]
    for i in 1 : 2 : length(scene)
        j = lead_follow.index_fore[i]
        k = lead_follow.index_fore[j]

        push!(assignments, (findfirst(vars, i, :s), findfirst(vars, j, :s), findfirst(vars, k, :s),
                            findfirst(vars, i, :l), findfirst(vars, j, :l), findfirst(vars, k, :l)))
    end

    return assignments
end

function create_penalty_model_from_global(model::FactorModel)
    features = tuple(model.features..., out_of_order)
    weights = vcat(model.weights, -Inf)
    return FactorModel(features, weights)
end

function shift_scene!(dest::MobiusScene, src::MobiusScene, Δ::Vector{Float64}, factorgraph::FactorGraph)
    empty!(dest)
    vars = factorgraph.vars
    for (i,veh) in enumerate(src)
        i_s = findfirst(vars, i, :s)
        i_v = findfirst(vars, i, :v)
        i_l = findfirst(vars, i, :l)

        s = mod_position_to_roadway(factorgraph.vars.values[i_s] + Δ[i_s], factorgraph.roadway)
        v = factorgraph.vars.values[i_v] + Δ[i_v]
        l = factorgraph.vars.values[i_l]

        state = PosSpeed1D(s, v)
        def = BoundingBoxDef(AgentClass.CAR, l, 2.0)
        push!(dest, MobiusEntity(state, def, veh.id),)
    end
    return dest
end