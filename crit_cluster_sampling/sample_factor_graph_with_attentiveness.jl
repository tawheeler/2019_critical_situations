const SPEED_LO = -2.0
const SPEED_HI = 39.0

using Polynomials
function legendre(i)
    n = i-1
    p = Poly([-1,0,1])^n
    for i in 1 : n
        p = polyder(p)
    end
    return p / (2^n * factorial(n))
end
const L1 = legendre(1)
const L2 = legendre(2)
const L3 = legendre(3)
const L4 = legendre(4)

#########################################
#     Penalty Parameterization
#
# used for sampling
#########################################


"""
Assign the vars:
mutable attributes: position (s), speed (v), timidness (t)
static attributes: length (l), attentiveness (a), criticality (c)
"""
function AutoScenes.Vars(packedscene::AttentiveMobiusScene, roadway::MobiusRoadway, critical_cars::Vector{Int})

    vars = Vars(0)
    L = get_s_max(roadway)
    l_bound = ZERO_BOUND
    a_bound = ZERO_BOUND
    c_bound = ZERO_BOUND
    s_bound = StateBounds(-L/2, L/2)

    scene = unpack!(MobiusScene(length(packedscene)), packedscene)
    lead_follow = LeadFollowRelationships(scene, roadway)

    for (vehicle_index, entity) in enumerate(packedscene)

        t = get_timidness(entity.def[2])
        v_bound = StateBounds(SPEED_LO - entity.state[1].v, SPEED_HI - entity.state[1].v)
        t_bound = StateBounds(-t, 1-t)

        push!(vars, entity.state[1].s,             s_bound, :s, vehicle_index)
        push!(vars, entity.state[1].v,             v_bound, :v, vehicle_index)
        push!(vars, entity.def[1].len,             l_bound, :l, vehicle_index)
        push!(vars, t,                             t_bound, :t, vehicle_index)
        push!(vars, entity.state[2].is_attentive,  a_bound, :a, vehicle_index)
        push!(vars, vehicle_index ∈ critical_cars, c_bound, :c, vehicle_index)
    end

    return vars
end

function get_factorgraph(scene::AttentiveMobiusScene, roadway::MobiusRoadway, critical_cars::Vector{Int}, features)
    vars = Vars(scene, roadway, critical_cars)
    assignments = assign_features(features, scene, roadway, vars)
    return FactorGraph(vars, assignments, roadway)
end

_invlerp(lo::Real, val::Real, hi::Real, a::Real=0, b::Real=1)::Float64 = lerp(a, b, (val - lo) / (hi - lo))


#=
The features we want are reverse lerped versions of:
L123(v), L123(Δv), L123(Δs), L1(vᵣ⋅Δv), L1(vᵣ⋅Δs), L1(Δv⋅Δs), L12(tᵣ⋅Δv), L12(tᵣ⋅Δs)
L12(aᵣ⋅Δs), L12(aᵣ⋅Δv)
=#
f_v(vars::Vars, a::Assignment, roadway::MobiusRoadway) = _invlerp(SPEED_LO, vars.values[a[1]], SPEED_HI, -1, 1)
v( vars::Vars, a::Assignment, roadway::MobiusRoadway) = L1(f_v(vars, a, roadway))
v²(vars::Vars, a::Assignment, roadway::MobiusRoadway) = L2(f_v(vars, a, roadway))
v³(vars::Vars, a::Assignment, roadway::MobiusRoadway) = L3(f_v(vars, a, roadway))

function f_Δv(vars::Vars, a::Assignment, roadway::MobiusRoadway)
    v_rear = vars.values[a[1]]
    v_fore = vars.values[a[2]]
    Δv_max = SPEED_HI - SPEED_LO
    return _invlerp(-Δv_max, v_fore - v_rear, Δv_max, -1, 1)
end
Δv( vars::Vars, a::Assignment, roadway::MobiusRoadway) = L1(f_Δv(vars, a, roadway))
Δv²(vars::Vars, a::Assignment, roadway::MobiusRoadway) = L2(f_Δv(vars, a, roadway))
Δv³(vars::Vars, a::Assignment, roadway::MobiusRoadway) = L3(f_Δv(vars, a, roadway))

function f_Δs(vars::Vars, a::Assignment, roadway::MobiusRoadway)
    sa = vars.values[a[1]]
    sb = vars.values[a[2]]
    la = vars.values[a[3]]
    lb = vars.values[a[4]]

    vehA = MobiusEntity(PosSpeed1D(sa, NaN), BoundingBoxDef(AgentClass.CAR, la, NaN), 1)
    vehB = MobiusEntity(PosSpeed1D(sb, NaN), BoundingBoxDef(AgentClass.CAR, lb, NaN), 1)
    ab = get_headway(vehA, vehB, roadway)
    # @printf("%10.6f  %10.6f  %10.6f\n", ab, log(ab + 1), _invlerp(log(1.0), log(ab + 1), log(50.0+1)))
    return _invlerp(log(1.0), log(ab + 1), log(50.0+1), -1, 1)
end
Δs( vars::Vars, a::Assignment, roadway::MobiusRoadway) = L1(f_Δs(vars, a, roadway))
Δs²(vars::Vars, a::Assignment, roadway::MobiusRoadway) = L2(f_Δs(vars, a, roadway))
Δs³(vars::Vars, a::Assignment, roadway::MobiusRoadway) = L3(f_Δs(vars, a, roadway))

vᵣΔv(vars::Vars, a::Assignment, roadway::MobiusRoadway) = L1(f_v(vars, (a[1],), roadway)*f_Δv(vars, a, roadway))
vᵣΔs(vars::Vars, a::Assignment, roadway::MobiusRoadway) = L1(f_v(vars, (a[5],), roadway)*f_Δs(vars, a[[1,2,3,4]], roadway))
ΔvΔs(vars::Vars, a::Assignment, roadway::MobiusRoadway) = L1(f_Δv(vars, a[[1,2]], roadway)*f_Δs(vars, a[[3,4,5,6]], roadway))

f_t(vars::Vars, a::Assignment, roadway::MobiusRoadway) = _invlerp(0, vars.values[a[1]], 1, -1, 1)
tᵣΔv( vars::Vars, a::Assignment, roadway::MobiusRoadway) = L1(f_t(vars, (a[1],), roadway) * f_Δv(vars, (a[2], a[3]), roadway))
tᵣΔv²(vars::Vars, a::Assignment, roadway::MobiusRoadway) = L2(f_t(vars, (a[1],), roadway) * f_Δv(vars, (a[2], a[3]), roadway))
tᵣΔs( vars::Vars, a::Assignment, roadway::MobiusRoadway) = L1(f_t(vars, (a[1],), roadway) * f_Δs(vars, a[[2,3,4,5]], roadway))
tᵣΔs²(vars::Vars, a::Assignment, roadway::MobiusRoadway) = L2(f_t(vars, (a[1],), roadway) * f_Δs(vars, a[[2,3,4,5]], roadway))

f_a(vars::Vars, a::Assignment, roadway::MobiusRoadway) = _invlerp(0, vars.values[a[1]], 1, -1, 1)
aᵣΔv( vars::Vars, a::Assignment, roadway::MobiusRoadway) = L1(f_a(vars, (a[1],), roadway) * f_Δv(vars, (a[2], a[3]), roadway))
aᵣΔv²(vars::Vars, a::Assignment, roadway::MobiusRoadway) = L2(f_a(vars, (a[1],), roadway) * f_Δv(vars, (a[2], a[3]), roadway))
aᵣΔs( vars::Vars, a::Assignment, roadway::MobiusRoadway) = L1(f_a(vars, (a[1],), roadway) * f_Δs(vars, a[[2,3,4,5]], roadway))
aᵣΔs²(vars::Vars, a::Assignment, roadway::MobiusRoadway) = L2(f_a(vars, (a[1],), roadway) * f_Δs(vars, a[[2,3,4,5]], roadway))
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
    scene::AttentiveMobiusScene,
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
    scene::AttentiveMobiusScene,
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
    scene::AttentiveMobiusScene,
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
    scene::AttentiveMobiusScene,
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
    scene::AttentiveMobiusScene,
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
function AutoScenes.assign_feature{F <: Union{typeof(tᵣΔv), typeof(tᵣΔv²)}}(
    f::F,
    scene::AttentiveMobiusScene,
    roadway::MobiusRoadway,
    vars::Vars,
    )

    lead_follow = LeadFollowRelationships(scene, roadway)

    assignments = Assignment[]
    for rear in 1 : length(scene)
        fore = lead_follow.index_fore[rear]
        push!(assignments, (findfirst(vars, rear, :t), findfirst(vars, rear, :v), findfirst(vars, fore, :v)))
    end

    return assignments
end
function AutoScenes.assign_feature{F <: Union{typeof(tᵣΔs), typeof(tᵣΔs²)}}(
    f::F,
    scene::AttentiveMobiusScene,
    roadway::MobiusRoadway,
    vars::Vars,
    )

    lead_follow = LeadFollowRelationships(scene, roadway)

    assignments = Assignment[]
    for rear in 1 : length(scene)
        fore = lead_follow.index_fore[rear]
        push!(assignments, (findfirst(vars, rear, :t),
                            findfirst(vars, rear, :s), findfirst(vars, fore, :s),
                            findfirst(vars, rear, :l), findfirst(vars, fore, :l)))
    end

    return assignments
end
function AutoScenes.assign_feature{F <: Union{typeof(aᵣΔv), typeof(aᵣΔv²)}}(
    f::F,
    scene::AttentiveMobiusScene,
    roadway::MobiusRoadway,
    vars::Vars,
    )

    lead_follow = LeadFollowRelationships(scene, roadway)

    assignments = Assignment[]
    for rear in 1 : length(scene)
        fore = lead_follow.index_fore[rear]
        push!(assignments, (findfirst(vars, rear, :a), findfirst(vars, rear, :v), findfirst(vars, fore, :v)))
    end

    return assignments
end
function AutoScenes.assign_feature{F <: Union{typeof(aᵣΔs), typeof(aᵣΔs²)}}(
    f::F,
    scene::AttentiveMobiusScene,
    roadway::MobiusRoadway,
    vars::Vars,
    )

    lead_follow = LeadFollowRelationships(scene, roadway)

    assignments = Assignment[]
    for rear in 1 : length(scene)
        fore = lead_follow.index_fore[rear]
        push!(assignments, (findfirst(vars, rear, :a),
                            findfirst(vars, rear, :s), findfirst(vars, fore, :s),
                            findfirst(vars, rear, :l), findfirst(vars, fore, :l)))
    end

    return assignments
end
function AutoScenes.assign_feature{F <: typeof(out_of_order)}(
    f::F,
    scene::AttentiveMobiusScene,
    roadway::MobiusRoadway,
    vars::Vars,
    )

    lead_follow = LeadFollowRelationships(scene, roadway)

    assignments = Assignment[]
    for i in 1 : length(scene)
        j = lead_follow.index_fore[i]
        k = lead_follow.index_fore[j]

        push!(assignments, (findfirst(vars, i, :s), findfirst(vars, j, :s), findfirst(vars, k, :s),
                            findfirst(vars, i, :l), findfirst(vars, j, :l), findfirst(vars, k, :l)))
    end

    return assignments
end

#=
For critical cars we also want:

crit: L12(v)
crit_crit: L12(Δv), L12(Δs)
crit_tame: L1(Δv), L1(Δs)
tame_crit: L1(Δv), L1(Δs)
=#

v_crit(vars::Vars, a::Assignment, roadway::MobiusRoadway) = L1(f_v(vars, a, roadway))
v_crit²(vars::Vars, a::Assignment, roadway::MobiusRoadway) = L2(f_v(vars, a, roadway))
Δv_crit(vars::Vars, a::Assignment, roadway::MobiusRoadway) = L1(f_Δv(vars, a, roadway))
Δv_crit²(vars::Vars, a::Assignment, roadway::MobiusRoadway) = L2(f_Δv(vars, a, roadway))
Δs_crit(vars::Vars, a::Assignment, roadway::MobiusRoadway) = L1(f_Δs(vars, a, roadway))
Δs_crit²(vars::Vars, a::Assignment, roadway::MobiusRoadway) = L2(f_Δs(vars, a, roadway))
Δv_crit_tame(vars::Vars, a::Assignment, roadway::MobiusRoadway) = L1(f_Δv(vars, a, roadway))
Δs_crit_tame(vars::Vars, a::Assignment, roadway::MobiusRoadway) = L1(f_Δs(vars, a, roadway))
Δv_tame_crit(vars::Vars, a::Assignment, roadway::MobiusRoadway) = L1(f_Δv(vars, a, roadway))
Δs_tame_crit(vars::Vars, a::Assignment, roadway::MobiusRoadway) = L1(f_Δs(vars, a, roadway))

function AutoScenes.assign_feature{F <: Union{typeof(v_crit), typeof(v_crit²)}}(
    f::F,
    scene::AttentiveMobiusScene,
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
    scene::AttentiveMobiusScene,
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
    scene::AttentiveMobiusScene,
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
    scene::AttentiveMobiusScene,
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
    scene::AttentiveMobiusScene,
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
    scene::AttentiveMobiusScene,
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
    scene::AttentiveMobiusScene,
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

###

function create_penalty_model_from_global(model::FactorModel)
    features = tuple(model.features..., out_of_order)
    weights = vcat(model.weights, -Inf)
    return FactorModel(features, weights)
end

function shift_scene!(dest::AttentiveMobiusScene, src::AttentiveMobiusScene, Δ::Vector{Float64}, factorgraph::FactorGraph)
    empty!(dest)
    vars = factorgraph.vars
    for (i,veh) in enumerate(src)
        i_s = findfirst(vars, i, :s)
        i_v = findfirst(vars, i, :v)
        i_l = findfirst(vars, i, :l)
        i_t = findfirst(vars, i, :t)

        s = mod_position_to_roadway(factorgraph.vars.values[i_s] + Δ[i_s], factorgraph.roadway)
        v = factorgraph.vars.values[i_v] + Δ[i_v]
        l = factorgraph.vars.values[i_l]
        t = factorgraph.vars.values[i_t] + Δ[i_t]

        state = (PosSpeed1D(s, v), src[i].state[2])
        def = (BoundingBoxDef(AgentClass.CAR, l, 2.0), IDMDef(get_correlated_IDM(t)))
        push!(dest, AttentiveMobiusEntity(state, def, veh.id))
    end
    return dest
end
