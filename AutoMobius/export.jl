struct IDMDef
    δ::Float64
    T::Float64
    k_spd::Float64
    v_des::Float64
    s_min::Float64
    a_max::Float64
    d_cmf::Float64
    d_max::Float64
end
IDMDef(idm::IntelligentDriverModel) = IDMDef(idm.δ, idm.T, idm.k_spd, idm.v_des, idm.s_min, idm.a_max, idm.d_cmf, idm.d_max)
function Base.write(io::IO, ::MIME"text/plain", d::IDMDef)
    @printf(io, "%.16e %.16e %.16e %.16e %.16e %.16e %.16e %.16e",
        d.δ, d.T, d.k_spd, d.v_des, d.s_min, d.a_max, d.d_cmf, d.d_max)
end
function Base.read(io::IO, ::MIME"text/plain", ::Type{IDMDef})
    i = 0
    tokens = split(strip(readline(io)), ' ')
    δ = parse(Float64, tokens[i+=1])
    T = parse(Float64, tokens[i+=1])
    k_spd = parse(Float64, tokens[i+=1])
    v_des = parse(Float64, tokens[i+=1])
    s_min = parse(Float64, tokens[i+=1])
    a_max = parse(Float64, tokens[i+=1])
    d_cmf = parse(Float64, tokens[i+=1])
    d_max = parse(Float64, tokens[i+=1])
    return IDMDef(δ, T, k_spd, v_des, s_min, a_max, d_cmf, d_max)
end
function set_idmdef(idm::IntelligentDriverModel, def::IDMDef)
    idm.δ = def.δ
    idm.T = def.T
    idm.k_spd = def.k_spd
    idm.v_des = def.v_des
    idm.s_min = def.s_min
    idm.a_max = def.a_max
    idm.d_cmf = def.d_cmf
    idm.d_max = def.d_max
    return idm
end
get_timidness(idmdef::IDMDef) = get_timidness(set_idmdef(IntelligentDriverModel(), idmdef))

####

const AttentiveMobiusState = Tuple{PosSpeed1D, AttentiveDriverState, DelayedDriverState}
const AttentiveMobiusDef = Tuple{BoundingBoxDef, IDMDef}
const AttentiveMobiusEntity = Entity{AttentiveMobiusState, AttentiveMobiusDef, Int}
const AttentiveMobiusScene = EntityFrame{AttentiveMobiusState, AttentiveMobiusDef, Int}
AttentiveMobiusScene(n::Int=100) = Frame(AttentiveMobiusEntity, n)

Base.convert(::Type{MobiusEntity}, entity::AttentiveMobiusEntity) = MobiusEntity(entity.state[1], entity.def[1], entity.id)

function AutomotiveDrivingModels.get_headway(
    veh_rear::AttentiveMobiusEntity,
    veh_fore::AttentiveMobiusEntity,
    roadway::MobiusRoadway,
    )

    return get_headway(convert(MobiusEntity, veh_rear),
                       convert(MobiusEntity, veh_fore),
                       roadway)
end
function AutomotiveDrivingModels.get_neighbor_fore(
    scene::AttentiveMobiusScene,
    vehicle_index::Int,
    roadway::MobiusRoadway,
    )

    ego = convert(MobiusEntity, scene[vehicle_index])
    best_ind = 0
    best_gap = Inf
    for (i,veh) in enumerate(scene)
        if i != vehicle_index
            Δs = get_headway(ego, convert(MobiusEntity, veh), roadway)
            if Δs > 0 && Δs < best_gap
                best_gap, best_ind = Δs, i
            end
        end
    end
    return NeighborLongitudinalResult(best_ind, best_gap)
end
function AutomotiveDrivingModels.get_neighbor_rear(
    scene::AttentiveMobiusScene,
    vehicle_index::Int,
    roadway::MobiusRoadway,
    )

    ego = convert(MobiusEntity, scene[vehicle_index])
    best_ind = 0
    best_gap = Inf
    for (i,veh) in enumerate(scene)
        if i != vehicle_index
            Δs = get_headway(convert(MobiusEntity, veh), ego, roadway)
            if Δs > 0 && Δs < best_gap
                best_gap, best_ind = Δs, i
            end
        end
    end
    return NeighborLongitudinalResult(best_ind, best_gap)
end
# function AutoScenes.LeadFollowRelationships(scene::AttentiveMobiusScene, roadway::MobiusRoadway, vehicle_indices::AbstractVector{Int} = 1:length(scene))

#     nvehicles = length(scene)
#     index_fore = zeros(Int, nvehicles)
#     index_rear = zeros(Int, nvehicles)

#     for vehicle_index in vehicle_indices
#         index_fore[vehicle_index] = get_neighbor_fore(scene, vehicle_index, roadway).ind
#         index_rear[vehicle_index] = get_neighbor_rear(scene, vehicle_index, roadway).ind
#     end

#     return LeadFollowRelationships(index_fore, index_rear)
# end

function Base.write(io::IO, mime::MIME"text/plain", s::AttentiveMobiusState)
    write(io, mime, s[1])
    print(io, "\n")
    write(io, mime, s[2])
    print(io, "\n")
    write(io, mime, s[3])
    return nothing
end
function Base.write(io::IO, mime::MIME"text/plain", s::AttentiveMobiusDef)
    write(io, mime, s[1])
    print(io, "\n")
    write(io, mime, s[2])
    return nothing
end

function Base.read(io::IO, mime::MIME"text/plain", ::Type{AttentiveMobiusState})
    s = read(io, mime, PosSpeed1D)
    d1 = read(io, mime, AttentiveDriverState)
    d2 = read(io, mime, DelayedDriverState)
    return (s,d1,d2)
end
function Base.read(io::IO, mime::MIME"text/plain", ::Type{AttentiveMobiusDef})
    a = read(io, mime, BoundingBoxDef)
    b = read(io, mime, IDMDef)
    return (a,b)
end

const PackedListRecord = ListRecord{AttentiveMobiusState,AttentiveMobiusDef,Int}

get_attentive_delayed_driver_state(model::LaneFollowingDriver)::Tuple{AttentiveDriverState, DelayedDriverState} = (model.state, model.submodel.submodel.state)

function pack_output_list_record{Dr<:LaneFollowingDriver}(
    qrec::QueueRecord{MobiusEntity},
    driverstate_rec::QueueRecord{Entity{Tuple{AttentiveMobiusState, DelayedDriverState},Void,Int}},
    models::Dict{Int,Dr},
    )

    S, D, I = AttentiveMobiusState, AttentiveMobiusDef, Int

    frames = Array{RecordFrame}(nframes(qrec))
    states = Array{RecordState{S,I}}(nstates(qrec))
    defs = Dict{I, D}()

    lo = 1
    for (i,pastframe) in enumerate(1-nframes(qrec) : 0)
        frame = qrec[pastframe]

        hi = lo-1
        for entity in frame
            hi += 1
            id = entity.id
            defs[id] = (entity.def, IDMDef(models[id].submodel.submodel))
            driver_entity = get_by_id(driverstate_rec[pastframe], id)
            states[hi] = RecordState{S,I}((entity.state, driver_entity.state), entity.id)
        end

        frames[i] = RecordFrame(lo, hi)
        lo = hi + 1
    end

    return ListRecord{S,D,I}(get_timestep(qrec), frames, states, defs)
end
function pack_attentive_mobius_frame{Dr<:LaneFollowingDriver}(
    frame::MobiusScene,
    models::Dict{Int,Dr},
    )

    S, D, I = AttentiveMobiusState, AttentiveMobiusDef, Int
    retval = AttentiveMobiusScene(length(frame))
    for (i,entity) in enumerate(frame)
        id = entity.id
        def = (entity.def, IDMDef(models[id].submodel.submodel))
        state = (entity.state, get_attentive_delayed_driver_state(models[id])...)
        push!(retval, Entity(state, def, id))
    end
    return retval
end

function unpack!(scene::MobiusScene, packedscene::AttentiveMobiusScene)
    empty!(scene)
    for entity in packedscene
        veh = MobiusEntity(entity.state[1], entity.def[1], entity.id)
        push!(scene, veh)
    end
    return scene
end
function unpack!{D<:LaneFollowingDriver}(models::Dict{Int,D}, packedscene::AttentiveMobiusScene, timestep::Float64=0.1)
    for entity in packedscene

        if !haskey(models, entity.id)
            submodel = StochasticLaneFollowingDriver(IntelligentDriverModel(), Normal(0.0,0.5))
            submodel = DelayedDriver(submodel, DelayedDriverState(2))
            models[entity.id] = LaneFollowingDriver(AttentiveLaneFollowingDriver(submodel=submodel, timestep=timestep))
        end

        models[entity.id].state = entity.state[2]
        models[entity.id].submodel.submodel.state = entity.state[3]
        set_idmdef(models[entity.id].submodel.submodel, entity.def[2])
    end
    return models
end
function unpack!{D<:LaneFollowingDriver}(
    scene::MobiusScene,
    models::Dict{Int,D},
    packedscene::AttentiveMobiusScene,
    timestep::Float64=0.1,
    )

    unpack!(scene, packedscene)
    unpack!(models, packedscene)
    return (scene, models)
end

function unpack!(
    rec::PackedListRecord,
    frame_index::Int,
    scene::MobiusScene;
    scene_extract::Frame{Entity{AttentiveMobiusState, AttentiveMobiusDef, Int}} = Frame(Entity{AttentiveMobiusState, AttentiveMobiusDef, Int}, n_objects_in_frame(rec, frame_index)),
    )

    get!(scene_extract, rec, frame_index)
    unpack!(scene, scene_extract)
    return scene
end
function unpack!{D<:LaneFollowingDriver}(
    rec::PackedListRecord,
    frame_index::Int,
    models::Dict{Int,D};
    scene_extract::Frame{Entity{AttentiveMobiusState, AttentiveMobiusDef, Int}} = Frame(Entity{AttentiveMobiusState, AttentiveMobiusDef, Int}, n_objects_in_frame(rec, frame_index)),
    )

    get!(scene_extract, rec, frame_index)
    unpack!(models, scene_extract, get_timestep(rec))
    return models
end
function unpack!{D<:LaneFollowingDriver}(
    rec::PackedListRecord,
    frame_index::Int,
    scene::MobiusScene,
    models::Dict{Int,D};
    scene_extract::AttentiveMobiusScene = Frame(AttentiveMobiusEntity, n_objects_in_frame(rec, frame_index)),
    )

    unpack!(rec, frame_index, scene, scene_extract=scene_extract)
    unpack!(rec, frame_index, models, scene_extract=scene_extract)
    return (scene, models)
end


function export_betas_array{B<:Beta}(io::IO, betas_arr::Vector{B})
    for β in betas_arr
        @printf(io, "%.3f %.3f\n", β.α, β.β)
    end
end
function import_betas_array(io::IO)
    betas_arr = Beta[]
    for line in readlines(io)
        tokens = split(strip(line), ' ')
        a = parse(Float64, tokens[1])
        b = parse(Float64, tokens[2])
        push!(betas_arr, Beta(a,b))
    end
    return betas_arr
end