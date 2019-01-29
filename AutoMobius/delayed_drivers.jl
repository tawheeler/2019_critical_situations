import DataStructures: CircularBuffer, capacity

struct LongitudinalReading
    v_ego::Float64
    v_oth::Float64
    headway::Float64
end

const DelayedDriverState = CircularBuffer{LongitudinalReading}

function Base.write(io::IO, ::MIME"text/plain", state::DelayedDriverState)
    print(io, capacity(state))
    for r in state
        @printf(io, " %.16e %.16e %.16e", r.v_ego, r.v_oth, r.headway)
    end
end
function Base.read(io::IO, ::MIME"text/plain", ::Type{DelayedDriverState})
    i = 0
    tokens = split(strip(readline(io)), ' ')
    state = DelayedDriverState(parse(Int, tokens[i+=1]))
    while i < length(tokens)
        v_ego = parse(Float64, tokens[i+=1])
        v_oth = parse(Float64, tokens[i+=1])
        headway = parse(Float64, tokens[i+=1])
        push!(state, LongitudinalReading(v_ego, v_oth, headway))
    end
    return state
end

####


import AutomotiveDrivingModels: get_name, set_desired_speed!, track_longitudinal!

mutable struct DelayedDriver{D<:LaneFollowingDriver} <: LaneFollowingDriver
    submodel::D
    state::DelayedDriverState
end
get_name(model::DelayedDriver) = "Delayed" * get_name(model.submodel)
function set_desired_speed!(model::DelayedDriver, v_des::Float64)
    set_desired_speed!(model.submodel, v_des)
    return model
end
function track_longitudinal!(model::DelayedDriver, v_ego::Float64, v_oth::Float64, headway::Float64)
    r = LongitudinalReading(v_ego, v_oth, headway)
    o = get(model.state, 1, r)
    track_longitudinal!(model.submodel, o.v_ego, o.v_oth, o_headway)
    push!(model.state, r)
    return model
end
Base.rand(model::DelayedDriver) = rand(model.submodel)
Distributions.pdf(model::DelayedDriver, a::Accel) = pdf(model.submodel, a)
Distributions.logpdf(model::DelayedDriver, a::Accel) = logpdf(model.submodel, a)