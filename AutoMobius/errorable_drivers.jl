struct AttentiveDriverState
    is_attentive::Bool
    steps_since_attention_swap::Int

    v_ego_log::Float64
    v_oth_log::Float64
    headway_log::Float64
end
AttentiveDriverState() = AttentiveDriverState(true, 0, NaN, NaN, NaN)

function Base.write(io::IO, ::MIME"text/plain", state::AttentiveDriverState)
    print(io, state.is_attentive, " ")
    @printf(io, "%d %.16e %.16e %.16e", state.steps_since_attention_swap, state.v_ego_log, state.v_oth_log, state.headway_log)
end
function Base.read(io::IO, ::MIME"text/plain", ::Type{AttentiveDriverState})
    i = 0
    tokens = split(strip(readline(io)), ' ')
    is_attentive = parse(Bool, tokens[i+=1])
    steps_since_attention_swap = parse(Int, tokens[i+=1])
    v_ego_log = parse(Float64, tokens[i+=1])
    v_oth_log = parse(Float64, tokens[i+=1])
    headway_log = parse(Float64, tokens[i+=1])
    return AttentiveDriverState(is_attentive, steps_since_attention_swap, v_ego_log, v_oth_log, headway_log)
end

####


import AutomotiveDrivingModels: get_name, set_desired_speed!, track_longitudinal!

@with_kw mutable struct AttentiveLaneFollowingDriver{D<:LaneFollowingDriver} <: LaneFollowingDriver
    timestep::Float64
    submodel::D
    state::AttentiveDriverState = AttentiveDriverState()

    P_seconds_to_attentive::LogNormal = LogNormal(0.6133, 0.8934)
    P_seconds_to_distracted::LogNormal = LogNormal(3.2817, 0.7516)
end
get_name(model::AttentiveLaneFollowingDriver) = "Attentive" * get_name(model.submodel)
function set_desired_speed!(model::AttentiveLaneFollowingDriver, v_des::Float64)
    set_desired_speed!(model.submodel, v_des)
    return model
end

"""
    Probability of transition in [t_lo, t_hi] given that
    it did not occur in [0, t_lo]
"""
function get_probability_of_transition_at(P::LogNormal, t_lo::Float64, t_hi::Float64)
    cdf_lo = cdf(P, t_lo)
    cdf_hi = cdf(P, t_hi)
    return (cdf_hi - cdf_lo) / (1 - cdf_lo)
end
function track_longitudinal!(model::AttentiveLaneFollowingDriver, v_ego::Float64, v_oth::Float64, headway::Float64)

    model_is_attentive = model.state.is_attentive
    model_steps_since_attention_swap = model.state.steps_since_attention_swap
    model_v_ego_log = model.state.v_ego_log
    model_v_oth_log = model.state.v_oth_log
    model_headway_log = model.state.headway_log

    if model_is_attentive || isnan(model_v_ego_log)
        # store measurements
        model_v_ego_log = v_ego
        model_v_oth_log = v_oth
        model_headway_log = headway
    else
        # use stored measurements
        v_ego = model_v_ego_log
        v_oth = model_v_oth_log
        headway = model_headway_log
    end

    # transition
    t_lo = model_steps_since_attention_swap*model.timestep
    t_hi = t_lo + model.timestep

    if model_is_attentive
        if rand() ≤ get_probability_of_transition_at(model.P_seconds_to_distracted, t_lo, t_hi)
            model_is_attentive = false
            model_steps_since_attention_swap = 0
        else
            model_steps_since_attention_swap += 1
        end
    else
        if rand() ≤ get_probability_of_transition_at(model.P_seconds_to_attentive, t_lo, t_hi)
            model_is_attentive = true
            model_steps_since_attention_swap = 0
        else
            model_steps_since_attention_swap += 1
        end
    end

    model.state = AttentiveDriverState(model_is_attentive, model_steps_since_attention_swap,
                    model_v_ego_log, model_v_oth_log, model_headway_log)

    track_longitudinal!(model.submodel, v_ego, v_oth, headway)

    return model
end
Base.rand(model::AttentiveLaneFollowingDriver) = rand(model.submodel)
Distributions.pdf(model::AttentiveLaneFollowingDriver, a::Accel) = pdf(model.submodel, a)
Distributions.logpdf(model::AttentiveLaneFollowingDriver, a::Accel) = logpdf(model.submodel, a)

mutable struct AttentionOverlay <: SceneOverlay
    target_id::Int
    model::AttentiveLaneFollowingDriver
end
function AutoViz.render!(rendermodel::RenderModel, overlay::AttentionOverlay, scene::MobiusScene, roadway::MobiusRoadway)
    ind = findfirst(scene, overlay.target_id)
    if ind > 0
        veh = scene[ind]
        color = overlay.model.state.is_attentive ? colorant"green" : colorant"red"
        add_instruction!(rendermodel, render_circle,
                (veh.state.s, 5.0, 1.5, color, colorant"white"))
    end
    return rendermodel
end

######

# immutable ThresholdedRangeRatePerceptionState
#     prev_range_rate::Float64
# end
# ThresholdedRangeRatePerceptionState() = ThresholdedRangeRatePerceptionState(NaN)

# @with_kw type ThresholdedRangeRatePerceptionLaneFollowingDriver{D<:LaneFollowingDriver} <: LaneFollowingDriver
#     submodel::D
#     state::ThresholdedRangeRatePerceptionState = ThresholdedRangeRatePerceptionState()
#     threshold::Float64 = 0.1 # [m/s]
# end
# get_name(model::ThresholdedRangeRatePerceptionLaneFollowingDriver) = "ThresholdedRangeRatePerception" * get_name(model.submodel)
# function set_desired_speed!(model::ThresholdedRangeRatePerceptionLaneFollowingDriver, v_des::Float64)
#     set_desired_speed!(model.submodel, v_des)
#     return model
# end
# function track_longitudinal!(model::ThresholdedRangeRatePerceptionLaneFollowingDriver, v_ego::Float64, v_oth::Float64, headway::Float64)

#     range_rate = v_oth - v_ego

#     prev_range_rate = model.state.prev_range_rate

#     if isnan(prev_range_rate)
#         prev_range_rate = range_rate
#     end

#     if prev_range_rate < 1e-6 # in the case rrprev ~ 0
#         Web = 1.0
#     else
#         Web = abs((prev_range_rate - range_rate)/prev_range_rate)
#     end

#     if Web ≥ model.threshold
#         prev_range_rate = range_rate
#     else
#         range_rate = prev_range_rate
#     end

#     model.state = ThresholdedRangeRatePerceptionState(prev_range_rate)

#     v_oth = v_ego + range_rate
#     track_longitudinal!(model.submodel, v_ego, v_oth, headway)

#     return model
# end
# Base.rand(model::ThresholdedRangeRatePerceptionLaneFollowingDriver) = rand(model.submodel)
# Distributions.pdf(model::ThresholdedRangeRatePerceptionLaneFollowingDriver, a_lon::Float64) = pdf(model.submodel, a_lon)
# Distributions.logpdf(model::ThresholdedRangeRatePerceptionLaneFollowingDriver, a_lon::Float64) = logpdf(model.submodel, a_lon)