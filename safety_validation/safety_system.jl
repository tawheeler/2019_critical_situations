const HumanModel = WrappedLaneFollowingDriver{StoppingAccel,AttentiveLaneFollowingDriver{StochasticLaneFollowingDriver{IntelligentDriverModel,Normal{Float64}}}}

"""
    AEB_TTC_point

A point which determines the TTC vs. speed for an AEB.
A single point defines a line segment in the TTC vs. speed profile,
whose lower, left point is (v,TTC) and with slope given by ΔTTC_per_v
"""
struct AEB_TTC_point
    TTC::Float64 # [s]
    v::Float64 # [m/s]
    ΔTTC_per_v::Float64 # [1/m]
end
const AEB_TTC_Profile = Vector{AEB_TTC_point} # in sorted order

"""
    get(AEB_TTC_Profile, v)

Returns the TTC for the given speed.
"""
function Base.get(profile::AEB_TTC_Profile, v::Float64)
    if v < profile[1].v
        return 0.0
    else
        i = 1
        while i < length(profile) && profile[i+1].v < v
            i += 1
        end
        p = profile[i]
        return p.v + p.ΔTTC_per_v*(v-p.TTC)
    end
end

"""
Based on the Autonomous Emergency Braking system in ArXiV 1605.04965
"""
@with_kw mutable struct ZhaoAEB <: DriverModel{StoppingAccel}
    human::HumanModel
    brake::Float64 = -10.0 # [m/s²]
    brake_rate_limit::Float64 = -16.0*0.1 # [m/s² per tick]
    ttc_profile::AEB_TTC_Profile = [AEB_TTC_point(1.11111,0.835,0.0075)]
    ticks_since_activation::Int = 0 # [s]
end
InfinitiAEB(human::HumanModel) = ZhaoAEB(human,
    brake = -5.8,
    brake_rate_limit = -12.0*0.1,
    ttc_profile = [
        AEB_TTC_point(4.16667,0.49,0.03456),
        AEB_TTC_point(8.33334,0.634,0.04464),
        AEB_TTC_point(15.5242,0.955,0.0),
    ],
)
VolvoAEB(human::HumanModel) = ZhaoAEB(
    human = human,
    brake = -10.0,
    brake_rate_limit = -16.0*0.1,
    ttc_profile = [
        AEB_TTC_point(1.11111,0.835,0.0075),
        AEB_TTC_point(11.11111,0.91,0.0),
    ],
)

AutomotiveDrivingModels.get_name(model::ZhaoAEB) = "ZhaoAEB"
function AutomotiveDrivingModels.track_longitudinal!(model::ZhaoAEB, v_ego::Float64, v_oth::Float64, headway::Float64)

    # run the submodel either way, to keep it up to date
    track_longitudinal!(model.human, v_ego, v_oth, headway)

    ticks_since_activation_prev = model.ticks_since_activation
    model.ticks_since_activation = 0

    Δv = v_oth - v_ego
    if Δv < 0.0 && 0 < headway < Inf  # we are approaching front
        ttc = -headway / Δv
        ttc_threshold = get(model.ttc_profile, v_ego)
        if ttc < ttc_threshold
            model.ticks_since_activation = ticks_since_activation_prev + 1
        end
    end

    return model
end
AutoMobius.get_attentive_driver_state(model::ZhaoAEB) = get_attentive_driver_state(model.human.submodel)

function Base.rand(model::ZhaoAEB)
    if model.ticks_since_activation > 0
        return StoppingAccel(min(model.brake, model.ticks_since_activation*model.brake_rate_limit))
    else
        return rand(model.human)
    end
end
Distributions.pdf(model::ZhaoAEB, a::StoppingAccel) = pdf(model.human, a)
Distributions.logpdf(model::ZhaoAEB, a::StoppingAccel) = logpdf(model.human, a)

mutable struct ZhaoAEBOverlay <: SceneOverlay
    target_id::Int
    model::ZhaoAEB
end
function AutoViz.render!(rendermodel::RenderModel, overlay::ZhaoAEBOverlay, scene::MobiusScene, roadway::MobiusRoadway)

    render!(rendermodel, AttentionOverlay(overlay.target_id, overlay.model.human.submodel), scene, roadway)
    if overlay.model.ticks_since_activation > 0
        #render a red transparent rectangle
        ind = findfirst(scene, overlay.target_id)
        if ind > 0
            veh = scene[ind]
            color = RGBA(1.0,0.0,0.0,0.5)
            add_instruction!(rendermodel, render_round_rect,
                    (veh.state.s, 0.0, veh.def.len*1.2, veh.def.wid*1.2, 1.0, 0.5, color))
        end
    end


    return rendermodel
end



