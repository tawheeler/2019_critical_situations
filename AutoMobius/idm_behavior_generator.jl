const IDM_AGGRESSIVE = IntelligentDriverModel(
        k_spd =  1.0,
        δ     =  4.0,
        T     =  1.5, # 1.0
        v_des = 38.9,
        s_min =  0.0,
        a_max =  2.0,
        d_cmf =  3.0,
        d_max =  5.0,
    )
const IDM_TIMID = IntelligentDriverModel(
        k_spd =  0.9,
        δ     =  4.0,
        T     =  2.0,
        v_des = 27.8,
        s_min =  4.0,
        a_max =  0.8,
        d_cmf =  1.0,
        d_max =  5.0,
    )

_get_uniform(a::Real, b::Real) = a == b ? Normal(a,eps()) : a < b ? Uniform(a,b) : Uniform(b,a)

"""
Extract the timidness used to obtain an IDM driver
For IDM drivers generated with CorrelatedIDMBehaviorGenerator the extraction is exact.
For other IDM drivers it is approximate.
α = 0 is an aggressive driver
α = 1 is a timid driver
"""
function get_timidness(idm::IntelligentDriverModel)
    α  = clamp(invlerp(IDM_TIMID.k_spd, IDM_AGGRESSIVE.k_spd,idm.k_spd), 0, 1)
    α += clamp(invlerp(IDM_TIMID.T,     IDM_AGGRESSIVE.T,    idm.T), 0, 1)
    α += clamp(invlerp(IDM_TIMID.v_des, IDM_AGGRESSIVE.v_des,idm.v_des), 0, 1)
    α += clamp(invlerp(IDM_TIMID.s_min, IDM_AGGRESSIVE.s_min,idm.s_min), 0, 1)
    α += clamp(invlerp(IDM_TIMID.a_max, IDM_AGGRESSIVE.a_max,idm.a_max), 0, 1)
    α += clamp(invlerp(IDM_TIMID.d_cmf, IDM_AGGRESSIVE.d_cmf,idm.d_cmf), 0, 1)
    return α / 6
end
get_timidness(model::StochasticLaneFollowingDriver{IntelligentDriverModel}) = get_timidness(model.submodel)

"""
A distribution over the linear interpolation between aggressive and timid
α = 0 is an aggressive driver
α = 1 is a timid driver
"""
@with_kw mutable struct CorrelatedIDMBehaviorGenerator
    D_α::ContinuousUnivariateDistribution = Uniform(0.0,1.0)
    σ::Float64 = 0.5
end
function get_correlated_IDM(α::Float64)
    IntelligentDriverModel(
            k_spd = lerp(IDM_TIMID.k_spd, IDM_AGGRESSIVE.k_spd,α),
            δ     = lerp(IDM_TIMID.δ,     IDM_AGGRESSIVE.δ,    α),
            T     = lerp(IDM_TIMID.T,     IDM_AGGRESSIVE.T,    α),
            v_des = lerp(IDM_TIMID.v_des, IDM_AGGRESSIVE.v_des,α),
            s_min = lerp(IDM_TIMID.s_min, IDM_AGGRESSIVE.s_min,α),
            a_max = lerp(IDM_TIMID.a_max, IDM_AGGRESSIVE.a_max,α),
            d_cmf = lerp(IDM_TIMID.d_cmf, IDM_AGGRESSIVE.d_cmf,α),
            d_max = lerp(IDM_TIMID.d_max, IDM_AGGRESSIVE.d_max,α),
        )
end
function Base.rand(gen::CorrelatedIDMBehaviorGenerator)
    α = rand(gen.D_α)
    idm = get_correlated_IDM(α)
    return StochasticLaneFollowingDriver(idm, Normal(0.0,gen.σ))
end

@with_kw mutable struct UncorrelatedIDMBehaviorGenerator
    # This assumes all are independent
    D_k_spd::ContinuousUnivariateDistribution = _get_uniform(IDM_TIMID.k_spd, IDM_AGGRESSIVE.k_spd)
    D_δ::ContinuousUnivariateDistribution     = _get_uniform(IDM_TIMID.δ,     IDM_AGGRESSIVE.δ)
    D_T::ContinuousUnivariateDistribution     = _get_uniform(IDM_TIMID.T,     IDM_AGGRESSIVE.T)
    D_v_des::ContinuousUnivariateDistribution = _get_uniform(IDM_TIMID.v_des, IDM_AGGRESSIVE.v_des)
    D_s_min::ContinuousUnivariateDistribution = _get_uniform(IDM_TIMID.s_min, IDM_AGGRESSIVE.s_min)
    D_a_max::ContinuousUnivariateDistribution = _get_uniform(IDM_TIMID.a_max, IDM_AGGRESSIVE.a_max)
    D_d_cmf::ContinuousUnivariateDistribution = _get_uniform(IDM_TIMID.d_cmf, IDM_AGGRESSIVE.d_cmf)
    D_d_max::ContinuousUnivariateDistribution = _get_uniform(IDM_TIMID.d_max, IDM_AGGRESSIVE.d_max)
    σ::Float64 = 0.5
end
function Base.rand(gen::UncorrelatedIDMBehaviorGenerator)
    return StochasticLaneFollowingDriver(IntelligentDriverModel(
            k_spd = rand(gen.D_k_spd),
            δ     = rand(gen.D_δ    ),
            T     = rand(gen.D_T    ),
            v_des = rand(gen.D_v_des),
            s_min = rand(gen.D_s_min),
            a_max = rand(gen.D_a_max),
            d_cmf = rand(gen.D_d_cmf),
            d_max = rand(gen.D_d_max),
        ), Normal(0.0,gen.σ))
end