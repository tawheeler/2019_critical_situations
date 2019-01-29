using Distributions
using Parameters

const Dset = Vector{Vector{Float64}}

μ(X::Dset, m::Function) = [m(x) for x in X]
Σ(X::Dset, k::Function) = [k(x,x′) for x in X, x′ in X]
K(X::Dset, X′::Dset, k::Function) = [k(x,x′) for x in X, x′ in X′]

μ!(μ::Vector{Float64}, X::Dset, m::Function) = map!(m, μ, X)
function K!(K::Matrix{Float64}, X::Dset, X′::Dset, k::Function)
    for (i,x) in enumerate(X)
        for (j,x′) in enumerate(X′)
            K[i,j] = k(x,x′)
        end
    end
    return K
end
function K!(K::Matrix{Float64}, x::Vector{Float64}, X′::Dset, k::Function)
    for (j,x′) in enumerate(X′)
        K[1,j] = k(x,x′)
    end
    return K
end
Σ!(Σ::Matrix{Float64}, X::Dset, k::Function) = K!(Σ, X, X, k)

@with_kw mutable struct GaussianProcess
    m::Function = x -> 0.0
    k::Function = (x,x′)->exp(-(norm(x-x′))^2)
    X::Vector{Vector{Float64}} = Vector{Float64}[] # points sampled thus far
    y::Vector{Float64} = Float64[] # results sampled thus far
    ν::Float64 = 0.0 # variance when sampling f

    # matrices allocated for prediction
    M₁::Matrix{Float64} = Matrix{Float64}(0,0)
    M₂::Matrix{Float64} = Matrix{Float64}(0,0)
    M₃::Vector{Float64} = Vector{Float64}(0)
    M₄::Matrix{Float64} = Matrix{Float64}(0,0)
end

function _allocate_predict_memory!(GP::GaussianProcess)
    m, k, ν, X, n = GP.m, GP.k, GP.ν, GP.X, length(GP.X)

    GP.M₁ = inv(K(X, X, k) + ν*I)
    GP.M₂ = Matrix{Float64}(1, n)
    GP.M₃ = μ(X, m)
    GP.M₄ = Matrix{Float64}(n, 1)

    return GP
end

mvnrand(μ::Vector{Float64}, Σ::Matrix{Float64}, inflation=1e-6) = rand(MvNormal(μ, Σ + inflation*I));
Base.rand(GP::GaussianProcess, X::Dset) = mvnrand(μ(X, GP.m), Σ(X, GP.k))
function Base.push!(GP::GaussianProcess, x::Vector{Float64}, y::Real)
    push!(GP.X, x)
    push!(GP.y, y)
    _allocate_predict_memory!(GP)
    return GP
end
function Base.pop!(GP::GaussianProcess)
    pop!(GP.X)
    pop!(GP.y)
    _allocate_predict_memory!(GP)
    return GP
end

function predict(GP::GaussianProcess, x_pred::Vector{Float64})

    m, k, X = GP.m, GP.k, GP.X
    M₁, M₂, M₃, M₄ = GP.M₁, GP.M₂, GP.M₃, GP.M₄

    K!(M₂, x_pred, X, k)
    M₄ = copy!(M₄, M₂) # same but transpose
    M₂ *= M₁

    μₚ = m(x_pred) + (M₂*(GP.y - M₃))[1]
    νₚ = max(k(x_pred, x_pred) - (M₂*M₄)[1], eps(Float64))

    return (μₚ, νₚ)
end
function predict!(
    μₚ::Vector{Float64},
    νₚ::Vector{Float64},
    GP::GaussianProcess,
    X_pred::Dset,
    )

    n = length(X)

    if n == 0 # no data
        μ!(μₚ, X_pred, GP.m)
        fill!(νₚ, GP.ν)
    else
        for (i,x_pred) in enumerate(X_pred)
            tup = predict(GP, x_pred)
            μₚ[i] = tup[1]
            νₚ[i] = tup[2]
        end
    end

    return (μₚ, νₚ)
end
function predict(GP::GaussianProcess, X_pred::Dset)
    m = length(X_pred)
    μₚ = Array{Float64}(m)
    νₚ = Array{Float64}(m)
    return predict!(μₚ, νₚ, GP, X_pred)
end

upperbound(μₚ::Float64, νₚ::Float64, β::Real) = μₚ + sqrt(β*νₚ)
function upperbound(GP::GaussianProcess, x_pred::Vector{Float64}, β::Real)
    μₚ, νₚ = predict(GP, x_pred)
    return upperbound(μₚ, νₚ, β)
end
lowerbound(μₚ::Float64, νₚ::Float64, β::Real) = μₚ - sqrt(β*νₚ)
function lowerbound(GP::GaussianProcess, x_pred::Vector{Float64}, β::Real)
    μₚ, νₚ = predict(GP, x_pred)
    return lowerbound(μₚ, νₚ, β)
end
width(νₚ::Float64, β::Real) = 2sqrt(β*νₚ)
function width(GP::GaussianProcess, x_pred::Vector{Float64}, β::Real)
    μₚ, νₚ = predict(GP, x_pred)
    return width(νₚ, β)
end

prob_of_improvement(N::Normal{Float64}, y_min::Real) = isapprox(N.σ, 0, atol=1e-4) ? 0.0 : cdf(N, y_min)
prob_of_improvement(y_min::Float64, μ::Float64, ν::Float64) = prob_of_improvement(Normal(μ, ν), y_min)
function expected_improvement(y_min::Float64, μ::Float64, ν::Float64; ν_atol::Float64=1e-4)
    if isnan(ν) || isinf(ν) || isapprox(ν, 0.0, atol=ν_atol)
        return 0.0
    end
    σ = sqrt(ν)
    p_imp = prob_of_improvement(y_min, μ, ν)
    p_ymin = pdf(Normal(μ, σ), y_min)
    return (y_min - μ)*p_imp + σ*p_ymin
end
expected_improvement(y_min::Float64, N::Normal; ν_atol::Float64=1e-4) = expected_improvement(y_min, mean(N), var(N), ν_atol=ν_atol)


function select_sample_index_via_expected_improvement(
    GP::GaussianProcess,
    μₚ::Vector{Float64},
    νₚ::Vector{Float64},
    sampled::BitVector,
    )

    best_index = -1
    best_expected_improvement = 0.0
    y_min = minimum(GP.y)
    for (i,(μ, ν)) in enumerate(zip(μₚ,νₚ))
        if !sampled[i]
            ei = expected_improvement(y_min, μ, ν)
            if ei > best_expected_improvement
                best_expected_improvement = ei
                best_index = i
            end
        end
    end
    return best_index
end

# optimize_via_expected_improvement

# f(x; a=1, b=5) = (a-x[1])^2 + b*(x[2] - x[1]^2)^2

# X = Vector{Float64}[]
# for x₁ in linspace(-1,3,11)
#     for x₂ in linspace(-2,2,11)
#         push!(X, [x₁, x₂])
#     end
# end

# GP = GaussianProcess(m = x -> 10.0)

# n = length(X)
# μₚ = Array{Float64}(n)
# νₚ = Array{Float64}(n)

# push!(GP, [0.0,0.0], f([0.0,0.0]))

# for i in 1 : 25
#     predict!(μₚ, νₚ, GP, X)
#     best_index = select_sample_index_via_expected_improvement(GP, μₚ, νₚ)
#     x = X[best_index]
#     y = f(x)
#     @printf("x = [%8.3f, %8.3f]  y = %8.3f\n", x[1], x[2], y)
#     push!(GP, x, y)
# end

