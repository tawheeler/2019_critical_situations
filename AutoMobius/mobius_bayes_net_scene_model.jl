mutable struct BayesNets1DSceneModel
    bn::DiscreteBayesNet
    discs::Dict{Symbol,LinearDiscretizer}
    rng::MersenneTwister
end
function BayesNets1DSceneModel()
    # BayesNets1DSceneModel(JLD.load(joinpath(splitdir(@__FILE__)[1], "../data/1dbnscenemodel.jld"), "bn"),
    #                       JLD.load(joinpath(splitdir(@__FILE__)[1], "../data/1dbnscenemodel.jld"), "discs"), MersenneTwister(0))

    bn = open(joinpath(splitdir(@__FILE__)[1], "../data/1dbnscenemodel_bn.txt"), "r") do io
                read(io, MIME"text/plain"(), DiscreteBayesNet)
            end
    discs = open(joinpath(splitdir(@__FILE__)[1], "../data/1dbnscenemodel_discs.txt"), "r") do io
                read(io, MIME"text/plain"(), Dict{Symbol, LinearDiscretizer})
            end
    return BayesNets1DSceneModel(bn, discs, MersenneTwister(0))
end
Base.show(io::IO, ::BayesNets1DSceneModel) = print(io, "BayesNets1DSceneModel")

Base.srand(scene_generator::BayesNets1DSceneModel, seed::Int64=0) = srand(scene_generator.rng, seed)

function discretize_dataset(df_cont::DataFrame, discs::Dict{Symbol,LinearDiscretizer})
    df_disc = DataFrame()
    for (sym,D) in discs
        df_disc[sym] = encode(D, df_cont[sym])
    end
    return df_disc
end

function Discretizers.decode(A::Assignment, discs::Dict{Symbol, LinearDiscretizer})
    for (sym,D) in discs
        A[sym] = decode(D, A[sym])
    end
    return A
end
sample_continuous_value(bn::DiscreteBayesNet, discs::Dict{Symbol, LinearDiscretizer}) = decode(rand(bn), discs)

function Base.rand!(scene::MobiusScene, scene_generator::BayesNets1DSceneModel, roadway::MobiusRoadway;
    def::BoundingBoxDef = BoundingBoxDef(), # used for its length
    )

    bn = scene_generator.bn
    discs = scene_generator.discs

    v_arr = Float64[]
    s_arr = Float64[]
    half_length = def.len/2

    # sample the first vehicle
    ϕ = infer(bn, :tailway) # marginal over tailway
    A = sample_continuous_value(bn, discs)
    push!(v_arr, A[:speed])
    push!(s_arr, get_s_max(roadway) - half_length - rand(Uniform(0.0, decode(discs[:tailway], sample(weights(ϕ.potential))))))
    s_next = s_arr[end] - def.len - A[:tailway]
    v_next = A[:speed] - A[:relspeed]
    while s_next ≥ half_length
        push!(s_arr, s_next)
        push!(v_arr, v_next)
        v_next_disc = encode(discs[:speed], v_next)
        A, w = get_weighted_sample!(A, bn, :speed=>v_next_disc) # we know the model structure so this will always work
        decode(A, discs)
        s_next -= (def.len + A[:tailway])
        v_next -= A[:relspeed]
    end

    empty!(scene)
    for (i,j) in enumerate(length(s_arr) : -1 : 1) # leftmost to rightmost vehicle
        state = PosSpeed1D(s_arr[j], v_arr[j])
        push!(scene, Entity(state, def, i))
    end

    return scene
end

"""
discs = open("1dbnscenemodel_discs.txt", "w") do io
            write(io, MIME"text/plain"(), discs)
        end
"""
function Base.write{L<:LinearDiscretizer}(io::IO, ::MIME"text/plain", ::Dict{Symbol,L})
    for (key,disc) in discs
        println(io, key)
        for (i,e) in enumerate(binedges(disc))
            print(io,  e, i == nlabels(disc) + 1 ? "\n" : " ")
        end
    end
    return nothing
end

"""
discs = open("1dbnscenemodel_discs.txt", "r") do io
            read(io, MIME"text/plain"(), Dict{Symbol, LinearDiscretizer})
        end
"""
function Base.read(io::IO, ::MIME"text/plain", ::Type{Dict{Symbol, LinearDiscretizer}})
    retval = Dict{Symbol, LinearDiscretizer{Float64,Int}}()

    lines = readlines(io)
    for i in 1 : 2 : length(lines)
        name = Symbol(split(lines[i])[1])
        binedges = [parse(Float64, s) for s in split(lines[i+1])]
        retval[name] = LinearDiscretizer(binedges)
    end
    return retval
end