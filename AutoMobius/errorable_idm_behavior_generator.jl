mutable struct ErrorableIDMBehaviorGenerator # <: BehaviorGenerator
    timestep::Float64
    IDMgen::CorrelatedIDMBehaviorGenerator

    function ErrorableIDMBehaviorGenerator(timestep::Float64,
        IDMgen::CorrelatedIDMBehaviorGenerator=CorrelatedIDMBehaviorGenerator())

        return new(timestep, IDMgen)
    end
end

function Base.rand(gen::ErrorableIDMBehaviorGenerator)

    driver = rand(gen.IDMgen)
    driver = DelayedDriver(driver, DelayedDriverState(2)) # 0.2 ms delay
    driver = AttentiveLaneFollowingDriver(submodel=driver, timestep=gen.timestep)
#     driver = ThresholdedRangeRatePerceptionLaneFollowingDriver(submodel=driver)

    LaneFollowingDriver(driver)
end

function Base.rand!(
    models::Dict{Int, LaneFollowingDriver},
    gen::ErrorableIDMBehaviorGenerator,
    scene::MobiusScene,
    )

    empty!(models)
    for veh in scene
        models[veh.id] = rand(gen)
    end
    return models
end