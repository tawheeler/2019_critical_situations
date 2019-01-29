
const PROJECT_DIR = "/home/tim/Documents/papers/2016_adas_validation/code/"

timestep = 0.1
roadway = Wraparound(Straight1DRoadway(200.0))

struct CollisionScenarioData
    list_records::Dict{String, PackedListRecord}
    risk_ests::Dict{String, Vector{Beta}}
    crit_rears::Dict{String, Int} # index
    crit_fores::Dict{String, Int} # index
end

function load_collision_scenarios(;
    data_dir::String = joinpath(PROJECT_DIR, "data/collision_scenarios_old"),
    )

    list_records = Dict{String, PackedListRecord}()
    risk_ests = Dict{String, Vector{Beta}}()
    crit_rears = Dict{String, Int}()
    crit_fores = Dict{String, Int}()

    scene = MobiusScene()

    for file in readdir(data_dir)
        if ismatch(r"collision_listrecord_sim_count_\d+_datetime_\d+_\d+_\d+.txt", file)
            filepath = joinpath(data_dir, file)
            key = file[search(file, "sim_count")[1] : end]
            list_records[key] = rec = open(filepath, "r") do io
                read(io, MIME"text/plain"(), PackedListRecord)
            end

            unpack!(rec, nframes(rec), scene) # unpack most recent scene, which should contain the collision.
            i, j = get_first_collision(scene, roadway) # in any order

            unpack!(rec, nframes(rec) - 10, scene)
            delta_s_ij = get_headway(scene[i], scene[j], roadway)
            delta_s_ji = get_headway(scene[j], scene[i], roadway)
            if delta_s_ji < delta_s_ij
                 j,i = i,j # ensure i is following j
            end

            # if loop_order(scene[i], scene[j], roadway) == -1
            #     j,i = i,j # ensure i is following j
            # end
            crit_rears[key] = i
            crit_fores[key] = j
        elseif ismatch(r"collision_estimation_sim_count_\d+_datetime_\d+_\d+_\d+.txt", file)
            filepath = joinpath(data_dir, file)
            key = file[search(file, "sim_count")[1] : end]
            risk_ests[key] = open(import_betas_array, filepath, "r")
        end
    end

    return CollisionScenarioData(
            list_records,
            risk_ests,
            crit_rears,
            crit_fores,
        )
end


struct CritEntry
    key::String
    y::Bool # whether critical
    assignment::String
end

function load_crit_entries(; filepath=joinpath(PROJECT_DIR, "data/clusters/assignment_0914.txt"))
    crit_entries = CritEntry[]
    open(filepath, "r") do io
        lines = readlines(io)
        for line in lines
            tokens = split(line, ",")
            i = parse(Int, tokens[1])
            key = tokens[2]
            y = parse(Bool, tokens[3])
            a = tokens[4]
            push!(crit_entries, CritEntry(key, y, a))
        end
    end
    return crit_entries
end
function map_cluster_names_to_indices(crit_entries::Vector{CritEntry})
    clusters = Dict{String, Vector{Int}}() # map cluster name to indices
    for (i, entry) in enumerate(crit_entries)
        if entry.y
            clusters[entry.assignment] = push!(get(clusters, entry.assignment, Int[]), i)
        end
    end
    return clusters
end
function load_crit_scenes_and_models(crit_entries::Vector{CritEntry}, dicts::CollisionScenarioData;
    risk_threshold::Float64 = 0.5,
    )

    crit_scenes = Dict{String, MobiusScene}()
    crit_models = Dict{String, Dict{Int, LaneFollowingDriver}}()
    for (i,entry) in enumerate(crit_entries)
        key = entry.key
        if !haskey(crit_scenes, key)
            rec = dicts.list_records[key]
            collision_risk = dicts.risk_ests[key]

            #DEBUG
            # frame_index = findlast(β->mean(β) < 0.5,  collision_risk)
            # est = get_collision_estimation(rec, frame_index, roadway, 0.1, nsimulations=100, nticks=100)
            # println("est: ", mean(est), "  ", sqrt(var(est)), "    expected: ", mean(collision_risk[frame_index]))

            models = Dict{Int, LaneFollowingDriver}()
            scene = unpack_critical_scene!(MobiusScene(), rec, roadway, collision_risk, risk_threshold, models=models)
            crit_scenes[key] = scene
            crit_models[key] = models
        end
    end
    return (crit_scenes, crit_models)
end
function load_benign_scenes(; filepath = joinpath(PROJECT_DIR, "data/scenes_NGSIM.txt"))
    return open(filepath) do io
        read(io, MIME"text/plain"(), Vector{MobiusScene})
    end
end
get_crit_cars(dicts::CollisionScenarioData, key::String) = [dicts.crit_rears[key], dicts.crit_fores[key]]
get_crit_cars(dicts::CollisionScenarioData, crit_entries::Vector{CritEntry}, i::Int) = get_crit_cars(dicts, crit_entries[i].key)
