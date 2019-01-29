using AutomotiveDrivingModels
using AutoScenes
using Records
using Distributions

include("AutoMobius/AutoMobius.jl")
using AutoMobius

##################################################
# save scenario data
##################################################

roadway = Wraparound(Straight1DRoadway(200.0))

const TIMESTEP = 0.1
const RISK_THRESHOLD = 0.1
const DATA_DIR = "/media/tim/DATAPART1/PublicationData/2017_adas_validation/"
const OUTPUT_FILE_SUFFIX = "1019"
const COLLISION_SCENARIO_DIR = joinpath(DATA_DIR, "collision_scenarios_$OUTPUT_FILE_SUFFIX")

const NSIMS = 1000
const NTICKS = 100 # simulation horizon
const SCENE = MobiusScene()
const MODELS = Dict{Int, LaneFollowingDriver}()
const REC = QueueRecord(MobiusEntity, 2, TIMESTEP)
const COLLISION_CALLBACK = MobiusCollisionCallback()

struct MCRiskAssessment
    nsims::Int
    nticks::Int
    ncol::Int # number of collisions observed
end
get_risk_beta(a::MCRiskAssessment, prior::Float64=1.0) = Beta(prior + a.ncol, prior + a.nsims - a.ncol)
get_risk(a::MCRiskAssessment, prior::Float64=1.0) = mean(get_risk_beta(a, prior))
function monte_carlo_risk_assessment(
    packedscene::AttentiveMobiusScene,
    roadway::MobiusRoadway;
    nsims::Int = 100,
    nticks::Int = 100, # simulation horizon
    scene::MobiusScene = MobiusScene(),
    models::Dict{Int, LaneFollowingDriver} = Dict{Int, LaneFollowingDriver}(),
    rec::QueueRecord{MobiusEntity} = QueueRecord(MobiusEntity, 2, TIMESTEP),
    collision_callback::MobiusCollisionCallback = MobiusCollisionCallback(),
    )

    ncol = 0
    for i in 1 : nsims
        unpack!(scene, models, packedscene, get_timestep(rec))
        if has_collision(scene, roadway)
            ncol += 1
        else
            empty!(rec)
            simulate!(rec, scene, roadway, models, nticks, (collision_callback,))
            @assert nframes(rec) > 1
            ncol += has_collision(rec[0], roadway)
        end
    end

    return MCRiskAssessment(nsims, nticks, ncol)
end

struct MCBlameAssessment
    risk_assessment::MCRiskAssessment
    blame_counts::Dict{Tuple{Int,Int}, Int} # (rear_ind, lead_ind) -> count
end
function get_lead_follow_ordering(scene::MobiusScene, i::Int, j::Int, roadway::MobiusRoadway)
    delta_s_ij = get_headway(scene[i], scene[j], roadway)
    delta_s_ji = get_headway(scene[j], scene[i], roadway)
    if delta_s_ji < delta_s_ij
        i,j = j,i
    end
    return (i,j)
end
function monte_carlo_blame_assignment(
    packedscene::AttentiveMobiusScene,
    roadway::MobiusRoadway;
    nsims::Int = 100,
    nticks::Int = 100, # simulation horizon
    scene::MobiusScene = MobiusScene(),
    models::Dict{Int, LaneFollowingDriver} = Dict{Int, LaneFollowingDriver}(),
    rec::QueueRecord{MobiusEntity} = QueueRecord(MobiusEntity, 2, TIMESTEP),
    collision_callback::MobiusCollisionCallback = MobiusCollisionCallback(),
    )


    blame_counts = Dict{Tuple{Int,Int}, Int}() # (rear_ind, lead_ind) -> count

    ncol = 0
    for i in 1 : nsims
        unpack!(scene, models, packedscene, get_timestep(rec))
        @assert !has_collision(scene, roadway)


        empty!(rec)
        simulate!(rec, scene, roadway, models, nticks, (collision_callback,))
        @assert nframes(rec) > 1
        pair = get_first_collision(rec[0], roadway)
        if pair != (0,0)
            # had collision
            ncol += 1
            pair = get_lead_follow_ordering(rec[-1],pair...,roadway)
            blame_counts[pair] = get(blame_counts, pair, 0) + 1
        end
    end

    return MCBlameAssessment(MCRiskAssessment(nsims, nticks, ncol), blame_counts)
end
function assign_blame(res::MCBlameAssessment)
    best_pair = (0,0)
    max_count = 0
    for (k,v) in res.blame_counts
        if v > max_count
            best_pair = k
            max_count = v
        end
    end
    return best_pair
end

function export_risk_assessments(
    lrec::PackedListRecord,
    roadway::MobiusRoadway;
    nsims::Int = 100,
    nticks::Int = 100, # simulation horizon
    scene::MobiusScene = MobiusScene(),
    models::Dict{Int, LaneFollowingDriver} = Dict{Int, LaneFollowingDriver}(),
    rec::QueueRecord{MobiusEntity} = QueueRecord(MobiusEntity, 2, TIMESTEP),
    collision_callback::MobiusCollisionCallback = MobiusCollisionCallback(),
    )

    return [
        monte_carlo_risk_assessment(packedscene, roadway,
            nsims=nsims, nticks=nticks, scene=scene, models=models, rec=rec, collision_callback=collision_callback)
        for packedscene in ListRecordFrameIterator(lrec)]
end

println("loading risk estimations")
tic()
risk_ests = let
    filepath = joinpath(COLLISION_SCENARIO_DIR, "risks.dat")
    nfloats = div(filesize(filepath), sizeof(Float32))
    open(filepath, "r") do io
        read(io, Float32, nfloats)
    end
end
toc()

crit_scenes = AttentiveMobiusScene[]
crit_scene_local_ticks = Int[]
crit_scene_global_ticks = Int[]
crit_scene_files = String[]

println("loading crit scenarios")
tic()
t_start = now()
files = readdir(COLLISION_SCENARIO_DIR)
for (i,file) in enumerate(files)
    println("loading $i / $(length(files)), elapsed time = ", now() - t_start)
    if ismatch(r"collision_listrecord_tick_\d+_datetime_\d+_\d+_\d+.txt", file)

        tick = parse(Int, match(r"(?<=tick_)\d+", file).match)
        @show file
        @show tick
        lrec = open(joinpath(COLLISION_SCENARIO_DIR, file), "r") do io
            read(io, MIME"text/plain"(), PackedListRecord)
        end

        mc_risks = export_risk_assessments(lrec, roadway,
            nsims=NSIMS, nticks=NTICKS, scene=SCENE, models=MODELS, rec=REC, collision_callback=COLLISION_CALLBACK)

        local_crit_scene_tick = findlast(a->get_risk(a) < RISK_THRESHOLD, mc_risks) # get last index where risk is below
        local_crit_scene_tick = min(local_crit_scene_tick+1, nframes(lrec)) # the tick after that
        @show local_crit_scene_tick
        @assert local_crit_scene_tick != 0

        push!(crit_scenes, get!(allocate_frame(lrec), lrec, local_crit_scene_tick))
        push!(crit_scene_local_ticks, local_crit_scene_tick)
        push!(crit_scene_global_ticks, tick + local_crit_scene_tick - 1)
        push!(crit_scene_files, file)

        open(joinpath(COLLISION_SCENARIO_DIR, splitext(file)[1] * "_risks.txt"), "w") do io
            for (ind_local, mc_risk) in enumerate(mc_risks)
                ind_global = tick + ind_local - 1
                est_risk = risk_ests[ind_global]
                @printf(io, "%d,%d,%d,%.16e\n", mc_risk.nsims, mc_risk.nticks, mc_risk.ncol, est_risk)
            end
        end
    end
end
toc()

tic()
println("SAVING CRIT SCENE DATA")
open(joinpath(COLLISION_SCENARIO_DIR, "crit_transition_scenes.txt"), "w") do io
    write(io, MIME"text/plain"(), crit_scenes)
end
toc()

println("assigning blames!")

tic()
t_start = now()
t_prev = t_start
open(joinpath(COLLISION_SCENARIO_DIR, "crit_transition_blames.txt"), "w") do io
    global t_prev
    for (i,(packedscene,local_tick,global_tick,file)) in enumerate(zip(crit_scenes,crit_scene_local_ticks,crit_scene_global_ticks,crit_scene_files))
        t_now = now()
        println("running $i / $(length(crit_scenes)): elapsed time = ", t_now - t_start, "  delta = ", t_now - t_prev)
        t_prev = t_now
        mc_blame = monte_carlo_blame_assignment(packedscene, roadway,
            nsims=NSIMS, nticks=NTICKS, scene=SCENE, models=MODELS, rec=REC, collision_callback=COLLISION_CALLBACK)
        b = assign_blame(mc_blame)
        @show b
        @printf(io, "%s,%d,%d,%d,%d\n", file, b[1], b[2], local_tick, global_tick)
    end
end

println("[DONE]")
toc()