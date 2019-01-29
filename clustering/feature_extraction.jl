using AutomotiveDrivingModels
using AutoRisk
using DataFrames
using Records
using HDF5, JLD
using BenchmarkTools

include("../AutoMobius/AutoMobius.jl")
using AutoMobius
include("scenario_feature_extractor.jl")

REGEX = r"collision_listrecord_sim_count_\d+_datetime_\d+_\d+_\d+.txt"

type CriticalScenario
    name::String
    rec::PackedListRecord
    frame_ind::Int 
end

function load_records(data_dir::String)
    list_records = Dict{String, PackedListRecord}()
    for file in readdir(data_dir)
        if ismatch(REGEX, file)
            filepath = joinpath(data_dir, file)
            list_records[file] = open(filepath, "r") do io
                read(io, MIME"text/plain"(), PackedListRecord) 
            end
        end
    end
    return list_records
end

function load_critical_scenarios(critical_scenarios_filepath::String, 
        list_records::Dict{String, PackedListRecord})
    df = readtable(critical_scenarios_filepath)
    num_scenarios = size(df, 1)
    critical_scenarios = Array{CriticalScenario}(num_scenarios)
    for i in 1:num_scenarios
        name = basename(df[i, 1])
        rec = list_records[name]
        critical_scenarios[i] = CriticalScenario(name, rec, df[i, 2])
    end
    return critical_scenarios
end

function extract_scenario_features(critical_scenarios::Array{CriticalScenario},
        timesteps::Int = 1)
    roadway = StraightRoadway(200.0)
    scene = Scene1D()
    models = Dict{Int, LaneFollowingDriver}()
    ext = ScenarioFeatureExtractor()
    num_scenarios = length(critical_scenarios)
    features = zeros(length(ext), timesteps, num_scenarios)

    for i in 1:num_scenarios
        empty!(scene)
        empty!(models)
        scenario = critical_scenarios[i]
        start_frame = max(scenario.frame_ind - timesteps + 1, 1)
        for (t, frame_ind) in enumerate(start_frame:scenario.frame_ind)
            unpack!(scenario.rec, frame_ind, scene, models)
            pull_features!(ext, scene, roadway, models)
            features[:, t, i] = ext.features[:]
        end
    end
    return features, feature_names(ext)
end

function write_scenario_features(output_filepath::String, features::Array{Float64}, 
        feature_names::Array{String}, filenames::Array{String}, timesteps::Int = 1)
    outfile = open(output_filepath, "w")
    write(outfile, "filename," * join(feature_names, ",") * "\n")
    num_scenarios = size(features, 3)
    for i in 1:num_scenarios
        outstr = filenames[i] * ","
        for t in 1:timesteps
            outstr *= join(["$(@sprintf("%.4f", f))" for f in features[:,t,i]], ",")
            outstr *= ","
        end
        outstr *= "\n"
        write(outfile, outstr)
    end
    close(outfile)
end

function main_feature_extraction(data_dir::String = "../data/collision_scenarios",
        critical_scenarios_filepath::String = "../data/clustering/critical_frames.txt",
        output_filepath::String = "../data/clustering/critical_features.txt",
        timesteps::Int = 5)
    println("loading records...")
    list_records = load_records(data_dir)
    println("loading scenarios...")
    critical_scenarios = load_critical_scenarios(
        critical_scenarios_filepath, list_records)
    println("extracting features...")
    features, feature_names = extract_scenario_features(
        critical_scenarios, timesteps)
    filenames = String[cs.name for cs in critical_scenarios]
    println("saving to file...")
    write_scenario_features(
        output_filepath, features, feature_names, filenames, timesteps)
end