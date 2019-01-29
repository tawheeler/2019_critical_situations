using AutomotiveDrivingModels
using AutoViz
using Records
using HDF5, JLD

@everywhere include("AutoMobius/AutoMobius.jl")
using AutoMobius

prior = 0.1
nsimulations = 100
nticks = 100
roadway = Wraparound(Straight1DRoadway(200.0))

data_dir = "data/collision_scenarios"
regex = r"collision_listrecord_sim_count_\d+_datetime_\d+_\d+_\d+.txt"
files = [file for file in readdir(data_dir) if ismatch(regex, file)]

@parallel (+) for i in 1:length(files)
    file = files[i]

    println("file #", i)
    println("file name ", file)

    filepath = joinpath(data_dir, file)
    list_record = open(filepath, "r") do io
        read(io, MIME"text/plain"(), PackedListRecord)
    end

    tic()
    betas_arr = [get_collision_estimation(list_record, j, roadway, prior, nsimulations=nsimulations, nticks=nticks) for j in 1  : nframes(list_record)-1]
    toc()

    filepath = joinpath(data_dir, "collision_estimation_"*file[22:end])
    open(filepath, "w") do io
        export_betas_array(io, betas_arr)
    end
    1
end

println("done")