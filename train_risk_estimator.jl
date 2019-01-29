using Mocha

# const DATA_DIR = "/media/tim/DATAPART1/PublicationData/2017_adas_validation/"
const DATA_DIR = "/media/tim/Tim 1500 GB/2017_adas_validation/"

data_layer = AsyncHDF5DataLayer(name="train-data", source=joinpath(DATA_DIR, "train_risk_estimator.txt"), tops=[:X,:y], batch_size=64, shuffle=true) # -> (14,12,1,N)
# data_layer = AsyncHDF5DataLayer(name="train-data", source=joinpath(DATA_DIR, "train_risk_estimator_full.txt"), tops=[:X,:y], batch_size=64, shuffle=true) # -> (14,12,1,N)
conv1_layer = ConvolutionLayer(name="conv1", n_filter=16, kernel=(14,1), bottoms=[:X], tops=[:conv1], neuron=Neurons.ReLU()) # -> (1,12,16,N)
conv2_layer = ConvolutionLayer(name="conv2", n_filter=16, kernel=(1,1), bottoms=[:conv1], tops=[:conv2], neuron=Neurons.ReLU()) # -> (1,12,16,N)
ip1_layer = InnerProductLayer(name="ip1", output_dim=64, neuron=Neurons.ReLU(), bottoms=[:conv2], tops=[:ip1]) # -> (64,N)
ip2_layer = InnerProductLayer(name="ip2", output_dim=32, neuron=Neurons.ReLU(), bottoms=[:ip1], tops=[:ip2]) # -> (32,N)
ip3_layer = InnerProductLayer(name="ip3", output_dim=1, bottoms=[:ip2], neuron=Neurons.Sigmoid(), tops=[:ip3]) # -> (1,N) ∈ [0,1]
loss_layer = SquareLossLayer(name="loss", bottoms=[:ip3,:y])

backend = DefaultBackend()
init(backend)

common_layers = [conv1_layer, conv2_layer, ip1_layer, ip2_layer, ip3_layer]
net = Net("risk-est-train", backend, [data_layer, common_layers..., loss_layer])

exp_dir = joinpath(DATA_DIR, "mocha_snapshots")

method = SGD()
params = make_solver_parameters(method,
            max_iter=100000,
            regu_coef=0.0005,
            mom_policy=MomPolicy.Fixed(0.9),
            lr_policy=LRPolicy.Inv(0.01, 0.0001, 0.75),
            load_from=exp_dir,
         )
solver = Solver(method, params)

setup_coffee_lounge(solver, save_into="$exp_dir/risk_estimator_stats.jld", every_n_iter=1000)

# report training progress every 100 iterations
add_coffee_break(solver, TrainingSummary(), every_n_iter=100)

# save snapshots every 5000 iterations
add_coffee_break(solver, Snapshot(exp_dir), every_n_iter=5000)

# show performance on test data every 1000 iterations
data_layer_test = AsyncHDF5DataLayer(name="test-data", source=joinpath(DATA_DIR, "test_risk_estimator.txt"), batch_size=64, tops=[:X, :y])
test_net = Net("risk-est-test", backend, [data_layer_test, common_layers..., loss_layer])
add_coffee_break(solver, ValidationPerformance(test_net), every_n_iter=1000)

solve(solver, net)

destroy(net)
destroy(test_net)
shutdown(backend)

println("DONE")