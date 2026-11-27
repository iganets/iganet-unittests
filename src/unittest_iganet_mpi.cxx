/**
   @file unittests/src/unittest_iganet_mpi.cxx

   @brief Data-parallel IgANet training unittests

   The tests train one small function-fitting network on all
   collocation points in a single process (reference) and a second
   network on blocks of the points that are distributed over the MPI
   processes. Both networks must end with the same loss and parameters,
   and all processes must hold the same parameters. Without MPI the
   distributed network owns all points and the tests compare two
   identical serial runs.

   @author Matthias Moller

   @copyright This file is part of the IgANet project

   This Source Code Form is subject to the terms of the Mozilla Public
   License, v. 2.0. If a copy of the MPL was not distributed with this
   file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#include <functional>
#include <iomanip>
#include <iganet/iganet.h>
#include <iostream>

#include <gtest/gtest.h>
#include <unittest_config.hpp>

namespace {

using real_t = double;

using inputs_t = std::tuple<iganet::S<iganet::UniformBSpline<real_t, 2, 1, 1>>>;
using outputs_t =
    std::tuple<iganet::S<iganet::UniformBSpline<real_t, 1, 2, 2>>>;

constexpr int64_t ncoeffs = 8;
constexpr int64_t npoints = 576;
constexpr int64_t nsamples = 16;
constexpr int64_t globalBatch = 4;
constexpr double tolerance = 1e-10;

/// @brief Number of processes
int nranks() { return iganet::mpi::size(); }

/// @brief Fitting of sin(pi x) sin(pi y) at the given collocation points
template <typename Optimizer>
class Fitting : public iganet::IgANet<Optimizer, inputs_t, outputs_t> {
private:
  using Base = iganet::IgANet<Optimizer, inputs_t, outputs_t>;

  /// @brief Collocation points owned by this network
  std::array<torch::Tensor, 2> xi_;

  /// @brief Function called at the start of every epoch
  std::function<void(int64_t, Fitting &)> hook_;

public:
  using Base::Base;

  void points(const std::array<torch::Tensor, 2> &xi) { xi_ = xi; }

  void hook(std::function<void(int64_t, Fitting &)> hook) { hook_ = hook; }

  bool epoch(int64_t epoch) override {
    if (hook_)
      hook_(epoch, *this);
    if (epoch == 0) {
      Base::inputs(epoch);
      return true;
    }
    return false;
  }

  torch::Tensor loss(const torch::Tensor &outputs, int64_t epoch) override {
    auto target = sin(M_PI * xi_[0]) * sin(M_PI * xi_[1]);

    if (outputs.dim() > 1) {
      // Batch of samples: one column of coefficients per sample
      Base::outputs(outputs.t());
      return torch::mse_loss(*Base::template output<0>().eval(xi_)[0],
                             target.expand({outputs.size(0), -1}).t());
    } else {
      Base::outputs(outputs.flatten());
      return torch::mse_loss(*Base::template output<0>().eval(xi_)[0], target);
    }
  }
};

/// @brief All collocation points of the problem, identical on all processes
std::array<torch::Tensor, 2> allPoints() {
  static const std::array<torch::Tensor, 2> xi = [] {
    torch::manual_seed(4711);
    auto opts = torch::TensorOptions().dtype(torch::kFloat64);
    return std::array<torch::Tensor, 2>{torch::rand({npoints}, opts),
                                        torch::rand({npoints}, opts)};
  }();
  return xi;
}

template <typename Optimizer>
void configure(Fitting<Optimizer> &net);

template <> void configure(Fitting<torch::optim::Adam> &net) {
  net.optimizerReset(torch::optim::AdamOptions(1e-2));
}

template <> void configure(Fitting<torch::optim::SGD> &net) {
  net.optimizerReset(torch::optim::SGDOptions(5e-2));
}

template <> void configure(Fitting<torch::optim::LBFGS> &net) {
  net.optimizerReset(torch::optim::LBFGSOptions()
                         .line_search_fn("strong_wolfe")
                         .max_iter(5)
                         .history_size(10)
                         .tolerance_grad(1e-14)
                         .tolerance_change(1e-16));
}

template <typename Optimizer>
std::unique_ptr<Fitting<Optimizer>> make(int64_t epochs, bool distributed,
                                         int seed) {
  torch::manual_seed(seed);
  using namespace iganet::literals;
  std::vector<std::vector<std::any>> activations{{iganet::activation::tanh},
                                                 {iganet::activation::none}};
  auto net = std::make_unique<Fitting<Optimizer>>(
      std::vector<int64_t>{12}, activations,
      std::tuple{iganet::utils::to_array(2_i64, 2_i64)},
      std::tuple{iganet::utils::to_array(ncoeffs, ncoeffs)});
  configure(*net);
  net->options()
      .max_epoch(epochs)
      .min_loss(0.0)
      .min_loss_change(0.0)
      .min_loss_rel_change(0.0)
      .mpi_data_parallel(distributed)
      .mpi_check_interval(distributed ? 1 : 0);
  net->points(distributed ? iganet::mpi::share(allPoints()) : allPoints());
  return net;
}

template <typename Net> std::vector<double> flatParameters(const Net &net) {
  std::vector<double> result;
  for (const auto &p : net.parameters()) {
    auto flat = p.detach().flatten().contiguous();
    result.insert(result.end(), flat.template data_ptr<double>(),
                  flat.template data_ptr<double>() + flat.numel());
  }
  return result;
}

/// @brief Largest difference between the values on all processes
double rankSpread(const std::vector<double> &v) {
  double spread = 0;
#ifdef IGANET_WITH_MPI
  std::vector<double> lo(v.size()), hi(v.size());
  MPI_Allreduce(v.data(), lo.data(), v.size(), MPI_DOUBLE, MPI_MIN,
                MPI_COMM_WORLD);
  MPI_Allreduce(v.data(), hi.data(), v.size(), MPI_DOUBLE, MPI_MAX,
                MPI_COMM_WORLD);
  for (std::size_t i = 0; i < v.size(); ++i)
    spread = std::max(spread, hi[i] - lo[i]);
#endif
  return spread;
}

/// @brief Largest difference relative to the largest reference entry
double relError(const std::vector<double> &a, const std::vector<double> &b) {
  double diff = 0, ref = 0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    diff = std::max(diff, std::abs(a[i] - b[i]));
    ref = std::max(ref, std::abs(b[i]));
  }
  return diff / ref;
}

/// @brief Loss over all points evaluated without the library reduction:
/// mean over the local block, averaged over the processes
template <typename Net>
double globalLoss(Net &net, const torch::Tensor &in, bool distributed) {
  torch::NoGradGuard guard;
  double local = net.loss(net.net()->forward(in), 0).template item<double>();
  double global = local;
#ifdef IGANET_WITH_MPI
  if (distributed) {
    local /= nranks();
    MPI_Allreduce(&local, &global, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  }
#endif
  return global;
}

template <typename Optimizer>
void compareWithSerial(int64_t epochs, int seedSerial, int seedRank,
                       const char *name) {
  auto serial = make<Optimizer>(epochs, false, seedSerial);
  serial->train();
  auto ps = flatParameters(*serial);
  double ls = globalLoss(*serial, serial->inputs(0), false);

  auto dist = make<Optimizer>(epochs, true, seedRank);
  dist->train();
  auto pd = flatParameters(*dist);
  double ld = globalLoss(*dist, dist->inputs(0), true);

  double perr = relError(pd, ps);
  double lerr = std::abs(ld - ls) / std::abs(ls);
  double spread = rankSpread(pd);

  if (iganet::mpi::rank() == 0)
    std::cout << "[mpi-test] " << name << " ranks=" << nranks()
              << " serial_loss=" << std::setprecision(17) << ls
              << " dist_loss=" << ld << " loss_rel_err=" << std::setprecision(3)
              << lerr << " param_rel_err=" << perr
              << " rank_param_spread=" << spread << std::endl;

  EXPECT_LE(perr, tolerance);
  EXPECT_LE(lerr, tolerance);
  EXPECT_EQ(spread, 0.0);
}

} // namespace

TEST(IgANetMPI, Adam_matches_serial) {
  compareWithSerial<torch::optim::Adam>(10, 1234, 1234, "Adam");
}

TEST(IgANetMPI, SGD_matches_serial) {
  compareWithSerial<torch::optim::SGD>(10, 1234, 1234, "SGD");
}

TEST(IgANetMPI, LBFGS_matches_serial) {
  compareWithSerial<torch::optim::LBFGS>(4, 1234, 1234, "LBFGS");
}

TEST(IgANetMPI, Initial_parameters_are_taken_from_rank_zero) {
  // Rank r seeds its initialization with 1234 + r; the serial reference
  // uses the seed of rank 0.
  compareWithSerial<torch::optim::Adam>(5, 1234, 1234 + iganet::mpi::rank(),
                                        "Adam_rank_seeds");
}

/// @brief Samples of the network input: each rank holds the rows of
/// every global batch that belong to it
TEST(IgANetMPI, DataLoader_matches_serial) {
  auto run = []<typename Optimizer>(const char *name, int64_t epochs) {
    auto serial = make<Optimizer>(epochs, false, 1234);
    auto dist = make<Optimizer>(epochs, true, 1234);
    // Every process uses all collocation points; the samples are split
    serial->points(allPoints());
    dist->points(allPoints());
    auto in0 = serial->inputs(0);

    torch::manual_seed(99);
    auto samples =
        in0.unsqueeze(0) +
        0.1 * torch::randn({nsamples, in0.size(0)},
                           torch::TensorOptions().dtype(torch::kFloat64));

    const int64_t b = globalBatch / nranks();
    std::vector<int64_t> idx;
    for (int64_t k = 0; k < nsamples / globalBatch; ++k)
      for (int64_t j = 0; j < b; ++j)
        idx.push_back(k * globalBatch + iganet::mpi::rank() * b + j);
    auto local = samples.index_select(
        0, torch::tensor(idx, torch::TensorOptions().dtype(torch::kInt64)));

    auto makeLoader = [](const torch::Tensor &data, int64_t batch) {
      return torch::data::make_data_loader<
          torch::data::samplers::SequentialSampler>(
          torch::data::datasets::TensorDataset(data).map(
              torch::data::transforms::Stack<torch::data::TensorExample>()),
          torch::data::DataLoaderOptions().batch_size(batch).workers(0));
    };

    auto loaderSerial = makeLoader(samples, globalBatch);
    serial->train(*loaderSerial);
    auto ps = flatParameters(*serial);

    auto loaderDist = makeLoader(local, b);
    dist->train(*loaderDist);
    auto pd = flatParameters(*dist);

    double perr = relError(pd, ps);
    double spread = rankSpread(pd);
    if (iganet::mpi::rank() == 0)
      std::cout << "[mpi-test] DataLoader " << name << " ranks=" << nranks()
                << " param_rel_err=" << std::setprecision(3) << perr
                << " rank_param_spread=" << spread << std::endl;
    EXPECT_LE(perr, tolerance);
    EXPECT_EQ(spread, 0.0);
  };

  run.template operator()<torch::optim::Adam>("Adam", 3);
  run.template operator()<torch::optim::LBFGS>("LBFGS", 2);
}

TEST(IgANetMPI, Consistency_check_detects_diverged_parameters) {
  auto net = make<torch::optim::Adam>(6, true, 1234);
  net->options().mpi_check_interval(1);
  // The last rank perturbs one parameter in epoch 2
  net->hook([](int64_t epoch, Fitting<torch::optim::Adam> &n) {
    if (epoch == 2 && iganet::mpi::rank() == iganet::mpi::size() - 1) {
      torch::NoGradGuard guard;
      n.parameters()[0].view(-1)[0] += 1e-3;
    }
  });

  if (nranks() > 1) {
    try {
      net->train();
      ADD_FAILURE() << "the consistency check did not detect the perturbation";
    } catch (const std::runtime_error &e) {
      EXPECT_NE(std::string(e.what()).find("different network parameters"),
                std::string::npos)
          << e.what();
      if (iganet::mpi::rank() == 0)
        std::cout << "[mpi-test] guard: " << e.what() << std::endl;
    }
  } else
    EXPECT_NO_THROW(net->train());
}

namespace {

/// @brief Network of the deprecated v1 interface whose loss is the mean
/// square of the output coefficients
template <typename Optimizer, typename GeometryMap, typename Variable>
class SquareV1 : public iganet::v1::IgANet<Optimizer, GeometryMap, Variable> {
public:
  using iganet::v1::IgANet<Optimizer, GeometryMap, Variable>::IgANet;

  bool epoch(int64_t epoch) override { return epoch == 0; }

  torch::Tensor loss(const torch::Tensor &outputs, int64_t epoch) override {
    return torch::mean(outputs.square());
  }
};

} // namespace

TEST(IgANetMPI, V1_train_matches_serial) {
  using namespace iganet::literals;
  using Optimizer = torch::optim::Adam;
  using Space = iganet::S<iganet::UniformBSpline<real_t, 1, 5>>;

  auto make = [](bool distributed) {
    torch::manual_seed(77);
    auto net = std::make_unique<SquareV1<Optimizer, Space, Space>>(
        std::vector<int64_t>{20, 10},
        std::vector<std::vector<std::any>>{{iganet::activation::tanh},
                                           {iganet::activation::sigmoid},
                                           {iganet::activation::none}},
        std::tuple(iganet::utils::to_array(6_i64)));
    net->options()
        .max_epoch(8)
        .min_loss(0.0)
        .min_loss_change(0.0)
        .min_loss_rel_change(0.0)
        .mpi_data_parallel(distributed)
        .mpi_check_interval(1);
    return net;
  };

  auto serial = make(false);
  serial->train();
  auto dist = make(true);
  dist->train();

  auto ps = flatParameters(*serial);
  auto pd = flatParameters(*dist);
  EXPECT_LE(relError(pd, ps), tolerance);
  EXPECT_EQ(rankSpread(pd), 0.0);
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  iganet::init();
  int result = RUN_ALL_TESTS();
  iganet::finalize();
  return result;
}
