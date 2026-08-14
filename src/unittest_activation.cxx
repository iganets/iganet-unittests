/**
   @file unittests/src/unittest_activation.cxx

   @brief Activation function unittests

   @author Matthias Moller

   @copyright This file is part of the IgANet project

   This Source Code Form is subject to the terms of the Mozilla Public
   License, v. 2.0. If a copy of the MPL was not distributed with this
   file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#include <iganet/iganet.h>
#include <gtest/gtest.h>

#include <sstream>

namespace {

template <typename Activation, typename Expected>
void expect_activation(const Activation &activation, const torch::Tensor &input,
                       Expected expected) {
  EXPECT_TRUE(torch::allclose(activation.apply(input), expected(input)));
  std::ostringstream stream;
  stream << activation;
  EXPECT_FALSE(stream.str().empty());
}

} // namespace

TEST(Activation, ElementwiseWrappersAgreeWithTorch) {
  auto input = torch::tensor({-2., -0.5, 0., 0.5, 2.}, torch::kFloat64);

  expect_activation(iganet::None{}, input,
                    [](const auto &x) { return x; });
  expect_activation(iganet::GELU{}, input,
                    [](const auto &x) { return torch::gelu(x); });
  expect_activation(iganet::Hardsigmoid{}, input,
                    [](const auto &x) { return torch::hardsigmoid(x); });
  expect_activation(iganet::Hardswish{}, input,
                    [](const auto &x) { return torch::hardswish(x); });
  expect_activation(iganet::LogSigmoid{}, input,
                    [](const auto &x) { return torch::log_sigmoid(x); });
  expect_activation(iganet::Mish{}, input,
                    [](const auto &x) { return torch::mish(x); });
  expect_activation(iganet::Sigmoid{}, input,
                    [](const auto &x) { return torch::sigmoid(x); });
  expect_activation(iganet::SiLU{}, input,
                    [](const auto &x) { return torch::silu(x); });
  expect_activation(iganet::Softsign{}, input, [](const auto &x) {
    return torch::nn::functional::softsign(x);
  });
  expect_activation(iganet::Tanh{}, input,
                    [](const auto &x) { return torch::tanh(x); });
  expect_activation(iganet::Tanhshrink{}, input, [](const auto &x) {
    return torch::nn::functional::tanhshrink(x);
  });
}

TEST(Activation, ConfiguredWrappersHonorTheirOptions) {
  auto input = torch::tensor({-2., -0.5, 0., 0.5, 2.}, torch::kFloat64);

  EXPECT_TRUE(torch::allclose(iganet::CELU(2.0).apply(input),
                              torch::nn::functional::celu(
                                  input, torch::nn::functional::CELUFuncOptions()
                                             .alpha(2.0))));
  EXPECT_TRUE(torch::allclose(iganet::ELU(1.5).apply(input),
                              torch::nn::functional::elu(
                                  input, torch::nn::functional::ELUFuncOptions()
                                             .alpha(1.5))));
  EXPECT_TRUE(torch::allclose(iganet::Hardshrink(0.75).apply(input),
                              torch::nn::functional::hardshrink(
                                  input,
                                  torch::nn::functional::HardshrinkFuncOptions()
                                      .lambda(0.75))));
  EXPECT_TRUE(torch::allclose(iganet::Hardtanh(-0.25, 0.75).apply(input),
                              torch::clamp(input, -0.25, 0.75)));
  EXPECT_TRUE(torch::allclose(iganet::LeakyReLU(0.2).apply(input),
                              torch::leaky_relu(input, 0.2)));
  EXPECT_TRUE(torch::equal(iganet::ReLU{}.apply(input), torch::relu(input)));
  EXPECT_TRUE(torch::equal(iganet::ReLU6{}.apply(input),
                           torch::clamp(input, 0., 6.)));
  EXPECT_TRUE(torch::allclose(iganet::SELU{}.apply(input), torch::selu(input)));
  EXPECT_TRUE(torch::allclose(iganet::Softplus(2.0, 10.0).apply(input),
                              torch::softplus(input, 2.0, 10.0)));
  EXPECT_TRUE(torch::equal(iganet::Softshrink(0.75).apply(input),
                           torch::softshrink(input, 0.75)));
  EXPECT_TRUE(torch::equal(iganet::Threshold(0.25, -3.0).apply(input),
                           torch::threshold(input, 0.25, -3.0)));
}

TEST(Activation, DimensionChangingAndNormalizationWrappers) {
  auto matrix = torch::tensor({{1., 2., 3., 4.}, {4., 3., 2., 1.}},
                              torch::kFloat64);
  auto glu = iganet::GLU(1).apply(matrix);
  EXPECT_EQ(glu.sizes(), (torch::IntArrayRef{2, 2}));

  auto softmax = iganet::Softmax(1).apply(matrix);
  auto softmin = iganet::Softmin(1).apply(matrix);
  auto logsoftmax = iganet::LogSoftmax(1).apply(matrix);
  EXPECT_TRUE(torch::allclose(softmax.sum(1), torch::ones(2, torch::kFloat64)));
  EXPECT_TRUE(torch::allclose(softmin.sum(1), torch::ones(2, torch::kFloat64)));
  EXPECT_TRUE(torch::allclose(logsoftmax.exp(), softmax));

  auto normalized = iganet::Normalize(2.0, 1e-12, 1).apply(matrix);
  EXPECT_TRUE(torch::allclose(normalized.norm(2, 1),
                              torch::ones(2, torch::kFloat64)));
  EXPECT_TRUE(torch::allclose(iganet::LayerNorm({4}).apply(matrix),
                              torch::layer_norm(matrix, {4})));
  EXPECT_TRUE(torch::allclose(iganet::PReLU(torch::tensor({0.25}, torch::kFloat64))
                                  .apply(matrix - 2.5),
                              torch::prelu(matrix - 2.5,
                                           torch::tensor({0.25}, torch::kFloat64))));
}

TEST(Activation, BatchAndGroupNormalizationExposeState) {
  auto input = torch::tensor({{{1., 2.}, {3., 4.}},
                              {{2., 4.}, {6., 8.}}}, torch::kFloat64);
  auto mean = torch::zeros(2, torch::kFloat64);
  auto variance = torch::ones(2, torch::kFloat64);
  iganet::BatchNorm batch_norm(mean, variance);
  EXPECT_EQ(batch_norm.apply(input).sizes(), input.sizes());
  EXPECT_TRUE(torch::equal(batch_norm.running_mean(), mean));

  iganet::GroupNorm group_norm(1);
  EXPECT_EQ(group_norm.apply(input).sizes(), input.sizes());
  iganet::InstanceNorm instance_norm;
  EXPECT_EQ(instance_norm.apply(input).sizes(), input.sizes());
  iganet::LocalResponseNorm local_response_norm(2);
  EXPECT_EQ(local_response_norm.apply(input).sizes(), input.sizes());
}
