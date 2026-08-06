/**
   @file unittests/unittest_blocktensor.cxx

   @brief Compile-time block-tensor unittests

   @author Matthias Moller

   @copyright This file is part of the IgANet project

   This Source Code Form is subject to the terms of the Mozilla Public
   License, v. 2.0. If a copy of the MPL was not distributed with this
   file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#include <iganet.h>
#include <iostream>

#include <gtest/gtest.h>

TEST(BlockTensor, BlockTensor) {
  iganet::utils::BlockTensor<torch::Tensor, 2, 1> A(torch::ones({5, 5}),
                                                    2 * torch::ones({5, 5}));
  iganet::utils::BlockTensor<torch::Tensor, 2, 1> B(3 * torch::ones({5, 5}),
                                                    4 * torch::ones({5, 5}));

  EXPECT_EQ(A + B, (iganet::utils::BlockTensor<torch::Tensor, 2, 1>(
                       4 * torch::ones({5, 5}), 6 * torch::ones({5, 5}))));

  EXPECT_EQ(A - B, (iganet::utils::BlockTensor<torch::Tensor, 2, 1>(
                       -2 * torch::ones({5, 5}), -2 * torch::ones({5, 5}))));
  EXPECT_EQ(A.tr(), (iganet::utils::BlockTensor<torch::Tensor, 1, 2>(
                        torch::ones({5, 5}), 2 * torch::ones({5, 5}))));

  EXPECT_EQ(B.tr(), (iganet::utils::BlockTensor<torch::Tensor, 1, 2>(
                        3 * torch::ones({5, 5}), 4 * torch::ones({5, 5}))));

  EXPECT_EQ(A * B.tr(), (iganet::utils::BlockTensor<torch::Tensor, 2, 2>(
                            3 * torch::ones({5, 5}), 4 * torch::ones({5, 5}),
                            6 * torch::ones({5, 5}), 8 * torch::ones({5, 5}))));

  EXPECT_EQ(A.tr() * B, (iganet::utils::BlockTensor<torch::Tensor, 1, 1>(
                            11 * torch::ones({5, 5}))));

  iganet::utils::BlockTensor<torch::Tensor, 1, 1> C(5 * torch::ones({5, 5}));

  EXPECT_EQ(C.inv(), (iganet::utils::BlockTensor<torch::Tensor, 1, 1>(
                         0.2 * torch::ones({5, 5}))));

  EXPECT_EQ(C.invtr(), C.inv().tr());

  iganet::utils::BlockTensor<torch::Tensor, 2, 2> D(
      torch::ones({5, 5}), 2 * torch::ones({5, 5}), 3 * torch::ones({5, 5}),
      4 * torch::ones({5, 5}));

  EXPECT_EQ(D.inv(),
            (iganet::utils::BlockTensor<torch::Tensor, 2, 2>(
                -2.0 * torch::ones({5, 5}), 1.5 * torch::ones({5, 5}),
                1.0 * torch::ones({5, 5}), -0.5 * torch::ones({5, 5}))));

  EXPECT_EQ(D.invtr(), D.inv().tr());

  iganet::utils::BlockTensor<torch::Tensor, 3, 3> E(
      torch::ones({5, 5}), 2 * torch::ones({5, 5}), 3 * torch::ones({5, 5}),
      torch::zeros({5, 5}), torch::ones({5, 5}), 4 * torch::ones({5, 5}),
      5 * torch::ones({5, 5}), 6 * torch::ones({5, 5}), torch::zeros({5, 5}));

  EXPECT_EQ(E.inv(), (iganet::utils::BlockTensor<torch::Tensor, 3, 3>(
                         -24 * torch::ones({5, 5}), 18 * torch::ones({5, 5}),
                         5 * torch::ones({5, 5}), 20 * torch::ones({5, 5}),
                         -15 * torch::ones({5, 5}), -4 * torch::ones({5, 5}),
                         -5 * torch::ones({5, 5}), 4 * torch::ones({5, 5}),
                         torch::ones({5, 5}))));

  EXPECT_EQ(E.invtr(), E.inv().tr());

  iganet::utils::BlockTensor<torch::Tensor, 4, 4> F(
      5 * torch::ones({5, 5}), 6 * torch::ones({5, 5}), 6 * torch::ones({5, 5}),
      8 * torch::ones({5, 5}), 2 * torch::ones({5, 5}), 2 * torch::ones({5, 5}),
      2 * torch::ones({5, 5}), 8 * torch::ones({5, 5}), 6 * torch::ones({5, 5}),
      6 * torch::ones({5, 5}), 2 * torch::ones({5, 5}), 8 * torch::ones({5, 5}),
      2 * torch::ones({5, 5}), 3 * torch::ones({5, 5}), 6 * torch::ones({5, 5}),
      7 * torch::ones({5, 5}));

  EXPECT_EQ(F.inv(),
            (iganet::utils::BlockTensor<torch::Tensor, 4, 4>(
                -17.00 * torch::ones({5, 5}), -9.00 * torch::ones({5, 5}),
                12.00 * torch::ones({5, 5}), 16.00 * torch::ones({5, 5}),
                17.00 * torch::ones({5, 5}), 8.75 * torch::ones({5, 5}),
                -11.75 * torch::ones({5, 5}), -16.00 * torch::ones({5, 5}),
                -4.00 * torch::ones({5, 5}), -2.25 * torch::ones({5, 5}),
                2.75 * torch::ones({5, 5}), 4.00 * torch::ones({5, 5}),
                torch::ones({5, 5}), 0.75 * torch::ones({5, 5}),
                -0.75 * torch::ones({5, 5}), -1.00 * torch::ones({5, 5}))));

  EXPECT_EQ(F.invtr(), F.inv().tr());
}

#define test_unary_op(op, A, ...)                                              \
  EXPECT_TRUE(torch::equal(iganet::utils::op((A), ##__VA_ARGS__)(0, 0),        \
                           torch::op((A)(0, 0), ##__VA_ARGS__)))

#define test_unary_special_op(op, A, ...)                                      \
  EXPECT_TRUE(torch::equal(iganet::utils::op((A), ##__VA_ARGS__)(0, 0),        \
                           torch::special::op((A)(0, 0), ##__VA_ARGS__)))

#define test_binary_op(op, A, B, ...)                                          \
  EXPECT_TRUE(torch::equal(iganet::utils::op((A), (B), ##__VA_ARGS__)(0, 0),   \
                           torch::op((A)(0, 0), (B)(0, 0), ##__VA_ARGS__)))

#define test_binary_special_op(op, A, B, ...)                                  \
  EXPECT_TRUE(                                                                 \
      torch::equal(iganet::utils::op((A), (B), ##__VA_ARGS__)(0, 0),           \
                   torch::special::op((A)(0, 0), (B)(0, 0), ##__VA_ARGS__)))

#define test_ternary_op(op, A, B, C, ...)                                      \
  EXPECT_TRUE(                                                                 \
      torch::equal(iganet::utils::op((A), (B), (C), ##__VA_ARGS__)(0, 0),      \
                   torch::op((A)(0, 0), (B)(0, 0), (C)(0, 0), ##__VA_ARGS__)))

#define test_ternary_special_op(op, A, B, C, ...)                              \
  EXPECT_TRUE(torch::equal(                                                    \
      iganet::utils::op((A), (B), (C), ##__VA_ARGS__)(0, 0),                   \
      torch::special::op((A)(0, 0), (B)(0, 0), (C)(0, 0), ##__VA_ARGS__)))

TEST(BlockTensor, BlockTensor_ops) {

  iganet::utils::BlockTensor<torch::Tensor, 1, 1> A(-torch::rand({5, 5})),
      B(torch::rand({5, 5})), C(torch::rand({5, 5})),
      I(torch::rand({5, 5}).to(at::kInt)), J(torch::rand({5, 5}).to(at::kInt)),
      K(torch::rand({5, 5}).to(at::kBool)),
      L(torch::rand({5, 5}).to(at::kBool)),
      X(torch::rand({5, 5}).to(at::kComplexDouble));

  test_unary_op(abs, A);
  test_unary_op(absolute, A);
  test_unary_op(acos, A);
  test_unary_op(arccos, A);
  test_unary_op(acosh, B + 1);
  test_unary_op(acosh, 1 + B);
  test_unary_op(arccosh, B + 1);
  test_unary_op(arccosh, 1 + B);
  test_binary_op(add, A, B, 0.5);
  test_ternary_op(addcdiv, A, B, C, 0.5);
  test_ternary_op(addcmul, A, B, C, 0.5);
  test_unary_op(angle, A);
  test_unary_op(asin, A);
  test_unary_op(arcsin, A);
  test_unary_op(atan, A);
  test_unary_op(arctan, A);
  test_binary_op(atan2, A, B);
  test_binary_op(arctan2, A, B);
  test_unary_op(bitwise_not, K);
  test_binary_op(bitwise_and, K, L);
  test_binary_op(bitwise_or, K, L);
  test_binary_op(bitwise_xor, K, L);
  test_binary_op(bitwise_left_shift, I, J);
  test_binary_op(bitwise_right_shift, I, J);
  test_unary_op(ceil, A);
  test_unary_op(clamp, A, 0.0, 0.5);
  test_unary_op(clip, A, 0.0, 0.5);
  test_unary_op(conj_physical, A);
  test_binary_op(copysign, A, B);
  test_unary_op(cos, A);
  test_unary_op(cosh, A);
  test_unary_op(deg2rad, A);
  test_binary_op(div, A, B);
  test_binary_op(divide, A, B);
  test_unary_op(erf, A);
  test_unary_op(erfc, A);
  test_unary_op(erfinv, A);
  test_unary_op(exp, A);
  test_unary_op(exp2, A);
  test_unary_op(expm1, A);
  test_unary_op(trunc, A);
  test_binary_op(float_power, A, I);
  test_unary_op(floor, A);
  test_binary_op(fmod, A, B);
  test_unary_op(frac, A);
  test_unary_op(imag, X);
  test_binary_op(ldexp, A, I);
  test_unary_op(lgamma, A);
  test_unary_op(log, B);
  test_unary_op(log10, B);
  test_unary_op(log1p, B);
  test_unary_op(log2, B);
  test_binary_op(logaddexp, A, B);
  test_binary_op(logaddexp2, A, B);
  test_binary_op(logical_and, K, L);
  test_unary_op(logical_not, K);
  test_binary_op(logical_or, K, L);
  test_binary_op(logical_xor, K, L);
  test_binary_op(hypot, A, A);
  test_unary_op(i0, A);
  test_binary_special_op(gammainc, B, B);
  test_binary_op(igamma, B, B);
  test_binary_special_op(gammaincc, B, B);
  test_binary_op(igammac, B, B);
  test_binary_op(mul, A, B);
  test_binary_op(multiply, A, B);
  test_unary_op(neg, A);
  test_unary_op(negative, A);
  test_binary_op(nextafter, A, B);
  test_unary_op(positive, A);
  test_binary_op(pow, A, I);
  test_unary_op(rad2deg, A);
  test_unary_op(real, X);
  test_unary_op(reciprocal, A);
  test_binary_op(remainder, A, B);
  test_unary_op(round, A);
  test_unary_op(rsqrt, B);
  test_unary_special_op(expit, A);
  test_unary_op(sigmoid, A);
  test_unary_op(sign, A);
  test_unary_op(sgn, X);
  test_unary_op(signbit, A);
  test_unary_op(sin, A);
  test_unary_op(sinc, A);
  test_unary_op(sinh, A);
  test_unary_op(sqrt, B);
  test_unary_op(square, A);
  test_binary_op(sub, A, B, 0.5);
  test_binary_op(subtract, A, B, 0.5);
  test_unary_op(tan, A);
  test_unary_op(tanh, A);
  test_unary_op(trunc, A);
  test_binary_op(xlogy, A, B);
}

TEST(BlockTensor, MetadataStorageAndFormatting) {
  using BT = iganet::utils::BlockTensor<torch::Tensor, 2, 2>;
  static_assert(BT::dims() == std::array<std::size_t, 2>{2, 2});
  static_assert(BT::dim<0>() == 2);
  static_assert(BT::dim<2>() == 0);
  static_assert(BT::size() == 2);
  static_assert(BT::entries() == 4);

  auto shared = std::make_shared<torch::Tensor>(torch::tensor(1.));
  auto *shared_raw = shared.get();
  BT tensor(shared, torch::tensor(2.), torch::tensor(3.), torch::tensor(4.));
  EXPECT_EQ(tensor[0].get(), shared_raw);
  EXPECT_DOUBLE_EQ(tensor(0, 1).item<double>(), 2.0);
  tensor.set(1, 0, torch::tensor(8.));
  tensor.set(3, torch::tensor(9.));
  EXPECT_DOUBLE_EQ(tensor(1, 0).item<double>(), 8.0);
  EXPECT_DOUBLE_EQ(tensor(1, 1).item<double>(), 9.0);

  std::ostringstream stream;
  stream << tensor;
  EXPECT_NE(stream.str().find("[1,1]"), std::string::npos);
}

TEST(BlockTensor, DeterminantTraceNormNormalizeAndDot) {
  using BT = iganet::utils::BlockTensor<torch::Tensor, 2, 2>;
  BT matrix(torch::tensor(1.), torch::tensor(2.), torch::tensor(3.),
            torch::tensor(4.));
  EXPECT_DOUBLE_EQ(matrix.det().item<double>(), -2.0);
  EXPECT_DOUBLE_EQ(matrix.trace()(0, 0).item<double>(), 5.0);
  EXPECT_NEAR(matrix.norm()(0, 0).item<double>(), std::sqrt(30.0), 1e-6);
  auto normalized = matrix.normalize();
  EXPECT_NEAR(normalized.norm()(0, 0).item<double>(), 1.0, 1e-6);
  EXPECT_DOUBLE_EQ(matrix.dot(matrix)(0, 0).item<double>(), 30.0);
  EXPECT_DOUBLE_EQ(iganet::utils::dot(matrix, matrix)(0, 0).item<double>(),
                   30.0);
}

TEST(BlockTensor, GeneralizedInverseAndCompoundScalarOperators) {
  iganet::utils::BlockTensor<torch::Tensor, 2, 1> column(torch::tensor(3.),
                                                        torch::tensor(4.));
  auto inverse = column.ginv();
  EXPECT_NEAR(inverse(0, 0).item<double>(), 0.12, 1e-6);
  EXPECT_NEAR(inverse(0, 1).item<double>(), 0.16, 1e-6);
  EXPECT_EQ(column.ginvtr(), inverse.tr());

  auto original = column;
  column += original;
  column -= 1.0;
  auto scaled = column * torch::tensor(2.0);
  EXPECT_DOUBLE_EQ(scaled(0, 0).item<double>(), 10.0);
  EXPECT_DOUBLE_EQ(scaled(1, 0).item<double>(), 14.0);
  EXPECT_NE(column, original);
}

TEST(BlockTensor, NumericScalarMultiplicationOnBothSides) {
  iganet::utils::BlockTensor<torch::Tensor, 1, 2> tensor(
      torch::tensor({1., 2.}), torch::tensor({3., 4.}));
  auto right = tensor * 2.5;
  auto left = 2 * tensor;
  EXPECT_TRUE(torch::equal(right(0, 0), torch::tensor({2.5, 5.0})));
  EXPECT_TRUE(torch::equal(right(0, 1), torch::tensor({7.5, 10.0})));
  EXPECT_TRUE(torch::equal(left(0, 0), torch::tensor({2., 4.})));
  EXPECT_TRUE(torch::equal(left(0, 1), torch::tensor({6., 8.})));
}

TEST(BlockTensor, RankThreeIndexSliceReorderAndMultiplication) {
  using BT3 = iganet::utils::BlockTensor<torch::Tensor, 2, 2, 2>;
  BT3 tensor(torch::tensor(0.), torch::tensor(1.), torch::tensor(2.),
             torch::tensor(3.), torch::tensor(4.), torch::tensor(5.),
             torch::tensor(6.), torch::tensor(7.));
  EXPECT_EQ(BT3::rows(), 2);
  EXPECT_EQ(BT3::cols(), 2);
  EXPECT_EQ(BT3::slices(), 2);
  EXPECT_DOUBLE_EQ(tensor(1, 0, 1).item<double>(), 6.0);
  tensor.set(1, 0, 1, torch::tensor(9.));
  EXPECT_DOUBLE_EQ(tensor.slice(1)(1, 0).item<double>(), 9.0);
  EXPECT_DOUBLE_EQ(tensor.reorder_ikj()(1, 1, 0).item<double>(),
                   tensor(1, 0, 1).item<double>());
  EXPECT_DOUBLE_EQ(tensor.reorder_jik()(0, 1, 1).item<double>(),
                   tensor(1, 0, 1).item<double>());
  EXPECT_DOUBLE_EQ(tensor.reorder_kji()(1, 0, 1).item<double>(),
                   tensor(1, 0, 1).item<double>());
  EXPECT_DOUBLE_EQ(tensor.reorder_kij()(1, 1, 0).item<double>(),
                   tensor(1, 0, 1).item<double>());

  iganet::utils::BlockTensor<torch::Tensor, 1, 2> left(torch::tensor(1.),
                                                       torch::tensor(2.));
  auto left_product = left * tensor;
  EXPECT_DOUBLE_EQ(left_product(0, 0, 0).item<double>(), 4.0);

  iganet::utils::BlockTensor<torch::Tensor, 2, 1> right(torch::tensor(1.),
                                                        torch::tensor(2.));
  auto right_product = tensor * right;
  EXPECT_DOUBLE_EQ(right_product(0, 0, 0).item<double>(), 2.0);
  EXPECT_DOUBLE_EQ(right_product(1, 0, 0).item<double>(), 8.0);
  EXPECT_DOUBLE_EQ(right_product(0, 0, 1).item<double>(), 14.0);
  EXPECT_DOUBLE_EQ(right_product(1, 0, 1).item<double>(), 23.0);
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  iganet::init();
  int result = RUN_ALL_TESTS();
  iganet::finalize();
  return result;
}
