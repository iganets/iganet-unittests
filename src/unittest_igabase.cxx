/**
   @file unittests/src/unittest_igabase.cxx

   @brief Isogeometric dataset base unittests

   @author Matthias Moller

   @copyright This file is part of the IgANet project

   This Source Code Form is subject to the terms of the Mozilla Public
   License, v. 2.0. If a copy of the MPL was not distributed with this
   file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#include <iganet.h>
#include <gtest/gtest.h>

namespace {

class TensorObject {
public:
  explicit TensorObject(torch::Tensor value) : value_(std::move(value)) {}

  template <typename Function> TensorObject &transform(Function &&function) {
    value_ = function(value_);
    return *this;
  }

  torch::Tensor as_tensor() const { return value_; }

  TensorObject &from_xml(const pugi::xml_node &, int = 0,
                         std::string = "") {
    return *this;
  }

private:
  torch::Tensor value_;
};

} // namespace

TEST(IgADataset, EmptyDatasetsRejectElementAccess) {
  iganet::IgADataset<false> inputs;
  EXPECT_EQ(*inputs.size(), 1);
  EXPECT_THROW((void)inputs.get(0), std::runtime_error);

  iganet::IgADataset<true> supervised;
  EXPECT_EQ(*supervised.size(), 1);
  EXPECT_THROW((void)supervised.get(0), std::runtime_error);
}

TEST(IgADataset, ReferenceOnlyDatasetReturnsTransformedSamples) {
  iganet::IgADataset<false> dataset;
  TensorObject first(torch::tensor({1., 2.}));
  TensorObject second(torch::tensor({3., 4.}));
  dataset.add_referenceData(first,
                            [](const auto &value) { return value * 2.; });
  dataset.add_referenceData(second,
                            [](const auto &value) { return value + 1.; });
  EXPECT_EQ(*dataset.size(), 2);
  EXPECT_TRUE(torch::equal(dataset.get(0).data, torch::tensor({2., 4.})));
  EXPECT_TRUE(torch::equal(dataset.get(1).data, torch::tensor({4., 5.})));
}

TEST(IgADataset, SupervisedDatasetReturnsDataAndTargets) {
  iganet::IgADataset<true> dataset;
  TensorObject data(torch::tensor({1., 2.}));
  TensorObject target(torch::tensor({5.}));
  dataset.add_referenceData(data, [](const auto &value) { return value; });
  pugi::xml_document document;
  document.append_child("xml");
  dataset.add_solution(target, document);
  auto example = dataset.get(0);
  EXPECT_TRUE(torch::equal(example.data, torch::tensor({1., 2.})));
  EXPECT_TRUE(torch::equal(example.target, torch::tensor({5.})));
}

TEST(IgADataset, MissingFilesystemInputIsRejected) {
  iganet::IgADataset<false> dataset;
  iganet::UniformBSpline<double, 1, 1> spline({2});
  EXPECT_THROW(dataset.add_referenceData(
                   spline,
                   std::string("/path/that/does/not/exist/iganet.xml")),
               std::runtime_error);
}
