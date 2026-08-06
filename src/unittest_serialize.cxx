/**
   @file unittests/unittest_serialize.cxx

   @brief Serialization utility unittests

   @author Matthias Moller

   @copyright This file is part of the IgANet project

   This Source Code Form is subject to the terms of the Mozilla Public
   License, v. 2.0. If a copy of the MPL was not distributed with this
   file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#include <iganet.h>
#include <gtest/gtest.h>
#include <sstream>

namespace {
class SerializableValue final : public iganet::utils::Serializable {
public:
  nlohmann::json to_json() const override { return {{"value", 7}}; }
  void pretty_print(std::ostream &os) const override { os << "value=7"; }
};
}

TEST(Serialize, TensorAndTensorArrayToJson) {
  auto tensor = torch::tensor({{1., 2.}, {3., 4.}}, torch::kFloat64);
  auto json = iganet::utils::to_json<double, 2>(tensor);
  EXPECT_EQ(json, nlohmann::json({1., 2., 3., 4.}));

  iganet::utils::TensorArray<2> tensors{tensor, 2 * tensor};
  auto array_json = iganet::utils::to_json<double, 2>(tensors);
  ASSERT_EQ(array_json.size(), 2);
  EXPECT_EQ(array_json[1], nlohmann::json({2., 4., 6., 8.}));
}

TEST(Serialize, SerializableInterfaceDispatchesVirtually) {
  std::unique_ptr<iganet::utils::Serializable> value =
      std::make_unique<SerializableValue>();
  EXPECT_EQ(value->to_json(), nlohmann::json({{"value", 7}}));
  std::ostringstream stream;
  value->pretty_print(stream);
  EXPECT_EQ(stream.str(), "value=7");
}

TEST(Serialize, MatrixXmlContainsMetadataAndValues) {
  auto tensor = torch::tensor({{1., 2.}, {3., 4.}}, torch::kFloat64);
  auto accessor = tensor.accessor<double, 2>();
  pugi::xml_document doc;
  auto root = doc.append_child("xml");
  iganet::utils::to_xml(accessor, tensor.sizes(), root, "Matrix", 7, "test");
  auto node = doc.child("xml").child("Matrix");
  ASSERT_TRUE(node);
  EXPECT_EQ(node.attribute("id").as_int(), 7);
  EXPECT_STREQ(node.attribute("label").value(), "test");
  EXPECT_EQ(node.attribute("rows").as_int(), 2);
  EXPECT_EQ(node.attribute("cols").as_int(), 2);
  EXPECT_NE(std::string(node.text().get()).find("4.000000"), std::string::npos);
}

TEST(Serialize, TensorArrayXmlAssignsIndices) {
  auto tensors = iganet::utils::to_tensorArray({1., 2.}, {3., 4.});
  pugi::xml_document doc;
  auto root = doc.append_child("xml");
  iganet::utils::to_xml<double, 1>(tensors, root, "Matrix", 2, "items");
  auto range = doc.child("xml").children("Matrix");
  auto iterator = range.begin();
  ASSERT_NE(iterator, range.end());
  EXPECT_EQ(iterator->attribute("index").as_int(), 0);
  ++iterator;
  ASSERT_NE(iterator, range.end());
  EXPECT_EQ(iterator->attribute("index").as_int(), 1);
}

TEST(Serialize, JsonSupportsOneThreeAndFourDimensions) {
  EXPECT_EQ((iganet::utils::to_json<double, 1>(
                torch::tensor({1., 2.}, torch::kFloat64))),
            nlohmann::json({1., 2.}));
  EXPECT_EQ((iganet::utils::to_json<double, 3>(
                torch::arange(4., torch::kFloat64).reshape({1, 2, 2}))),
            nlohmann::json({0., 1., 2., 3.}));
  EXPECT_EQ((iganet::utils::to_json<double, 4>(
                torch::arange(4., torch::kFloat64).reshape({1, 1, 2, 2}))),
            nlohmann::json({0., 1., 2., 3.}));
}

TEST(Serialize, ReadsMatrixXmlAndValidatesAllocation) {
  pugi::xml_document doc;
  auto root = doc.append_child("xml");
  auto node = root.append_child("Matrix");
  node.append_attribute("id") = 3;
  node.append_attribute("rows") = 2;
  node.append_attribute("cols") = 2;
  node.text().set("1 2 3 4");

  auto tensor = torch::zeros({1, 1}, torch::kFloat64);
  iganet::utils::from_xml<double, 2>(root, tensor, "Matrix", 3);
  EXPECT_TRUE(torch::equal(tensor, torch::tensor({{1., 2.}, {3., 4.}})));

  auto wrong_shape = torch::zeros({1, 1}, torch::kFloat64);
  EXPECT_THROW((iganet::utils::from_xml<double, 2>(
                   root, wrong_shape, "Matrix", 3, "", false)),
               std::runtime_error);
  EXPECT_THROW((iganet::utils::from_xml<double, 2>(
                   root, tensor, "Matrix", 99)),
               std::runtime_error);
}

TEST(Serialize, ReadsGenericTensorAndRejectsMalformedData) {
  pugi::xml_document doc;
  auto root = doc.append_child("xml");
  auto node = root.append_child("Tensor");
  node.append_attribute("id") = 1;
  node.append_child("Dimensions").text().set("2 2");
  node.append_child("Data").text().set("1 2 3 4");
  auto tensor = torch::zeros({0}, torch::kFloat64);
  iganet::utils::from_xml<double, 2>(root, tensor, "Tensor", 1);
  EXPECT_EQ(tensor.sizes(), (torch::IntArrayRef{2, 2}));
  EXPECT_DOUBLE_EQ(tensor[1][1].item<double>(), 4.0);

  node.child("Data").text().set("1 2 3");
  EXPECT_THROW((iganet::utils::from_xml<double, 2>(root, tensor, "Tensor", 1)),
               std::runtime_error);
}
