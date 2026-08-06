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

TEST(Serialize, TensorAndTensorArrayToJson) {
  auto tensor = torch::tensor({{1., 2.}, {3., 4.}}, torch::kFloat64);
  auto json = iganet::utils::to_json<double, 2>(tensor);
  EXPECT_EQ(json, nlohmann::json({1., 2., 3., 4.}));

  iganet::utils::TensorArray<2> tensors{tensor, 2 * tensor};
  auto array_json = iganet::utils::to_json<double, 2>(tensors);
  ASSERT_EQ(array_json.size(), 2);
  EXPECT_EQ(array_json[1], nlohmann::json({2., 4., 6., 8.}));
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
