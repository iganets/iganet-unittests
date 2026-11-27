/**
   @file unittests/src/unittest_serialize.cxx

   @brief Serialization utility unittests

   @author Matthias Moller

   @copyright This file is part of the IgANet project

   This Source Code Form is subject to the terms of the Mozilla Public
   License, v. 2.0. If a copy of the MPL was not distributed with this
   file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#include <iganet/iganet.h>
#include <xml_charconv.hpp>
#include <gtest/gtest.h>
#include <algorithm>
#include <iterator>
#include <sstream>
#include <vector>

namespace {
class SerializableValue final : public iganet::utils::Serializable {
public:
  nlohmann::json to_json() const override { return {{"value", 7}}; }
  void pretty_print(std::ostream &os) const override { os << "value=7"; }
};

template <std::size_t N>
void expectGenericXmlRoundTrip(const std::vector<int64_t> &sizes) {
  auto source = torch::arange(1, torch::prod(torch::tensor(sizes)).item<int64_t>() + 1,
                              torch::kFloat64)
                    .reshape(sizes);
  auto doc = iganet::utils::to_xml<double, N>(source, "Tensor", 11, "rank");
  auto restored = torch::zeros({0}, torch::kFloat64);
  iganet::utils::from_xml<double, N>(doc, restored, "Tensor", 11, "rank");
  EXPECT_TRUE(torch::equal(restored, source));
}
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
  // The text holds the four values, in any notation that reads back exactly.
  std::istringstream text(node.text().get());
  std::vector<double> values{std::istream_iterator<double>(text),
                             std::istream_iterator<double>()};
  std::sort(values.begin(), values.end());
  EXPECT_EQ(values, (std::vector<double>{1., 2., 3., 4.}));
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

TEST(Serialize, DocumentReturningOverloadsForwardMetadata) {
  auto tensor = torch::tensor({{1., 2.}, {3., 4.}}, torch::kFloat64);
  auto tensor_doc =
      iganet::utils::to_xml<double, 2>(tensor, "Matrix", 4, "tensor");
  auto tensor_node = tensor_doc.child("xml").child("Matrix");
  ASSERT_TRUE(tensor_node);
  EXPECT_EQ(tensor_node.attribute("id").as_int(), 4);
  EXPECT_STREQ(tensor_node.attribute("label").value(), "tensor");

  auto accessor = tensor.accessor<double, 2>();
  auto accessor_doc = iganet::utils::to_xml(
      accessor, tensor.sizes(), "Matrix", 5, "accessor");
  EXPECT_EQ(accessor_doc.child("xml").child("Matrix").attribute("id").as_int(),
            5);

  iganet::utils::TensorArray<2> tensors{tensor, 2 * tensor};
  auto array_doc =
      iganet::utils::to_xml<double, 2>(tensors, "Matrix", 6, "array");
  auto nodes = array_doc.child("xml").children("Matrix");
  auto iterator = nodes.begin();
  ASSERT_NE(iterator, nodes.end());
  EXPECT_EQ(iterator->attribute("index").as_int(), 0);
  ++iterator;
  ASSERT_NE(iterator, nodes.end());
  EXPECT_EQ(iterator->attribute("index").as_int(), 1);
}

TEST(Serialize, RoundTripsGenericXmlRanksThreeThroughSix) {
  expectGenericXmlRoundTrip<3>({1, 2, 2});
  expectGenericXmlRoundTrip<4>({1, 1, 2, 2});
  expectGenericXmlRoundTrip<5>({1, 1, 1, 2, 2});
  expectGenericXmlRoundTrip<6>({1, 1, 1, 1, 2, 2});
}

TEST(Serialize, RoundTripsOneDimensionalMatrix) {
  auto source = torch::tensor({1., 2., 3.}, torch::kFloat64);
  auto doc = iganet::utils::to_xml<double, 1>(source);
  auto restored = torch::zeros({0}, torch::kFloat64);
  iganet::utils::from_xml<double, 1>(doc, restored);
  EXPECT_TRUE(torch::equal(restored, source));
}

TEST(SerializeProperties, RoundTripPreservesValuesShapeDtypeAndMetadata) {
  for (const auto dtype : {torch::kFloat32, torch::kFloat64}) {
    auto source = torch::arange(24, torch::TensorOptions{}.dtype(dtype))
                      .reshape({2, 3, 4});
    auto doc = dtype == torch::kFloat32
                   ? iganet::utils::to_xml<float, 3>(source, "Tensor", 37,
                                                      "property", 9)
                   : iganet::utils::to_xml<double, 3>(source, "Tensor", 37,
                                                       "property", 9);

    auto node = doc.child("xml").child("Tensor");
    ASSERT_TRUE(node);
    EXPECT_EQ(node.attribute("id").as_int(), 37);
    EXPECT_EQ(node.attribute("index").as_int(), 9);
    EXPECT_STREQ(node.attribute("label").value(), "property");

    auto restored = torch::zeros({0}, torch::TensorOptions{}.dtype(dtype));
    if (dtype == torch::kFloat32)
      iganet::utils::from_xml<float, 3>(doc, restored, "Tensor", 37,
                                        "property", true, 9);
    else
      iganet::utils::from_xml<double, 3>(doc, restored, "Tensor", 37,
                                         "property", true, 9);

    EXPECT_TRUE(torch::equal(restored, source));
    EXPECT_EQ(restored.sizes(), source.sizes());
    EXPECT_EQ(restored.scalar_type(), source.scalar_type());
  }
}

namespace {
template <std::size_t N>
torch::Tensor extremeTensor() {
  const auto values = iganet::unittests::xml::extreme_values<double>();
  auto tensor = torch::from_blob(const_cast<double *>(values.data()),
                                 {static_cast<int64_t>(values.size())},
                                 torch::kFloat64)
                    .clone();
  return N == 1 ? tensor : tensor.reshape({1, static_cast<int64_t>(values.size())});
}
} // namespace

TEST(Serialize, XmlRoundTripKeepsEveryDigit) {
  for (const std::string tag : {"Matrix", "Tensor"}) {
    SCOPED_TRACE(tag);

    auto source1 = extremeTensor<1>();
    auto doc1 = iganet::utils::to_xml<double, 1>(source1, tag, 3, "values");
    auto restored1 = torch::zeros({0}, torch::kFloat64);
    iganet::utils::from_xml<double, 1>(doc1, restored1, tag, 3, "values");
    EXPECT_TRUE(torch::equal(restored1, source1));
    EXPECT_TRUE(iganet::unittests::xml::bit_equal<double>(restored1, source1));

    auto source2 = extremeTensor<2>();
    auto doc2 = iganet::utils::to_xml<double, 2>(source2, tag, 3, "values");
    auto restored2 = torch::zeros({0, 0}, torch::kFloat64);
    iganet::utils::from_xml<double, 2>(doc2, restored2, tag, 3, "values");
    EXPECT_TRUE(torch::equal(restored2, source2));
    EXPECT_TRUE(iganet::unittests::xml::bit_equal<double>(restored2, source2));
  }
}

TEST(Serialize, XmlImportAcceptsWhitespaceForms) {
  const auto values = iganet::unittests::xml::extreme_values<double>();
  auto source = extremeTensor<1>();

  for (const auto &[name, separators] :
       iganet::unittests::xml::whitespace_forms()) {
    SCOPED_TRACE(name);

    auto matrix = iganet::utils::to_xml<double, 1>(source, "Matrix", 3, "v");
    matrix.child("xml").child("Matrix").text().set(
        iganet::unittests::xml::join(values, separators, "\n ", "\t\n")
            .c_str());
    auto restored = torch::zeros({0}, torch::kFloat64);
    iganet::utils::from_xml<double, 1>(matrix, restored, "Matrix", 3, "v");
    EXPECT_TRUE(iganet::unittests::xml::bit_equal<double>(restored, source));

    auto tensor = iganet::utils::to_xml<double, 1>(source, "Tensor", 3, "v");
    tensor.child("xml").child("Tensor").child("Dimensions").text().set(
        "\t 7 \r\n");
    tensor.child("xml").child("Tensor").child("Data").text().set(
        iganet::unittests::xml::join(values, separators, "  ", "\n").c_str());
    auto restored_tensor = torch::zeros({0}, torch::kFloat64);
    iganet::utils::from_xml<double, 1>(tensor, restored_tensor, "Tensor", 3,
                                       "v");
    EXPECT_TRUE(
        iganet::unittests::xml::bit_equal<double>(restored_tensor, source));
  }
}

TEST(Serialize, XmlImportRejectsMalformedNumbers) {
  auto source = torch::tensor({1., 2., 3.}, torch::kFloat64);

  const auto import = [&](const std::string &tag, const std::string &child,
                          const std::string &text) {
    auto doc = iganet::utils::to_xml<double, 1>(source, tag, 3, "v");
    auto node = doc.child("xml").child(tag.c_str());
    (child.empty() ? node : node.child(child.c_str())).text().set(text.c_str());
    auto restored = torch::zeros({0}, torch::kFloat64);
    iganet::utils::from_xml<double, 1>(doc, restored, tag, 3, "v");
  };

  for (const std::string bad : {"1.5abc", "abc", "1,5", "1e"}) {
    SCOPED_TRACE(bad);
    const auto matrix = iganet::unittests::xml::runtime_error_message(
        [&] { import("Matrix", "", "1 " + bad + " 3"); });
    EXPECT_NE(matrix.find("offset 2"), std::string::npos) << matrix;
    EXPECT_NE(matrix.find("\"" + bad + "\""), std::string::npos) << matrix;

    EXPECT_FALSE(iganet::unittests::xml::runtime_error_message(
                     [&] { import("Tensor", "Data", "1 " + bad + " 3"); })
                     .empty());
    EXPECT_FALSE(iganet::unittests::xml::runtime_error_message(
                     [&] { import("Tensor", "Dimensions", bad); })
                     .empty());
  }

  EXPECT_EQ(iganet::unittests::xml::runtime_error_message(
                [&] { import("Matrix", "", ""); }),
            "XML object does not provide enough coefficients");
  EXPECT_EQ(iganet::unittests::xml::runtime_error_message(
                [&] { import("Tensor", "Data", " \n"); }),
            "XML object does not provide enough coefficients");
  EXPECT_EQ(iganet::unittests::xml::runtime_error_message(
                [&] { import("Matrix", "", "1 2 3 4"); }),
            "XML object provides too many coefficients");
}

TEST(Serialize, XmlExportAndImportIgnoreTheGlobalLocale) {
  iganet::unittests::xml::DecimalCommaLocale german;
  if (!german.active())
    GTEST_SKIP() << "locale de_DE.UTF-8 with a decimal comma is not installed";

  ASSERT_EQ(std::to_string(1.5), "1,500000");

  auto source = extremeTensor<1>();
  for (const std::string tag : {"Matrix", "Tensor"}) {
    auto doc = iganet::utils::to_xml<double, 1>(source, tag, 3, "v");
    const auto node = doc.child("xml").child(tag.c_str());
    const std::string text =
        (tag == "Matrix" ? node : node.child("Data")).text().get();
    EXPECT_EQ(text.find(','), std::string::npos) << text;
    auto restored = torch::zeros({0}, torch::kFloat64);
    iganet::utils::from_xml<double, 1>(doc, restored, tag, 3, "v");
    EXPECT_TRUE(iganet::unittests::xml::bit_equal<double>(restored, source));
  }
}
