/**
   @file benchmarks/xml_roundtrip.cxx

   @brief XML export and import benchmark of a B-spline volume

   @author Matthias Moller

   @copyright This file is part of the IgANet project

   This Source Code Form is subject to the terms of the Mozilla Public
   License, v. 2.0. If a copy of the MPL was not distributed with this
   file, You can obtain one at http://mozilla.org/MPL/2.0/.

   Builds a 3D B-spline volume with geoDim 3 and N^3 coefficients (default
   N=70, about 1e6 coefficients, 3e6 values), writes it to an XML string and
   reads it back. Reports the median over the repeats of the export time
   (`to_xml` plus serialisation to a string), the import time (XML parse plus
   `from_xml`), the `from_xml` time alone, the size of the XML text and whether
   the round trip reproduces every coefficient bit for bit.

   Usage: xml_roundtrip [N] [repeats] [label]
*/

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <iganet/iganet.h>
#include <unittest_config.hpp>

namespace {

using clock_type = std::chrono::steady_clock;

double milliseconds(clock_type::time_point begin, clock_type::time_point end) {
  return std::chrono::duration<double, std::milli>(end - begin).count();
}

double median(std::vector<double> values) {
  std::sort(values.begin(), values.end());
  return values[values.size() / 2];
}

} // namespace

int main(int argc, char **argv) {
  const int64_t n = argc > 1 ? std::atoll(argv[1]) : 70;
  const int repeats = argc > 2 ? std::atoi(argv[2]) : 5;
  const std::string label = argc > 3 ? argv[3] : "";

  iganet::init();
  int status = 0;
  {
    using real_t = iganet::unittests::real_t;
    iganet::Options<real_t> options;

    torch::manual_seed(42);
    iganet::UniformBSpline<real_t, 3, 2, 2, 2> bspline_out(
        {n, n, n}, iganet::init::random, options);

    std::string text;
    std::vector<double> export_ms, import_ms, from_xml_ms;
    bool exact = true;

    // The first pass is a warm-up and is not timed.
    for (int r = -1; r < repeats; ++r) {
      auto t0 = clock_type::now();
      pugi::xml_document doc_out = bspline_out.to_xml();
      std::ostringstream os;
      doc_out.save(os);
      text = os.str();
      auto t1 = clock_type::now();

      pugi::xml_document doc_in;
      pugi::xml_parse_result result = doc_in.load_string(text.c_str());
      auto t2 = clock_type::now();
      if (!result) {
        std::cerr << "XML parse error: " << result.description() << "\n";
        status = 1;
        break;
      }

      iganet::UniformBSpline<real_t, 3, 2, 2, 2> bspline_in(options);
      bspline_in.from_xml(doc_in);
      auto t3 = clock_type::now();

      exact = true;
      for (iganet::short_t g = 0; g < 3; ++g)
        exact = exact && torch::equal(bspline_in.coeffs(g), bspline_out.coeffs(g));

      if (r >= 0) {
        export_ms.push_back(milliseconds(t0, t1));
        import_ms.push_back(milliseconds(t1, t3));
        from_xml_ms.push_back(milliseconds(t2, t3));
      }
    }

    if (status == 0) {
      std::cout << "label\tcoefficients\tvalues\trepeats\texport_ms\timport_ms"
                   "\tfrom_xml_ms\tbytes\texact\n"
                << label << "\t" << n * n * n << "\t" << 3 * n * n * n << "\t"
                << repeats << "\t" << median(export_ms) << "\t"
                << median(import_ms) << "\t" << median(from_xml_ms) << "\t"
                << text.size() << "\t" << (exact ? "yes" : "no") << "\n";
    }
  }
  iganet::finalize();
  return status;
}
