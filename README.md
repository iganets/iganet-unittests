# IGAnets-UnitTests

[![GitlabSync](https://github.com/iganets/iganet-unittests/actions/workflows/gitlab-sync.yml/badge.svg)](https://github.com/iganets/iganet-unittests/actions/workflows/gitlab-sync.yml)
[![CI](https://github.com/iganets/iganet-unittests/actions/workflows/ci-push-pr.yml/badge.svg)](https://github.com/iganets/iganet-unittests/actions/workflows/ci-push-pr.yml)
[![Documentation](https://img.shields.io/badge/docs-mkdocs-blue.svg)](https://iganets.github.io/iganet/)

[![GitHub Releases](https://img.shields.io/github/release/iganets/iganet-unittests.svg)](https://github.com/iganets/iganet-unittests/releases)
[![GitHub Downloads](https://img.shields.io/github/downloads/iganets/iganet-unittests/total)](https://github.com/iganets/iganet-unittests/releases)
[![GitHub Issues](https://img.shields.io/github/issues/iganets/iganet-unittests.svg)](https://github.com/iganets/iganet-unittests/issues)

This repository contains unit tests for [IGAnets](https://github.com/iganets/iganet), a novel approach to combine the concept of deep operator learning with the mathematical framework of isogeometric analysis.

## Usage instructions

This repository can be used in two modes:

1. As standalone unit tests by running CMake on _this_ repository without flags

2. As optional module in [iganets::core](https://github.com/iganets/iganet) by running CMake on the [iganets::core](https://github.com/iganets/iganet) repository with the flag
   ```
   -DIGANET_OPTIONAL="unittests"
   ```

In both cases, the tests can be run via
```shell
make test
```
