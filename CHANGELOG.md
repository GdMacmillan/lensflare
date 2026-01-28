# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2025-01-28

### Changed
- Migrated to TensorFlow 2 API (from TF1 compatibility mode)
- Updated to modern `pyproject.toml` packaging with hatchling build backend
- Requires Python 3.12+

### Added
- Support for TensorFlow Metal on Apple Silicon (optional dependency)
- Development dependencies group (`dev` extras)
- Jupyter notebook example updated to TF2 API

### Fixed
- Keras build warning in BinaryClassifierNN

## [0.0.1] - 2018-09-17

### Added
- Initial release
- Hand-rolled deep learning models for educational purposes
- BinaryClassifierNN implementation
- Utility functions for visualization and model training
- Based on Coursera deeplearning.ai course materials

[Unreleased]: https://github.com/GdMacmillan/lensflare/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/GdMacmillan/lensflare/compare/v0.0.1...v0.1.0
[0.0.1]: https://github.com/GdMacmillan/lensflare/releases/tag/v0.0.1
