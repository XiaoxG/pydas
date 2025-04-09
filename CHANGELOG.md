# Changelog

## [1.1.0] - 2023-03-17

### Added

- Added optimized plotting function `plot_channel_optimized` for large datasets
- Implemented WebGL rendering for improved performance
- Added adaptive downsampling for large datasets
- Added zoom control buttons for better user experience
- Added performance testing scripts

### Optimized

- Improved data loading performance for large files
- Enhanced derivative calculation with Numba acceleration
- Optimized statistical calculations with parallel processing
- Reduced memory usage with chunked processing
- Improved HTML template for better visualization

### Fixed

- Fixed memory leak in data loading for large files
- Fixed incorrect scaling in derivative calculations
- Fixed unit conversion issues
- Fixed issues with `plot_channel_plotly` function when handling multiple channels
- Fixed performance issues with `updateST` method when processing large datasets

## [1.0.0] - 2023-02-15

### Added

- Initial release of PyDAS
- Basic data loading and processing functionality
- Statistical analysis features
- Plotting capabilities

### Optimized

- Basic performance optimizations for medium-sized datasets