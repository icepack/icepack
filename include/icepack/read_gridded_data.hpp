
#ifndef READ_GRIDDED_DATA_HPP
#define READ_GRIDDED_DATA_HPP

#include <deal.II/base/function_lib.h>

#include <string>


namespace icepack {
  using dealii::Functions::InterpolatedTensorProductGridData;

  /**
   * Class for reading gridded data into a `dealii::Function` object
   */
  class GridData : public InterpolatedTensorProductGridData<2>
  {
  public:
    /**
     * Construct a GridData object from arrays giving the locations of the
     * measurements and the data.
     * You will probably never need to use this method; instead, use the
     * functions defined below to read various common data formats into a
     * GridData object.
     */
    GridData(
      const std::array<std::vector<double>, 2>& coodinate_values,
      const dealii::Table<2, double>& data_values,
      const double missing
    );

    /**
     * Horizontal extent of the gridded data
     */
    const std::array<double, 2> xrange;

    /**
     * Vertical extent of the gridded data
     */
    const std::array<double, 2> yrange;

    /**
     * Value to indicate missing data
     */
    const double missing;
  };


  /**
   * Read a gridded data set stored in the ESRI Arc/Info ASCII grid format. See
   *   http://en.wikipedia.org/wiki/Esri_grid
   * for format specification and more info.
   */
  GridData readArcAsciiGrid(const std::string& filename);


  /**
   * Read gridded data from a georeferenced .tif image
   */
  GridData readGeoTIFF(const std::string& filename);

}

#endif
