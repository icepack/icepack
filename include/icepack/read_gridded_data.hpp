
/**
 * Procedures for reading gridded data sets in a variety of formats and
 * building deal.II Function objects out of them
 */

#ifndef READ_GRIDDED_DATA_HPP
#define READ_GRIDDED_DATA_HPP

#include <deal.II/base/function_lib.h>

#include <string>


namespace icepack
{

  class GridData :
    public dealii::Functions::InterpolatedTensorProductGridData<2>
  {
  private:
    double xrange[2];
    double yrange[2];

  public:
    GridData(const std::array<std::vector<double>, 2>& coodinate_values,
             const dealii::Table<2, double>&           data_values);

    double xmin() const { return xrange[0]; }
    double xmax() const { return xrange[1]; }
    double ymin() const { return yrange[0]; }
    double ymax() const { return yrange[1]; }
  };


  GridData readArcAsciiGrid(const std::string& filename);
  GridData readGeoDat(const std::string& filename);
  GridData readGeoTIFF(const std::string& filename);

}

#endif
