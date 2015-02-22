
/**
 * Procedures for reading gridded data sets in a variety of formats and
 * building deal.II Function objects out of them
 */

#ifndef READ_GRIDDED_DATA_HPP
#define READ_GRIDDED_DATA_HPP

#include <deal.II/base/function_lib.h>

#include <string>

// using namespace dealii;

typedef dealii::Functions::InterpolatedTensorProductGridData<2> GridData;

GridData readQgis(const std::string& filename);
GridData readGeoDat(const std::string& filename);

#endif
