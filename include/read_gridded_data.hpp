
#include <deal.II/base/function_lib.h>

#include <string>

// using namespace dealii;

typedef dealii::Functions::InterpolatedTensorProductGridData<2> GridData;

GridData readQgis(const std::string& filename);
GridData readGeoDat(const std::string& filename);
