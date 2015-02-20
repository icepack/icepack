
#include "read_gridded_data.hpp"

#include <iostream>


int main ()
{
  GridData helheim_b = readQgis("helheim_2006_2013_composite_bottom.txt");

  return 0;
}
