
#include "read_gridded_data.hpp"


int main(int argc, char **argv)
{

  GridData q = readGeoTiff(argv[1]);

  return 0;
}
