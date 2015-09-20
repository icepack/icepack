
with import <nixpkgs> {}; {
  icepackEnv = stdenv.mkDerivation {
    name = "icepack";
    buildInputs = [ stdenv cmake gdal gcc ];
  };
}
