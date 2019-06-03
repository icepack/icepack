import sys
import geojson
import pygmsh


def add_loop_to_geometry(geometry, multi_line_string):
    line_loop = []
    for line_index, line_string in enumerate(multi_line_string):
        arc = []
        for index in range(len(line_string) - 1):
            x1 = line_string[index]
            x2 = line_string[index + 1]
            arc.append(geometry.add_line(x1, x2))

        num_lines = len(multi_line_string)
        next_line_string = multi_line_string[(line_index + 1) % num_lines]
        x1 = line_string[-1]
        x2 = next_line_string[0]
        arc.append(geometry.add_line(x1, x2))

        geometry.add_physical(arc)
        line_loop.extend(arc)

    return geometry.add_line_loop(line_loop)


def collection_to_geo(collection, lcar=10e3):
    geometry = pygmsh.built_in.Geometry()

    features = collection['features']
    num_features = len(features)

    points = [[[geometry.add_point((point[0], point[1], 0.), lcar=lcar)
                for point in line_string[:-1]]
               for line_string in feature['geometry']['coordinates']]
              for feature in features]

    line_loops = [add_loop_to_geometry(geometry, multi_line_string)
                  for multi_line_string in points]
    plane_surface = geometry.add_plane_surface(line_loops[0],
                                               holes=line_loops[1:])
    physical_surface = geometry.add_physical(plane_surface)

    return geometry


def main(input_filename, output_filename):
    with open(input_filename, 'r') as input_file:
        collection = geojson.load(input_file)

    outline = collection_to_geo(collection, lcar=10e3)
    with open(output_filename, 'w') as output_file:
        output_file.write(outline.get_code())

if __name__ == '__main__':
    input_filename = sys.argv[1]
    output_filename = sys.argv[2]
    main(input_filename, output_filename)
