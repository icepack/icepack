
#ifndef ICEPACK_FACE_ITER_HPP
#define ICEPACK_FACE_ITER_HPP

namespace icepack {

  template <typename iterator>
  bool at_boundary(
    const iterator& it,
    const unsigned int face_number,
    const unsigned int id
  )
  {
    return it->face(face_number)->at_boundary()
      and it->face(face_number)->boundary_id() == id;
  }

}

#endif
