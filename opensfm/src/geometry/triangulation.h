#pragma once

#include <Eigen/LU>
#include <Eigen/QR>
#include <Eigen/SVD>
#include <Eigen/StdVector>
#include <fstream>
#include <iostream>
#include <string>
#include <foundation/types.h>

namespace geometry {

enum {
  TRIANGULATION_OK = 0,
  TRIANGULATION_SMALL_ANGLE,
  TRIANGULATION_BEHIND_CAMERA,
  TRIANGULATION_BAD_REPROJECTION
};

double AngleBetweenVectors(const Eigen::Vector3d &u, const Eigen::Vector3d &v);

py::list TriangulateReturn(int error, py::object value);

Eigen::Vector4d TriangulateBearingsDLTSolve(
    const Eigen::Matrix<double, 3, -1> &bs,
    const std::vector< Eigen::Matrix<double, 3, 4> > &Rts);

py::object TriangulateBearingsDLT(const py::list &Rts_list,
                                  const py::list &bs_list, double threshold,
                                  double min_angle);

// Point minimizing the squared distance to all rays
// Closed for solution from
//   Srikumar Ramalingam, Suresh K. Lodha and Peter Sturm
//   "A generic structure-from-motion framework"
//   CVIU 2006
template< class T >
Eigen::Matrix<T, 3, 1> TriangulateBearingsMidpointSolve(
    const Eigen::Matrix<T, 3, -1> &os,
    const Eigen::Matrix<T, 3, -1> &bs){
  int nviews = bs.cols();
  assert(nviews == os.cols());
  assert(nviews >= 2);

  Eigen::Matrix<T, 3, 3> BBt;
  Eigen::Matrix<T, 3, 1> BBtA, A;
  BBt.setZero();
  BBtA.setZero();
  A.setZero();
  for (int i = 0; i < nviews; ++i) {
    BBt += bs.col(i) * bs.col(i).transpose();
    BBtA += bs.col(i) * bs.col(i).transpose() * os.col(i);
    A += os.col(i);
  }
  Eigen::Matrix<T, 3, 3> Cinv = (T(nviews) * Eigen::Matrix<T, 3, 3>::Identity() - BBt).inverse();

  return (Eigen::Matrix<T, 3, 3>::Identity() + BBt * Cinv) * A / T(nviews) - Cinv * BBtA;
}

py::object TriangulateBearingsMidpoint(const py::list &os_list,
                                       const py::list &bs_list,
                                       const py::list &threshold_list,
                                       double min_angle);

}  // namespace geometry
