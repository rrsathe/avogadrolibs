#ifndef AVOGADRO_CALC_PARALLELCALC_H
#define AVOGADRO_CALC_PARALLELCALC_H

#include "avogadrocalcexport.h"
#include <Eigen/Core>
#include <thread>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace Avogadro::Calc {

/**
 * @brief Helper class for parallel energy and gradient calculations
 */
class AVOGADROCALC_EXPORT ParallelCalc
{
public:
  using TVector = Eigen::VectorXd;

  /**
   * @brief Compute finite differences gradient in parallel
   * @param valueFunc Function that computes energy for a given position vector
   * @param x Initial position vector
   * @param grad Output gradient vector
   * @param accuracy Accuracy level (0-3)
   * @param nThreads Number of threads to use (0 = auto)
   */
  template <typename F>
  static void finiteGradientParallel(F valueFunc, const TVector& x,
                                     TVector& grad, int accuracy, int nThreads)
  {
    if (nThreads <= 0)
      nThreads = std::thread::hardware_concurrency();

    grad.resize(x.rows());
    TVector xx = x; // Make a local copy to avoid const_cast

    const int innerSteps = 2 * (accuracy + 1);
    const double ddVal = s_dd[accuracy] * s_eps;

#ifdef _OPENMP
#pragma omp parallel for num_threads(nThreads) schedule(dynamic)
#endif
    for (Eigen::Index d = 0; d < x.rows(); d++) {
      grad[d] = 0;
      for (int s = 0; s < innerSteps; ++s) {
        double tmp = xx[d];
        xx[d] += s_coeff2[accuracy][s] * s_eps;
        grad[d] += s_coeff[accuracy][s] * valueFunc(xx);
        xx[d] = tmp;
      }
      grad[d] /= ddVal;
    }
  }
  /**
   * @brief Clean gradient values and apply mask in parallel
   * @param grad Gradient vector to clean
   * @param mask Mask vector to apply
   * @param nThreads Number of threads to use (0 = auto)
   */
  static void cleanGradientsParallel(TVector& grad, const TVector& mask,
                                     int nThreads = 0);

private:
  // Coefficients for finite difference methods of different accuracy
  static const std::array<std::vector<double>, 4> s_coeff;
  static const std::array<std::vector<double>, 4> s_coeff2;
  static const std::array<double, 4> s_dd;
  static constexpr double s_eps = 2.2204e-6;
};

} // namespace Avogadro::Calc

#endif // AVOGADRO_CALC_PARALLELCALC_H
