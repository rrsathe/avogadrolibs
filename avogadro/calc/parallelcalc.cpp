#include "parallelcalc.h"
#include <algorithm>
#include <thread>

namespace Avogadro::Calc {

// Initialize static coefficient arrays
const std::array<std::vector<double>, 4> ParallelCalc::s_coeff = {
  std::vector<double>{1, -1},
  std::vector<double>{1, -8, 8, -1},
  std::vector<double>{-1, 9, -45, 45, -9, 1},
  std::vector<double>{3, -32, 168, -672, 672, -168, 32, -3}
};

const std::array<std::vector<double>, 4> ParallelCalc::s_coeff2 = {
  std::vector<double>{1, -1},
  std::vector<double>{-2, -1, 1, 2},
  std::vector<double>{-3, -2, -1, 1, 2, 3},
  std::vector<double>{-4, -3, -2, -1, 1, 2, 3, 4}
};

const std::array<double, 4> ParallelCalc::s_dd = {2, 12, 60, 840};

void ParallelCalc::cleanGradientsParallel(TVector& grad, const TVector& mask,
                                         int nThreads)
{
  if (nThreads <= 0)
    nThreads = std::thread::hardware_concurrency();

  const Eigen::Index size = grad.rows();

#ifdef _OPENMP
  #pragma omp parallel for num_threads(nThreads)
#endif
  for (Eigen::Index i = 0; i < size; ++i) {
    if (!std::isfinite(grad[i]) || std::isnan(grad[i])) {
      grad[i] = 0.0;
    }
  }

  if (mask.rows() == size) {
    // Apply mask in parallel chunks
    const Eigen::Index chunkSize = std::max(size / nThreads, Eigen::Index(1));
#ifdef _OPENMP
    #pragma omp parallel for num_threads(nThreads)
#endif
    for (Eigen::Index i = 0; i < size; i += chunkSize) {
      const Eigen::Index end = std::min(i + chunkSize, size);
      grad.segment(i, end - i) = grad.segment(i, end - i).cwiseProduct(mask.segment(i, end - i));
    }
  }
}

} // namespace Avogadro::Calc
