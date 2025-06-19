#include "computemanager.h"
#include <avogadro/calc/energycalculator.h>
#include <avogadro/calc/parallelcalc.h>
#include <avogadro/core/array.h>
#include <avogadro/core/vector.h>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace Avogadro::QtPlugins {

using Calc::ParallelCalc;

ComputeManager::ComputeManager(QObject* parent)
  : QObject(parent), m_cancelFlag(false), m_numThreads(DefaultThreadCount),
    m_gpuAvailable(false), m_calculator(nullptr)
{
  initialize(DefaultThreadCount);
}

ComputeManager::~ComputeManager()
{
  cancelComputation();
  for (auto& thread : m_threads) {
    if (thread.joinable())
      thread.join();
  }
}

bool ComputeManager::initialize(size_t numThreads)
{
  // If numThreads is 0, use hardware concurrency
  if (numThreads == 0)
    numThreads = std::thread::hardware_concurrency();

  m_numThreads = numThreads;

  // Initialize OpenMP if available
#ifdef _OPENMP
  omp_set_num_threads(static_cast<int>(numThreads));
#endif

  return true;
}

std::pair<double, Eigen::VectorXd> ComputeManager::computeEnergyAndGradients(
  const Core::Array<Vector3>& positions, Calc::EnergyCalculator* calculator,
  [[maybe_unused]] size_t numThreads)
{
  const Eigen::Index atomCount = static_cast<Eigen::Index>(positions.size());
  const Eigen::Index dimension = 3 * atomCount;

  // Map position array to Eigen vector
  Eigen::VectorXd posVector(dimension);
  for (Eigen::Index i = 0; i < atomCount; ++i) {
    posVector.segment<3>(3 * i) = positions[i].cast<double>();
  }

  // Calculate energy
  const double energy = calculator->value(posVector);

  // Calculate gradients in parallel
  Eigen::VectorXd gradient(dimension);
  calculator->gradient(posVector, gradient);

  return { energy, gradient };
}

ComputeManager::PositionUpdateResult ComputeManager::updateAtomPositions(
  const Eigen::VectorXd& positions, const Eigen::Index atomCount,
  const Eigen::VectorXd& gradient)
{
  PositionUpdateResult result;
  result.success = true;
  result.positions.resize(static_cast<size_t>(atomCount));
  result.forces.resize(static_cast<size_t>(atomCount));

#ifdef _OPENMP
#pragma omp parallel for
#endif
  for (Eigen::Index i = 0; i < atomCount; ++i) {
    const Eigen::Vector3d pos = positions.segment<3>(3 * i);
    if (!pos.allFinite()) {
      result.success = false;
      continue;
    }

    result.positions[i] = Vector3(pos[0], pos[1], pos[2]);
    result.forces[i] = -0.1 * Vector3(gradient.segment<3>(3 * i).cast<float>());
  }

  return result;
}

std::future<ComputeManager::ComputeResult>
ComputeManager::computeEnergyGradientAsync(const Eigen::VectorXd& positions,
                                         const Eigen::VectorXd& mask)
{
  return std::async(std::launch::async, [this, positions, mask]() {
    ComputeResult result;
    result.success = false;
    result.gradient = Eigen::VectorXd::Zero(positions.size());
    result.energy = 0.0;

    if (!m_calculator) {
      return result;
    }

    // Use ParallelCalc for gradient calculation
    auto valueFunc = [this](const Eigen::VectorXd& pos) {
      return m_calculator->value(pos);
    };

    m_calculator->setMask(mask);
    result.energy = m_calculator->value(positions);

    // Convert thread count to int for ParallelCalc
    const int threads = static_cast<int>(std::min(
      static_cast<size_t>(std::numeric_limits<int>::max()), m_numThreads));
    
    // Use static methods
    ParallelCalc::finiteGradientParallel(valueFunc, positions, result.gradient, 0, threads);
    ParallelCalc::cleanGradientsParallel(result.gradient, mask, threads);
    result.success = true;

    return result;
  });
}

void ComputeManager::cancelComputation()
{
  m_cancelFlag = true;
}

} // namespace Avogadro::QtPlugins
