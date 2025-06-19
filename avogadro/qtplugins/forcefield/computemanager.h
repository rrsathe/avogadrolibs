#ifndef AVOGADRO_QTPLUGINS_FORCEFIELD_COMPUTEMANAGER_H
#define AVOGADRO_QTPLUGINS_FORCEFIELD_COMPUTEMANAGER_H

#include <QObject>
#include <Eigen/Core>
#include <atomic>
#include <future>
#include <thread>
#include <vector>
#include <memory>
#include <avogadro/core/array.h>
#include <avogadro/core/vector.h>

namespace Avogadro {
namespace Calc {
class EnergyCalculator;
}

namespace QtPlugins {

class ComputeManager : public QObject {
  Q_OBJECT
public:
  explicit ComputeManager(QObject* parent = nullptr);
  ~ComputeManager() override;

  // Disable copy/move operations
  ComputeManager(const ComputeManager&) = delete;
  ComputeManager& operator=(const ComputeManager&) = delete;
  ComputeManager(ComputeManager&&) = delete;
  ComputeManager& operator=(ComputeManager&&) = delete;

  static constexpr size_t DefaultThreadCount = 4;

  // Initialize thread pool and GPU resources if available
  bool initialize(size_t numThreads = DefaultThreadCount);

  // Set the calculator to use
  void setCalculator(Calc::EnergyCalculator* calculator) { m_calculator = calculator; }

  // Parallel force/energy calculation
  struct ComputeResult {
    double energy;
    Eigen::VectorXd gradient;
    bool success;
  };

  // Synchronous computation methods
  std::pair<double, Eigen::VectorXd> computeEnergyAndGradients(
    const Core::Array<Vector3>& positions,
    Calc::EnergyCalculator* calculator,
    [[maybe_unused]] size_t numThreads = DefaultThreadCount);

  struct PositionUpdateResult {
    bool success;
    Core::Array<Vector3> positions;
    Core::Array<Vector3> forces;
  };

  static PositionUpdateResult updateAtomPositions(
    const Eigen::VectorXd& positions,
    const Eigen::Index atomCount,
    const Eigen::VectorXd& gradient);

  // Async computation methods
  std::future<ComputeResult> computeEnergyGradientAsync(
    const Eigen::VectorXd& positions,
    const Eigen::VectorXd& mask);

  // Cancel ongoing computations
  void cancelComputation();

private:
  std::vector<std::thread> m_threads;
  std::atomic<bool> m_cancelFlag;
  size_t m_numThreads;
  bool m_gpuAvailable;
  Calc::EnergyCalculator* m_calculator;
};

} // namespace QtPlugins
} // namespace Avogadro

#endif // AVOGADRO_QTPLUGINS_FORCEFIELD_COMPUTEMANAGER_H
