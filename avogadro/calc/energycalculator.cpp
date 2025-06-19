/******************************************************************************
  This source file is part of the Avogadro project.
  This source code is released under the 3-Clause BSD License, (see "LICENSE").
******************************************************************************/

#include "energycalculator.h"
#include "parallelcalc.h"

#include <iostream>
#include <thread>

namespace Avogadro::Calc {

void EnergyCalculator::gradient(const TVector& x, TVector& grad)
{
  // Use lambda to capture this for value calculations
  auto valueFunc = [this](const TVector& pos) { return this->value(pos); };

  // Get optimal thread count
  int nThreads = std::thread::hardware_concurrency();

  // Use parallel implementation with accuracy level 0 (basic accuracy)
  // The last parameter (nThreads) controls parallel execution
  ParallelCalc::finiteGradientParallel(valueFunc, x, grad, 0, nThreads);

  // Clean gradients and apply mask using same thread count
  if (m_mask.size() > 0) {
    ParallelCalc::cleanGradientsParallel(grad, m_mask, nThreads);
  }
}

void EnergyCalculator::cleanGradients(TVector& grad)
{
  // Only clean and mask if we have a valid mask
  if (m_mask.size() > 0) {
    int nThreads = std::thread::hardware_concurrency();
    ParallelCalc::cleanGradientsParallel(grad, m_mask, nThreads);
  }
}

} // namespace Avogadro::Calc
