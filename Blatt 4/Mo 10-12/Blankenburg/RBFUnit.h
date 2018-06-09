#ifndef RBFUNIT_H
#define RBFUNIT_H

#include "Neuron.h"
/*
* RBFUnit as a subclass of Neuron. Most of the behavior is more or less the same,
* apart from having a center and a certain width of the kernel.
*/
namespace NN {
	class RBFUnit:  public Neuron	{
	public:
		RBFUnit(const Math::Point2D & center, double width ,NeuronId id);
		~RBFUnit();
		inline const double calculateOutput()const  override;
		void adjustAllWeights(double learningRate, bool isOnline) override;
		void setInput(const Math::Point2D & input) override;

	private:
		Math::Point2D _center;
		double _squaredSigma;
		Math::Point2D _input;

	};
}

#endif