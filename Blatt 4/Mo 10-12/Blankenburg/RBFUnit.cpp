#include "stdafx.h"
#include "RBFUnit.h"

namespace NN {
	//maintain this order
	RBFUnit::RBFUnit(const Math::Point2D & center,const double width, const NeuronId id): _center(center), _squaredSigma(width*width),Neuron(nullptr,id){
		_input = 0;
	}


	RBFUnit::~RBFUnit()
	{
	}

	void RBFUnit::setInput(const Math::Point2D  & input)  {
		_input = input;
	}


	//Calculating the weights, optionally also the center positions and the width can be learned.
	void RBFUnit::adjustAllWeights(double learningRate, bool isOnline) {
		double squaredSigmaDelta = 0;
		double x1delta = 0;
		double x2delta = 0;
		for (std::size_t i = 0; i < outputs.size(); ++i) {
			//Adjusting the weight according to the lecture.
			double weightDelta = -learningRate * outputs[i].first->getDelta() * calculateOutput();
			//Resulting delta for the width.
			squaredSigmaDelta += weightDelta * outputs[i].second
				* Math::euclideanDist(_center, _input)*Math::euclideanDist(_center, _input) / (2 * _squaredSigma * _squaredSigma);
			//Resulting delta for the center.
			x1delta += weightDelta * outputs[i].second / _squaredSigma * (_input.x1 - _center.x1);
			x2delta += weightDelta * outputs[i].second / _squaredSigma * (_input.x2 - _center.x2);
			if (isOnline) {
				outputs[i].second += weightDelta;
			}
			else {
				collectedDeltas[i] += weightDelta;
			}
		}
		_center.x1 += x1delta;
		_center.x2 += x2delta;
		//_squaredSigma += squaredSigmaDelta;
		//assert(_squaredSigma != 0);
	}

	inline const double RBFUnit::calculateOutput() const {
		assert(_squaredSigma != 0);
		//Typical RBF kernel
		return std::exp(-Math::euclideanDist(_center, _input)*Math::euclideanDist(_center, _input) / (2 * _squaredSigma));
	}
}//namespace NN

