#include "stdafx.h"
#include "Neuron.h"
namespace NN {
	
	Neuron::Neuron(std::function<double(double)> function, NeuronId id, std::function<double(double)> derivative, bool isBias) :
		activationFunction(function), inputValue(0), _id(id), activationFunctionDerivative(derivative), delta(0), _isBias(isBias)
	{
		if (isBias) inputValue = 1;
	}

	Neuron::~Neuron()
	{
	}

	void Neuron::addOutputEdge(Neuron&  destination, double weight) {
		assert(!destination.isBias());
		outputs.push_back({ std::shared_ptr<Neuron>(&destination), weight });
		//Also creating a new entry for the collected delta values.
		collectedDeltas.push_back(0.);
	}

	
	void Neuron::adjustAllWeights(double learningRate, bool isOnline) {
		for (std::size_t i = 0; i < outputs.size(); ++i) {
			//Adjusting the weight according to the lecture.
			double delta = -learningRate * outputs[i].first->getDelta() * calculateOutput();
			if (isOnline) {
				outputs[i].second += delta;
			}
			//Save the delta value for later
			else {
				collectedDeltas[i] += delta;
			}
		}
	}

	void Neuron::addToInput(double input) {
		inputValue += input;
	}
	inline const double Neuron::calculateOutput() const {
		return activationFunction(inputValue);
	}

	inline const double Neuron::getDelta() const {
		return delta;
	}

	inline const NeuronId Neuron::id() const {
		return _id;
	}

	inline const bool Neuron::isBias() const {
		return _isBias;
	}

	void Neuron::clearInput() {
		inputValue = 0;
	}

	//Calculation is done as stated in the lecture
	void Neuron::calculateDelta(double correctSolution, bool isInOutputLayer) {
		double derivative = (activationFunctionDerivative == nullptr) ?
			Math::estimateDerivative(activationFunction, inputValue) : activationFunctionDerivative(inputValue);
		if (isInOutputLayer) {
			delta = 2*derivative * (calculateOutput() - correctSolution);
		}
		else {
			double weightedDeltaSum = 0;
			for (auto &pairOfDestinationAndWeight : outputs) {
				weightedDeltaSum += pairOfDestinationAndWeight.first->getDelta() * pairOfDestinationAndWeight.second;
			}
			delta = derivative * weightedDeltaSum;
		}
	}


	void Neuron::propagateOutput() {
		double calculatedOutput = calculateOutput();
		double currentOutputWeight = 0;
		for (auto &pairOfDestinationAndWeight : outputs) {
			currentOutputWeight = pairOfDestinationAndWeight.second;
			pairOfDestinationAndWeight.first->addToInput(currentOutputWeight * calculatedOutput);
		}
	}

	void Neuron::addCollectedWeights() {
		for (auto i = 0; i < outputs.size(); ++i) {
			outputs[i].second += collectedDeltas[i];
			collectedDeltas[i] = 0;
		}
	}
}//namespace NN

