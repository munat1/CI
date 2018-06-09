#include "stdafx.h"
#include "Layer.h"
#include "RBFUnit.h"

namespace NN {
	//The id of a layer should never be bigger than 2^16-1
	Layer::Layer(Layertype layertype, LayerId id) : layertype(layertype), id(id) {
		assert(id < MAX_NUM_LAYER);
	}


	Layer::~Layer()
	{
	}

	void Layer::addNeuron(std::function<double(double)> activationFunktion,
		std::function<double(double)> activationFunctionDerivative, bool isBias) {
		neurons.push_back(std::make_shared<Neuron>(Neuron(activationFunktion, id *MAX_NUM_NEURONS_PER_LAYER + getNumberOfNeurons(), activationFunctionDerivative, isBias)));
	}
	void Layer::addRBFUnit(Math::Point2D center, double width) {
		assert(Layertype::INPUT_LAYER == layertype);
		neurons.push_back(std::make_shared<RBFUnit>(RBFUnit(center, width, id *MAX_NUM_NEURONS_PER_LAYER + getNumberOfNeurons())));
	}
	void Layer::initializeRBFInputs(Math::Point2D& inputValue)const {
		assert(Layertype::INPUT_LAYER == layertype);
		for (auto &neuron : neurons) {
			neuron->setInput(inputValue);
		}
	}

	const size_t Layer::getNumberOfNeurons() const {
		return neurons.size();
	}

	const std::vector<std::shared_ptr<Neuron>> &  Layer::getNeurons() const {
		return neurons;
	}

	const Layer::Layertype Layer::getLayertype() const {
		return layertype;
	}

	const bool Layer::isValidNeuronIndex(NeuronId neuron)const {
		return neuron < getNumberOfNeurons();
	}

	Neuron & Layer::getNeuron(NeuronId neuron) {
		return * neurons.at(inLayerId(neuron));
	}

	//This method should not be called for an output layer, we adjust the deltas of this layer by setTrainingError
	void Layer::calculateAllDeltas() const {
		assert(Layertype::OUTPUT_LAYER != layertype);
		//We will never need delta values of input neurons.
		if (Layertype::INPUT_LAYER == layertype) return;
		for (NeuronId id = 0; id < getNumberOfNeurons(); ++id) {
			//These neurons are all in hidden layers, so the deltas will not be computed indirectly
			neurons.at(id)->calculateDelta(0, false);
		}
	}

	void Layer::adjustAllWeights(double learningRate, bool isOnline) const {
		for (NeuronId id = 0; id < getNumberOfNeurons(); ++id) {
			neurons.at(id)->adjustAllWeights(learningRate, isOnline);
		}
	}

	const double Layer::summedOutput() const {
		assert(!neurons.empty());
		double outputValue = 0;
		for (auto & neuron : neurons) {
			outputValue += neuron->calculateOutput();
		}
		return outputValue;
	}

	void Layer::propagateOutputs() const {
		for (NeuronId id = 0; id < getNumberOfNeurons(); ++id) {
			neurons.at(id)->propagateOutput();
		}
	}

	void Layer::initializeInputs(std::vector<double> &inputValues)const {
		assert(Layertype::INPUT_LAYER == layertype);
		for (NeuronId id = 0; id < getNumberOfNeurons(); ++id) {
			neurons.at(id)->addToInput(inputValues[id]);
		}
	}
}//namespace NN
