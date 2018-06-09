#include "stdafx.h"
#include "NeuralNet.h"
#include <iostream>

namespace NN {
	NeuralNet::NeuralNet()
	{
	}

	
	void NeuralNet::addLayers(std::size_t number) {
		assert(number > 1);
		layers.push_back(Layer(Layer::Layertype::INPUT_LAYER, 0));
		for (std::size_t i = 1;i < number - 1;i++) {
			layers.push_back(Layer(Layer::Layertype::HIDDEN_LAYER, i));
		}
		layers.push_back(Layer(Layer::Layertype::OUTPUT_LAYER, number - 1));
	}

	void NeuralNet::addNeuronToLayer(size_t layerIndex, std::function<double(double)> activationFunction,
		std::function<double(double)> activationFunctionDerivative) {
		if (!isValidLayerIndex(layerIndex)) {
			std::cout << "Could not add the given neuron, the layer index is not valid.";
		}
		else {
			layers[layerIndex].addNeuron(activationFunction, activationFunctionDerivative);
		}
	}
	void NeuralNet::addRBFUnit(Math::Point2D center, double width) {
		layers[0].addRBFUnit(center, width);
	}

	NeuralNet::~NeuralNet()
	{
	}

	bool NeuralNet::isValidLayerIndex(size_t layerIndex) {
		return (layerIndex < layers.size());
	}

	const double NeuralNet::calculateOutput(std::vector<double> &inputValues)const {
		assert(!layers.empty());
		assert(inputValues.size() <= layers[0].getNumberOfNeurons());
		layers[0].initializeInputs(inputValues);

		for (size_t layerIndex = 0; layerIndex < getNumberOfLayers() - 1;++layerIndex) {
			propagateOutputs(layerIndex);
		}
		assert(layers.back().getLayertype() == Layer::Layertype::OUTPUT_LAYER);
		return summedOutput();
	}

	const double NeuralNet::calculateOutput(Math::Point2D &inputValue)const {
		assert(!layers.empty());
		layers[0].initializeRBFInputs(inputValue);

		for (size_t layerIndex = 0; layerIndex < getNumberOfLayers() - 1;++layerIndex) {
			propagateOutputs(layerIndex);
		}
		assert(layers.back().getLayertype() == Layer::Layertype::OUTPUT_LAYER);
		return summedOutput();
	}


	void NeuralNet::setTrainingError(std::vector<double> & correctSolutions) {
		assert(correctSolutions.size() == layers[getNumberOfLayers() - 1].getNumberOfNeurons());
		std::size_t index = 0;
		for (auto & neuron : layers[getNumberOfLayers() - 1].getNeurons()) {
			neuron->calculateDelta(correctSolutions[index], true);
			++index;
		}
	}

	void NeuralNet::clearInputs() {
		for (auto & layer : layers) {
			for (auto & neuron : layer.getNeurons()) {
				if (!neuron->isBias()) {
					neuron->clearInput();
				}
			}
		}
	}

	void NeuralNet::addBias() {
		assert(getNumberOfLayers() > 1);
		layers[getNumberOfLayers() - 2].addNeuron(Math::identity, nullptr, true);
	}

	void NeuralNet::addConnection(NeuronId fromNeuron, NeuronId toNeuron, double weight) {
		assert(isValidLayerIndex(layerId(fromNeuron)));
		assert(isValidLayerIndex(layerId(toNeuron)));
		assert(layerId(fromNeuron)< layerId(toNeuron));
		assert(layers[layerId(fromNeuron)].isValidNeuronIndex(inLayerId(fromNeuron)));
		assert(layers[layerId(toNeuron)].isValidNeuronIndex(inLayerId(toNeuron)));
		layers[layerId(fromNeuron)].getNeuron(inLayerId(fromNeuron)).addOutputEdge(layers[layerId(toNeuron)]
			.getNeuron(inLayerId(toNeuron)), weight);
	}

	inline const size_t NeuralNet::getNumberOfLayers() const {
		return layers.size();
	}

	inline const double NeuralNet::summedOutput() const {
		return layers.back().summedOutput();
	}

	inline void NeuralNet::propagateOutputs(LayerIndex layerIndex) const {
		layers[layerIndex].propagateOutputs();
	}
	/**
	* This function performs the backpropagation algorithm with a given @param learningRate. We want to start with the Layer @param from
	* assuming that all the delta values have already been calculated for the layer "from". We update all weights on arcs between the given layer indices.
	* We implicitly calculate and store all the delta values for neurons in layer "to",...,"from"-1.
	* NOTE: Make sure to call setTrainingError before, if you want to start with the output layer
	*/
	void NeuralNet::performBackpropagation(LayerIndex from, LayerIndex to, double learningRate, bool isOnline) {
		assert(from > to);
		assert(isValidLayerIndex(from));
		assert(isValidLayerIndex(to));
		LayerIndex layer = from - 1;
		do {
			layers.at(layer).calculateAllDeltas();
			layer--;
		} while (layer != to - 1);
		layer = from - 1;
		do {
			layers[layer].adjustAllWeights(learningRate, isOnline);
			layer--;
		} while (layer != to - 1);

	}

	void NeuralNet::adjustWeights() const {
		for (auto const &layer : layers) {
			for (auto &neuron : layer.getNeurons()) {
				neuron->addCollectedWeights();
			}
		}
	}
}
