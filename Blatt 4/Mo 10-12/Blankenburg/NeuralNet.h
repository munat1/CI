#pragma once
#include "Layer.h"
namespace NN {
	using LayerIndex = std::size_t;

	class NeuralNet
	{
	public:
		NeuralNet();
		~NeuralNet();
		const double calculateOutput(std::vector<double> &inputValues) const;
		//Giving all the RBF Units of the input layer the same point as input value and propagating everything through the net.
		const double calculateOutput(Math::Point2D &inputValue)const;
		void performBackpropagation(LayerIndex from, LayerIndex to, double learningRate, bool isOnline);
		//Adjust the collected weights that we saved if we do not learn in online mode.
		void adjustWeights() const;
		void addNeuronToLayer(size_t layerIndex, std::function<double(double)> activationFunction,
			std::function<double(double)> activationFunctionDerivative = nullptr);
		//RBF Units are added to the input layer as default
		void addRBFUnit(Math::Point2D center, double width);
		//Adding layers. This should only be called once at the beginning, since we set the first one to 
		// be an input layer and the last one to be an output layer as default.
		void addLayers(std::size_t number);
		//Add one bias neuron to the layer that comes before the output layer.
		void addBias();
		void addConnection(NeuronId fromNeuron, NeuronId toNeuron, double weight);
		void setTrainingError(std::vector<double> & correctSolutions);
		void clearInputs();
		inline const size_t getNumberOfLayers() const;
	private:
		std::vector<Layer> layers;
		bool isValidLayerIndex(std::size_t layerIndex);
		inline const double summedOutput()const;
		inline void propagateOutputs(LayerIndex layer) const;
	};
}//namespace NN


