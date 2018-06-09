#ifndef LAYER_H
#define LAYER_H
#include "Neuron.h"

namespace NN {
	using LayerId = uint32_t;
	/*
	* Basic class for layer of a feed forward net.
	*/
	class Layer
	{
	public:
		//Specifying different layer types.
		enum Layertype {
			INPUT_LAYER,
			HIDDEN_LAYER,
			OUTPUT_LAYER,
		};
		Layer(Layertype layertype, LayerId id);
		~Layer();
		void addNeuron(std::function<double(double)> activationFuntion, std::function<double(double)> activationFunctionDerivative = nullptr,
			bool isBias = false);
		void propagateOutputs() const;
		void adjustAllWeights(double learningRate, bool isOnline) const;
		void calculateAllDeltas() const;
		void initializeInputs(std::vector<double> &inputValues) const;
		void initializeRBFInputs(Math::Point2D & inputValue)const;
		void addRBFUnit(Math::Point2D center, double width);
		const std::vector<std::shared_ptr<Neuron>> & getNeurons() const;
		const size_t getNumberOfNeurons() const;
		const Layertype getLayertype() const;
		const double summedOutput() const;
		const bool isValidNeuronIndex(NeuronId neuron)const;
		Neuron & getNeuron(NeuronId neuron);
	private:
		std::vector<std::shared_ptr<Neuron>> neurons;
		Layertype layertype;
		LayerId id;
	};
}
#endif

