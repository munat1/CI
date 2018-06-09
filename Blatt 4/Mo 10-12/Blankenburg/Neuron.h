#ifndef NEURON_H
#define NEURON_H


#include <vector>
#include "assert.h"
#include "MyMath.h"
#include <memory>

namespace  NN {
	/*
	* We want to make shure that a NeuronId consists of 32 bits on every platform,
	* because we reserve the first 16 bits for the number of the layer and the last 16 bits for the number of the neuron in this layer.
	*/
	using NeuronId = uint32_t;
	static const int MAX_NUM_LAYER = (int)std::pow(2, 16);
	static const int MAX_NUM_NEURONS_PER_LAYER = (int)std::pow(2, 16);
	
	class Neuron
	{
	public:
		/*
		* constructor with all arguments. Derivative is optional, since we might not know it, or the function is not differentiable. In this case
		* we will apply a basic estimation of the derivative (see MyMath.h). isBias is also optional, since there is (usually) only one bias neuron.
		*/
		Neuron(std::function<double(double)> activationFuntion, NeuronId id, std::function<double(double)> activationFunctionDerivative = nullptr,
			bool isBiasNeuron = false);
		~Neuron();
		/**
		*Adding a double value to the input. We use this to collect all inputs from the neurons 
		* of the previous layer which are connected to this layer.
		*/
		void addToInput(double inputValue);
		/*
		* We save the output arcs as a pair of pointer to the destination and weight on that arc.
		*/
		void addOutputEdge(Neuron & destination, double weight);
		/*
		* We use this method to calculate the output of the neuron and to pass it to the next neurons.
		*/
		void propagateOutput();
		/*
		* Before calling this method one should compute all deltas of the destination neurons.
		*/
		virtual void adjustAllWeights(double learningRate, bool isOnline);
		/*
		* Resetting the input value to 0 (should be called before the next training example is processed)
		*/
		void clearInput();

		/*
		* Method to add all saved deltas to the weights if we are not learning in online mode.
		*/
		void addCollectedWeights();

		/*
		* getting the calculated output. This function is virtual, since we might want to use it in a different way for derived classes.
		*/
		virtual inline const double calculateOutput() const;
		//Should not be called here, sets the input for RBF Units
		virtual void setInput(const Math::Point2D & input) {};
		//Basic getters
		inline const double getDelta() const;
		inline const NeuronId id() const;
		inline const bool isBias() const;

		/*
		* Calculating the delta value of the neuron. If the neuron is in an output layer, we consider the correct solution,
		* else we calculate it from the delta values of the deeper layers
		*/
		void calculateDelta(double correctSolution, bool isInOutputLayer);
	
		//Protected, since we want to access the member variables in derived classes.
	protected:
		const std::function<double(double)> activationFunction;
		const std::function<double(double)> activationFunctionDerivative;
		double inputValue;
		double delta;
		const NeuronId _id;
		const bool _isBias;
		std::vector<std::pair<std::shared_ptr<Neuron>, double>> outputs;
		std::vector<double> collectedDeltas;
	};

	inline static const NeuronId inLayerId(NeuronId id) {
		return id % MAX_NUM_NEURONS_PER_LAYER;
	}
	inline static const NeuronId layerId(NeuronId id) {
		return id / MAX_NUM_NEURONS_PER_LAYER;
	}
}//namespace NN
#endif
