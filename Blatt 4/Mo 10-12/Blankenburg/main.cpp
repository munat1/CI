

#include "stdafx.h"
#include "NeuralNet.h"
#include "RBFAlgo.h"
#include <iostream>
#include <fstream>

//Writing outputs in a given file
void write_points_in_files(std::vector<Math::Point2D> & data1, std::vector<Math::Point2D> & data2, std::vector<Math::Point2D> & centers) {
	std::ofstream out1("data1.txt", std::ofstream::out);
	std::ofstream out2("data2.txt", std::ofstream::out);
	std::ofstream out3("centers.txt", std::ofstream::out);
	for (auto & point : data1) {
		out1 << point.x1 << " " << point.x2 << std::endl;
	}
	out1.close();
	for (auto & point : data2) {
		out2 << point.x1 << " " << point.x2 << std::endl;
	}
	out2.close();
	for (auto & point : centers) {
		out3 << point.x1 << " " << point.x2 << std::endl;
	}
	out3.close();
}

void prepareInputData(std::vector<Math::Point2D> &firstClass, std::vector<Math::Point2D> &secondClass) {
	for (int u = 0; u < firstClass.size(); u++) {
		double multiplier = 200.0 / firstClass.size();
		double input = u * multiplier;
		firstClass.at(u).x1 = Math::sheet4FirstCathx1(input);
		firstClass.at(u).x2 = Math::sheet4FirstCathx2(input);
		secondClass.at(u).x1 = Math::sheet4SecondCathx1(input);
		secondClass.at(u).x2 = Math::sheet4SecondCathx2(input);
		
		//Optional use the polar coordinates
		//firstClass.at(u) = Math::polarCoord(firstClass.at(u));
		//secondClass.at(u) = Math::polarCoord(secondClass.at(u));
	}
}

/*
* Initializing the neural net as it is specified in sheet 3
*/
void doNetSetup(NN::NeuralNet & net, const std::vector<Math::Point2D> &centers, double width) {
	boost::random::mt19937 gen(time(0));
	net.addLayers(2);
	//Add the input neurons
	for (auto const & center: centers) {
		net.addRBFUnit(center, width);
	}
	//Add the output neuron.
	net.addNeuronToLayer(1, Math::tanh, Math::tanhDerivative);

	//Adding all connections and initializing random weights
	Math::Interval interval = { -0.1,0.1 };
	for (int i = 0;i < centers.size();i++) {
		net.addConnection(i,NN::MAX_NUM_NEURONS_PER_LAYER , Math::getRandomDouble(interval, gen));
	}
}

/*
* The following methods are very similar. They write their results into txt-files, which we use later to plot the data with python.
*/
void performTask1(NN::NeuralNet & net, std::vector<Math::Point2D> &firstClass, std::vector<Math::Point2D> &secondClass) {
	//Defining the output file stream
	std::ofstream out("outputs.txt", std::ofstream::out);
	double squaredError = 0;
	double mse = 1;
	int numIterations = 0;
	std::vector<double> correctSolution(1);
	double learningRate = 0.05;
	double previousMse = 2;
	while ((mse > 0.01) && (numIterations < 1000)) {
		mse = 0;
		correctSolution.at(0) = 1;
		//Do online learning for all training examples
		for (auto & input : firstClass) {
			double calculatedOutput = net.calculateOutput(input);
			mse += (correctSolution.at(0) - calculatedOutput)* (correctSolution.at(0) - calculatedOutput);
			net.setTrainingError(correctSolution);
			net.performBackpropagation(1, 0, learningRate, true);
			net.clearInputs();
		}
		correctSolution.at(0) = -1;
		for (auto & input : secondClass) {
			double calculatedOutput = net.calculateOutput(input);
			mse += (correctSolution.at(0) - calculatedOutput)* (correctSolution.at(0) - calculatedOutput);
			net.setTrainingError(correctSolution);
			net.performBackpropagation(1, 0, learningRate, true);
			net.clearInputs();
		}
		//Optional: adjust the weights if we do not learn online
		//net.adjustWeights();
		mse /= (firstClass.size()+ secondClass.size());
		//Adjust learning rate according to the change in the mse
		learningRate *= (previousMse < mse) ? 0.9 : 1.01;
		previousMse = mse;
		++numIterations;
	}
	//Finally write the outputs of the trained net in a file
	for (double x = -16; x <= 16; x += 0.1) {
		for (double y = -16; y <= 16; y+= 0.1) {
			Math::Point2D temp = { x,y };
			out << x << " " << y << " "<<net.calculateOutput(temp)<<std::endl;
			net.clearInputs();
		}
	}
	out.close();
}

int main(){
	NN::NeuralNet net;
	std::vector<Math::Point2D> firstClass(200, { 0,0 });
	std::vector<Math::Point2D> secondClass(200, { 0,0 });
	//Save the coordinates
	prepareInputData(firstClass, secondClass);
	//Calculate center positions
	std::vector<Math::Point2D> centers1 = NN::RBF::kMeansClustering(firstClass, 25);
	std::vector<Math::Point2D> centers2 = NN::RBF::kMeansClustering(secondClass, 25);
	centers1.insert(centers1.end(), centers2.begin(), centers2.end());
	write_points_in_files(firstClass, secondClass, centers1);

	//Optional use width as standard deviation of the center distances.
	//auto width = NN::RBF::standardDeviation(centers1);

	//Optional use centers that are spread over the input space
	//std::vector<Math::Point2D> centers1 = Math::pointsInSquare();

	//construct the net
	doNetSetup(net, centers1, 0.5);
	//Do the learning
	performTask1(net, firstClass, secondClass);
}

