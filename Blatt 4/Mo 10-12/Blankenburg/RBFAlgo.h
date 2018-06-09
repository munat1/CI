#ifndef RBFALGO_H
#define RBFALGO_H

#include "NeuralNet.h"
namespace NN {
	namespace RBF {

		//Initializing the centers with random points of the input data vector.
		void initializePointPositions(std::vector<Math::Point2D> & data, std::vector<Math::Point2D> & centers) {
			if (0 == data.size()) { return; }
			assert(data.size() > centers.size());
			boost::random::mt19937 gen;
			boost::random::uniform_int_distribution<> dist(0, 10 * data.size());
			for (std::size_t i = 0; i < centers.size(); ++i) {
				auto index = dist(gen) % (data.size() - i);
				centers.at(i) = data.at(index);
				std::swap(data[index], data[data.size() - 1]);
			}

		}

		/*
		* Returns if a cluster is changed and computes all the new clusters and the centers as the mean point of the cluster they own.
		*/
		bool isChangedCluster(std::vector<Math::Point2D> & data, std::vector<Math::Point2D> & centers, std::vector<std::size_t> & clusterIds) {
			bool isChanged = false;
			//Vector that saves the sum of the points of the new cluster and the number of points that belong to it.
			std::vector<std::pair<Math::Point2D, std::size_t>> newCenterSum(centers.size(), { { 0,0 },0 });
			std::size_t currentClosestCenterId;
			double currentClosestDist;
			for (std::size_t dataIndex = 0; dataIndex < data.size(); dataIndex++) {
				//Calculate the new closest center for the data point
				currentClosestDist = Math::euclideanDist(data[dataIndex], centers[clusterIds[dataIndex]]);
				currentClosestCenterId = clusterIds[dataIndex];
				for (std::size_t clusterIndex = 0; clusterIndex < centers.size(); clusterIndex++) {
					//Choose the nearest center.
					if (currentClosestDist > Math::euclideanDist(data[dataIndex], centers[clusterIndex])) {
						currentClosestCenterId = clusterIndex;
						currentClosestDist = Math::euclideanDist(data[dataIndex], centers[clusterIndex]);
					}
				}
				//Check if the center has changed and saving the new cluster number for the data point
				if (clusterIds[dataIndex] != currentClosestCenterId) {
					isChanged = true;
					clusterIds[dataIndex] = currentClosestCenterId;
				}
				newCenterSum[clusterIds[dataIndex]].first += data[dataIndex];
				newCenterSum[clusterIds[dataIndex]].second++;
			}
			//Getting the mean point for every cluster
			for (std::size_t clusterIndex = 0; clusterIndex < centers.size(); clusterIndex++) {
				assert(newCenterSum[clusterIndex].second != 0);
				centers[clusterIndex] = newCenterSum[clusterIndex].first / newCenterSum[clusterIndex].second;
			}
			return isChanged;
		}

		
		const std::vector<Math::Point2D> kMeansClustering(std::vector<Math::Point2D> & data, std::size_t k) {
			assert(k > 0);
			std::vector<Math::Point2D> centers(k);
			std::vector<std::size_t> clusterIds(data.size(), 0);
			initializePointPositions(data, centers);
			const std::size_t MAX_ITERATIONS = 20;
			std::size_t count = 0;
			//Does the full computation of the new clusters and returns if any has changed
			while (isChangedCluster(data, centers, clusterIds) && count < MAX_ITERATIONS) {
				count++;
			}
			return centers;
		}
		
		
		 double standardDeviation(std::vector<Math::Point2D> &points) {
			 assert(0 < points.size());
			 std::size_t numOfPairs = (points.size() * (points.size() - 1)) / 2;
			 double sumSquaredDistances = 0;
			 double pSumClosestDist = 0;
			 for (std::size_t i = 0;i < points.size();++i) {
				 for (std::size_t j = i + 1; j < points.size(); ++j) {
					 sumSquaredDistances += Math::euclideanDist(points.at(i), points.at(j)) *Math::euclideanDist(points.at(i), points.at(j));
				 }
			 }
			 return std::sqrt(sumSquaredDistances / numOfPairs);
		 }

		 //Calculating the mean distance of the closest p pairs (can be used to calculate the width for the rbf units)
		 double pMeanClosestDistance(std::size_t p, const std::vector<Math::Point2D> & points) {
			 assert(0 < points.size());
			 std::size_t numOfPairs = (points.size() * (points.size() - 1)) / 2;
			 assert(0 < p && numOfPairs >= p);
			 std::vector<double> distances(numOfPairs);
			 std::size_t index = 0;
			 double pSumClosestDist = 0;
			 for (std::size_t i = 0;i < points.size();++i) {
				 for (std::size_t j = i + 1; j < points.size(); ++j) {
					 distances.at(index) = Math::euclideanDist(points.at(i), points.at(j));
					 index++;
				 }
			 }
			 std::sort(distances.begin(), distances.end());
			 for (std::size_t i = 0;i< p;++i) {
				 pSumClosestDist += distances.at(i);
			 }
			 return pSumClosestDist / p;
		 }
	}
}
#endif