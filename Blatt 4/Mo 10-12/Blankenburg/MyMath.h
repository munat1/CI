#ifndef MYMATH_H
#define MYMATH_H
#define _USE_MATH_DEFINES
#include <algorithm>
#include <functional>
#include <math.h>
#include <stdint.h>
#include <time.h>
#include <C:/Program Files/boost_1_67_0/boost/random/mersenne_twister.hpp>
#include <C:/Program Files/boost_1_67_0/boost/random/uniform_int_distribution.hpp>
namespace Math {

	struct Interval {
		double a;
		double b;
	};
	//Struct for two dimensional points with some operators that make interaction with other points and doubles easier.
	struct Point2D {
		double x1;
		double x2;
		Point2D &operator += (Point2D & other) {
			x1 += other.x1;
			x2 += other.x2;
			return *this;
		}
		Point2D &operator / (std::size_t divisor) {
			x1 /= divisor;
			x2 /= divisor;
			return *this;
		}
		Point2D &operator / (double divisor) {
			x1 /= divisor;
			x2 /= divisor;
			return *this;
		}
		Point2D &operator + (double summand) {
			x1 += summand;
			x2 += summand;
			return *this;
		}
		Point2D &operator - (double subtractor) {
			x1 -= subtractor;
			x2 -= subtractor;
			return *this;
		}
		Point2D &operator * (double multiplicator) {
			x1 *= multiplicator;
			x2 *= multiplicator;
			return *this;
		}
		Point2D &operator = (double value) {
			x1 = value;
			x2 = value;
			return *this;
		}
	};

	/*
	* Functions to calculate the coordinates of our data points
	*/
	inline static double sheet4FirstCathx1(double input) {
		assert(input >= -10);
		return 2 + std::sin(0.2*input + 8)*std::sqrt(input + 10);
	}

	inline static double sheet4FirstCathx2(double input) {
		assert(input >= -10);
		return -1 + std::cos(0.2*input + 8)*std::sqrt(input + 10);
	}

	inline static double sheet4SecondCathx1(double input) {
		assert(input >= -10);
		return 2 + std::sin(0.2*input -8)*std::sqrt(input + 10);
	}

	inline static double sheet4SecondCathx2(double input) {
		assert(input >= -10);
		return -1 + std::cos(0.2*input - 8)*std::sqrt(input + 10);
	}

	/*
	* Some activation functions.
	*/
	static double activationFunctionSheet2(double input) {
		return std::min(1.0, std::max(-1.0, 0.1*input));
	}

	static double constantOneFunction(double input) {
		return 1.0;
	}
	static double identity(double input) {
		return input;
	}
	static double fermiFunction(double input) {
		return 1.0 / (1.0 + std::exp(-input));
	}
	static double fermiFunctionDerivative(double input) {
		return fermiFunction(input)*(1.0 - fermiFunction(input));
	}
	static double tanh(double input) {
		return std::tanh(input);
	}
	static double tanhDerivative(double input) {
		return (1-(std::tanh(input)*std::tanh(input)));
	}


	inline const double euclideanDist(const Point2D & a, const Point2D & b) {
		return std::sqrt((a.x1 - b.x1)*(a.x1 - b.x1) + (a.x2 - b.x2)*(a.x2 - b.x2));
	}

	// A naive approach to estimate a derivative.
	template <typename Input>
	static double estimateDerivative(std::function<double(Input)> function, Input atPoint) {
		double epsilon = 0.0001;
		return (function(atPoint + epsilon) - function(atPoint - epsilon)) / (2 * epsilon);
	}

	inline Math::Point2D polarCoord(Math::Point2D point) {
		double r = euclideanDist(point, { 0,0 });
		if (r == 0)return { 0,0 };
		// phi will be in range [0,pi]
		double phi = std::acos(point.x1 / r);
		//Case distinction to get values in range [pi,2*pi] 
		if (point.x2 < 0) {
			return { r, 2 * M_PI - phi };
		}
		else {
			return { r, phi };
		}
	}

	static double functionSheet3(double x) {
		return cos(x / 2) + sin(5 / (abs(x) + 0.2)) - 0.1 * x;
	}


	inline double getRandomDouble(Interval interval, boost::random::mt19937 & gen) {
		const unsigned int MAX = 100000;
		boost::random::uniform_int_distribution<> dist(0,MAX);
		unsigned int randomNumber = dist(gen);
		return interval.a + (randomNumber) * (interval.b - interval.a) / MAX;
	}

	/*
	* Function that returns points which cover the input space of [-16,16] x [-16,16] evenly.
	*/
	static std::vector<Point2D> pointsInSquare() {
		std::vector<Point2D> points(49, { 0,0 });
		double width = 32. /9.;
		double x = width - 16.;
		double y = width - 16.;
		for (int i = 0; i < 49; ++i) {
			points[i] = { x,y };
			x += width;
			if (0 < i && i % 7 == 0) {
				y += width;
				x = width - 16.;
			}
		}
		return points;
	}

	static std::vector<Point2D>  getPoints(const std::function<double(double)> & x1Function, const std::function<double(double)> & x2Function,
		std::size_t size, double start, double increment) {
		std::vector<Point2D> points(size);
		double x = start;
		for (std::size_t i = 0; i < size; i++) {
			points[i] = { x1Function(x), x2Function(x) };
			x += increment;
		}
		return points;
	}
}
#endif 