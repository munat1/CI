package neuralnetwork;

import static java.lang.Math.exp;

public class GaussFunktion implements TransferFunktion {

    private double sigma = 1;

    public void setSigma(double sigma) {
        if (sigma == 0) throw new IllegalArgumentException();
        this.sigma = sigma;
    }

    @Override
    public double calculate(double x) {
        return exp(-1 * x * x / 2 / sigma / sigma);
    }

}
