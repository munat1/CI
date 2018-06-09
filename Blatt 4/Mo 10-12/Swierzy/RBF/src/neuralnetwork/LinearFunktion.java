package neuralnetwork;

public class LinearFunktion implements TransferFunktion {

    private double factor;

    public LinearFunktion(double factor) {
        this.factor = factor;
    }

    @Override
    public double calculate(double x) {
        return factor * x;
    }

}
