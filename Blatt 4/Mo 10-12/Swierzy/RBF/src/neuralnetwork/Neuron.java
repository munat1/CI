package neuralnetwork;

import java.util.Vector;

public class Neuron {

    IntegrationsFunktion sigma;
    TransferFunktion phi;

    private Vector<Double> weights;
    private Vector<Double> inputs;

    Double output;

    public Neuron(IntegrationsFunktion sigma, TransferFunktion phi) {
        this(sigma, phi, 0d);
    }

    public Neuron(IntegrationsFunktion sigma, TransferFunktion phi, Double output) {
        this.sigma = sigma;
        this.phi = phi;
        this.output = output;
    }

    public Vector<Double> getWeights() {
        return weights;
    }

    public void setWeights(Vector<Double> weights) {
        this.weights = weights;
    }

    public Vector<Double> getInputs() {
        return inputs;
    }

    public void setInputs(Vector<Double> inputs) {
        this.inputs = inputs;
    }

    public Double getOutput() {
        return output;
    }

    public void calculate() {
        output = phi.calculate(sigma.calculate(weights, inputs));
    }
}
