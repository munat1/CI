package neuralnetwork;

import java.util.Vector;

public class RBFNeuron extends Neuron {

    private Vector<Double> center;

    private RBFNeuron(IntegrationsFunktion sigma, TransferFunktion phi) {
        super(sigma, phi);
    }

    private RBFNeuron(IntegrationsFunktion sigma, TransferFunktion phi, Double output) {
        super(sigma, phi, output);
    }

    public RBFNeuron(IntegrationsFunktion sigma, TransferFunktion phi, Vector<Double> center) {
        super(sigma, phi);
        this.center = center;
    }

    @Override
    public void calculate() {
        if (sigma instanceof RBFIntegrationsFunktion) {
            output = phi.calculate(((RBFIntegrationsFunktion) sigma).calculate(getWeights(), getInputs(), center));
        } else {
            super.calculate();
        }
    }

    public Vector<Double> getCenter() {
        return center;
    }
}
