package neuralnetwork;

public class InputNeuron extends Neuron {
    private Double value;

    private InputNeuron(IntegrationsFunktion sigma, TransferFunktion phi) {
        super(sigma, phi);
    }

    private InputNeuron(IntegrationsFunktion sigma, TransferFunktion phi, Double output) {
        super(sigma, phi, output);
    }

    public InputNeuron(Double value) {
        this(null, null);
        this.value = value;
    }

    @Override
    public void calculate() {
        output = value;
    }

    public void setValue(double value) {
        this.value = value;
    }
}
