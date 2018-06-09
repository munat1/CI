package neuralnetwork;

public class LinearNeuron extends Neuron {

    public LinearNeuron(IntegrationsFunktion sigma, Double factor) {
        super(sigma, new LinearFunktion(factor));
    }

}
