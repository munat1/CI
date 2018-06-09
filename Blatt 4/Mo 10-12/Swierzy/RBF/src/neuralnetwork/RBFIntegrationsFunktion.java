package neuralnetwork;

import java.util.Vector;

public interface RBFIntegrationsFunktion extends IntegrationsFunktion {
    double calculate(Vector<Double> w, Vector<Double> x, Vector<Double> center);
}
