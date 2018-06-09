package neuralnetwork;

import java.util.Vector;

public class SkalarproduktFunktion implements IntegrationsFunktion {
    @Override
    public double calculate(Vector<Double> w, Vector<Double> x) {
        return Util.eukSkaProdStrict(w, x);
    }
}
