package neuralnetwork;

import java.util.Vector;

import static java.lang.Math.sqrt;

public class RBFsigma implements RBFIntegrationsFunktion {

    /**
     * @param w RBF Neuron ignores w
     * @param x Inputs
     * @param center RBF Neuron center
     * @return ||x-c||
     */
    @Override
    public double calculate(Vector<Double> w, Vector<Double> x, Vector<Double> center) {
        Vector<Double> diff = Util.vectorDiff(x, center);
        return sqrt(Util.eukSkaProd(diff, diff));
    }


    @Override
    public double calculate(Vector<Double> w, Vector<Double> x) {
        Vector<Double> zero = new Vector<>(x.size());
        for (int i = 0; i < x.size(); i++)
            zero.add(0d);
        return calculate(w, x, zero);
    }
}
