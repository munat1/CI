package neuralnetwork;

import java.util.Vector;

import static java.lang.Integer.min;

public class Util {

    public static double eukSkaProd(Vector<Double> a, Vector<Double> b) {
        double result = 0;
        for (int i = 0; i < min(a.size(), b.size()); i++)
            result += a.get(i) * b.get(i);
        return result;
    }

    public static double eukSkaProdStrict(Vector<Double> a, Vector<Double> b) throws IllegalArgumentException {
        if (a.size() != b.size()) throw new IllegalArgumentException("Vector size does not match");
        double result = 0;
        for (int i = 0; i < a.size(); i++)
            result += a.get(i) * b.get(i);
        return result;
    }

    public static Vector<Double> vectorDiff(Vector<Double> a, Vector<Double> b) {
        Vector<Double> output = new Vector<>(min(a.size(), b.size()));
        for (int i = 0; i < min(a.size(), b.size()); i++)
            output.add(a.get(i) - b.get(i));
        return output;
    }
}
