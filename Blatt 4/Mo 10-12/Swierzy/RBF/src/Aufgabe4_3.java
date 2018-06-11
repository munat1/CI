import neuralnetwork.*;

import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.PrintStream;
import java.util.*;

public class Aufgabe4_3 {

    private static InputNeuron inputNeuronX1, inputNeuronX2;
    private static List<RBFNeuron> hidden;
    private static LinearNeuron outputNeuron;
    private static Map<Point, Double> dataPositive;
    private static Map<Point, Double> dataNegative;
    private static Map<Point, Double> dataTotal;
    private static Vector<Double> weightsInner;
    private static Vector<Double> weightsOuter;

    public static void main(String[] args) throws FileNotFoundException {

        PrintStream outputStream = new PrintStream(new FileOutputStream("output4.3.r"));
        PrintStream dataStream = new PrintStream(new FileOutputStream("data4.3.r"));
        PrintStream hiddenStream = new PrintStream(new FileOutputStream("hidden4.3.r"));

        // Generate data
        dataPositive = new HashMap<>(200);
        dataNegative = new HashMap<>(200);
        for (int u = 0; u < 200; u++) {
            dataPositive.put(getPositivePoint(u), 1d);
            dataNegative.put(getNegativePoint(u), -1d);
        }
        dataTotal = new HashMap<>(400);
        dataTotal.putAll(dataPositive);
        dataTotal.putAll(dataNegative);

        // Extract 25 random center locations from each set
        Set<Point> rbfKeys = new HashSet<>(50);
        List<Point> posKeys = new ArrayList<>(dataPositive.keySet()), negKeys = new ArrayList<>(dataNegative.keySet());
        Collections.shuffle(posKeys);
        Collections.shuffle(negKeys);
        rbfKeys.addAll(posKeys.subList(0, 25));
        rbfKeys.addAll(negKeys.subList(0, 25));

        // Build RBF network
        // Weights
        weightsInner = new Vector<>(2); // weights between input and hidden layer, not used
        weightsInner.add(1d);
        weightsInner.add(1d);
        weightsOuter = new Vector<>(50); // weights between hidden and outer layer
        for (int i = 0; i < 50; i++)
            weightsOuter.add(1d); // Initialize all outer weights with 1
        // Neurons
        inputNeuronX1 = new InputNeuron(0d);
        inputNeuronX2 = new InputNeuron(0d);
        hidden = new ArrayList<>(50);
        for (Point p : rbfKeys) {
            RBFNeuron neuron = new RBFNeuron(new RBFsigma(), new GaussFunktion(), p.toVector());
            neuron.setWeights(weightsInner);
            hidden.add(neuron);
        }
        outputNeuron = new LinearNeuron(new SkalarproduktFunktion(), 1d);
        outputNeuron.setWeights(weightsOuter);

        System.out.println("Mean Square Error (untrained): " + meanSquareError());

        // Learning rate
        double eta = .03;
        learnExamples(eta);
        System.out.println("Mean Square Error (1x training): " + meanSquareError());
        for (int i = 0; i < 1000; i++)
            learnExamples(eta);

        System.out.println("Mean Square Error (1000x training): " + meanSquareError());

        // Print test data
        Aufgabe4_2.pointDiagram(dataStream, dataTotal);
        Aufgabe4_2.pointDiagram(hiddenStream, dataTotal);

        // Print scripts
        dataStream.println("library(\"plotrix\")");
        dataStream.println("png(\"E:\\\\coding\\\\RBF\\\\data4.3.png\")");
        dataStream.println("plot(data.frame(x=x, y=y, names=c(1:length(x)), row.names=\"names\"), col=color.scale(z,c(0,1,1),c(1,1,0),0), main=\"Test data\", xlab=\"x1\", ylab=\"x2\", pch=20)");
        dataStream.println("dev.off()");
        hiddenStream.println("library(\"plotrix\")");
        hiddenStream.println("png(\"E:\\\\coding\\\\RBF\\\\hidden4.3.png\")");
        hiddenStream.println("plot(data.frame(x=x, y=y, names=c(1:length(x)), row.names=\"names\"), col=color.scale(z,c(0,1,1),c(1,1,0),0), main=\"Position of the hidden units\", xlab=\"x1\", ylab=\"x2\", pch=3)");

        // Print hidden units
        for (int i = 0; i < 50; i++) {
            Vector<Double> center = hidden.get(i).getCenter();
            double weight = 1 / ( 1 - Math.exp(-weightsOuter.get(i)));
            hiddenStream.println("draw.circle("+String.format(Locale.ENGLISH, "%.3f", center.get(0))+", "+String.format(Locale.ENGLISH, "%.3f", center.get(1))+", "+String.format(Locale.ENGLISH, "%.3f", Math.abs(0.2 * weight))+", col=\""+(weight > 0?"#FFFF0080":"#00000080")+"\")");
        }

        hiddenStream.println("dev.off()");

        // Print network output to outputStream
        StringBuilder x_R = new StringBuilder("xtot <- c("), y_R = new StringBuilder("ytot <- c("), z_R = new StringBuilder("ztot <- c(");
        for (float z1 = -16; z1 <= 16; z1+=0.1) {
            for (float z2 = -16; z2 <= 16; z2 += 0.1) {
                double r = Math.sqrt(z1*z1+z2*z2), phi = r==0?0:Math.acos(z1 / r);
                x_R.append(String.format(Locale.ENGLISH, "%.3f", r)).append(",");
                y_R.append(String.format(Locale.ENGLISH, "%.3f", phi)).append(",");
                z_R.append(String.format(Locale.ENGLISH, "%.3f", computeOutput(r, phi))).append(",");
            }
        }
        x_R.deleteCharAt(x_R.length()-1).append(")");
        y_R.deleteCharAt(y_R.length()-1).append(")");
        z_R.deleteCharAt(z_R.length()-1).append(")");

        outputStream.println("library(\"plotrix\")");

        outputStream.println(x_R.toString());
        outputStream.println(y_R.toString());
        outputStream.println(z_R.toString());

        outputStream.println("ptot <- data.frame(names=c(1:length(xtot)), x=xtot, y=ytot, row.names=\"names\")\n" +
                "png(\"E:\\\\coding\\\\RBF\\\\output4.3.png\")\n" +
                "plot(ptot, col=color.scale(ztot,c(0,1,1),c(1,1,0),0), main=\"Output of the network\", xlab=\"x1\", ylab=\"x2\", pch=15)\n" +
                "dev.off()");

    }

    private static void learnExamples(double learnrate) {
        for (Point point : dataTotal.keySet()) {
            double output = computeOutput(point);
            for (int i = 0; i < 50; i++) {
                double delta = learnrate * (dataTotal.get(point) - output) * hidden.get(i).getOutput();
                weightsOuter.set(i, weightsOuter.get(i) + delta);
            }
        }
    }

    private static double computeOutput(Point point) {
        return computeOutput(point.getX(), point.getY());
    }

    private static double computeOutput(double x1, double x2) {
        Vector<Double> hiddenInput = new Vector<>(2);
        Vector<Double> hiddenOutput = new Vector<>(50);
        inputNeuronX1.setValue(x1);
        inputNeuronX2.setValue(x2);
        inputNeuronX1.calculate();
        inputNeuronX2.calculate();
        hiddenInput.add(0, inputNeuronX1.getOutput());
        hiddenInput.add(1, inputNeuronX2.getOutput());
        for (int i = 0; i < 50; i++) {
            Neuron rbf = hidden.get(i);
            rbf.setInputs(hiddenInput);
            rbf.calculate();
            hiddenOutput.add(i, rbf.getOutput());
        }
        outputNeuron.setInputs(hiddenOutput);
        outputNeuron.calculate();
        return outputNeuron.getOutput();
    }

    private static double meanSquareError() {
        double error = 0;
        for (Point point : dataTotal.keySet()) {
            double output = computeOutput(point);
            error += (dataTotal.get(point) - output) * (dataTotal.get(point) - output);
        }
        return error;
    }

    private static Point getNegativePoint(int u) {
        double x = 2 + Math.sin(0.2 * u - 8) * Math.sqrt(u + 10), y = -1 + Math.cos(0.2 * u - 8) * Math.sqrt(u + 10);
        double r = Math.sqrt(x*x+y*y);
        return new Point(r, Math.atan2(y,x));
    }

    private static Point getPositivePoint(int u) {
        double x = 2 + Math.sin(0.2 * u + 8) * Math.sqrt(u + 10), y = -1 + Math.cos(0.2 * u + 8) * Math.sqrt(u + 10);
        double r = Math.sqrt(x*x+y*y);
        return new Point(r, Math.atan2(y,x));
    }

}
