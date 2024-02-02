/**
 *
 * @author Franco Hernández Victor Alfonso
 */
public class NOT {

    public static void main(String args[]) {

        // hyperparameters
        double lr = 0.9;
        int num_epochs = 10;
        int seed = 8;

        double[][][] inputs = {{{0}}, {{1}}};
        double[][][] labels = {{{1}}, {{0}}};

        Perceptron perceptron = new Perceptron(inputs, labels, seed);
        perceptron.train(lr, num_epochs);

        System.out.println();
        perceptron.showPredictions();

        System.out.println();
        perceptron.showThresholdPredictions();

    }

}
