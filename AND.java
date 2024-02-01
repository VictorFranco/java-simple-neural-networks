public class AND {

    public static void main(String args[]) {

        // hyperparameters
        double lr = 0.5;
        int num_epochs = 20;
        int seed = 43;

        double[][][] inputs = {{{0, 0}}, {{0, 1}}, {{1, 0}}, {{1, 1}}};
        double[][][] labels = {{{0}}, {{0}}, {{0}}, {{1}}};

        Perceptron perceptron = new Perceptron(inputs, labels, seed);
        perceptron.train(lr, num_epochs);

        System.out.println();
        perceptron.showPredictions();

        System.out.println();
        perceptron.showThresholdPredictions();

    }

}
