/**
 *
 * @author Franco Hernández Victor Alfonso
 */
public class XNOR {

    public static void main(String args[]) {

        // hyperparameters
        double lr = 10;
        int num_epochs = 711;
        int seed = 85;

        double[][][] inputs = {{{0, 0}}, {{0, 1}}, {{1, 0}}, {{1, 1}}};
        double[][][] labels = {{{1}}, {{0}}, {{0}}, {{1}}};

        MLP mlp = new MLP(inputs, labels, seed);
        mlp.train(lr, num_epochs);

        System.out.println();
        mlp.showPredictions();

        System.out.println();
        mlp.showThresholdPredictions();

    }

}
