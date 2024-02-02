/**
 *
 * @author Franco Hernández Victor Alfonso
 */
interface NeuralNetwork {

    public double[][] forward(double[][] inputs);
    public double[][][] forward(double[][][] inputs);
    public void train(double lr, int num_epochs);
    public void showPredictions();
    public void showThresholdPredictions();

}
