import java.util.Random;

public class Perceptron implements NeuralNetwork {

    int data_size;
    Random rand;
    double[][][] inputs, labels, outputs;
    double[][] weights, bias;

    public Perceptron(double[][][] inputs, double[][][] labels, int seed) {
        this.inputs = inputs;
        this.labels = labels;
        this.data_size = labels.length;
        this.rand = new Random(seed);
        this.weights = new double[inputs[0][0].length][1];
        for(int i=0; i<inputs[0][0].length; i++)
            this.weights[i][0] = rand.nextDouble(1) - 0.5;
        this.bias = new double[][]{{rand.nextDouble(1) - 0.5}};
        this.outputs = forward(inputs);
    }

    public double[][] forward(double[][] input) {
        return NNUtils.sigmoid(Matrix.add(Matrix.dot(input, weights), bias));
    }

    public double[][][] forward(double[][][] inputs) {
        double[][][] outputs = new double[data_size][1][1];
        for(int data=0; data<data_size; data++)
            outputs[data] = forward(inputs[data]);
        return outputs;
    }

    public void train(double lr, int num_epochs) {
        double loss, acc;
        double[][] d_weights, d_bias, logit, error, delta;
        // training
        for(int epoch=0; epoch<num_epochs; epoch++) {
            d_weights = Matrix.fill(2, 1, 0);
            d_bias = Matrix.fill(1, 1, 0);
            for(int data=0; data<data_size; data++) {
                // forward pass
                logit = Matrix.add(Matrix.dot(inputs[data], weights), bias);
                outputs[data] = NNUtils.sigmoid(logit);
                // backpropagation
                error = Matrix.sub(labels[data], outputs[data]);
                delta = Matrix.hadamard(error, NNUtils.sigmoidPrime(logit));
                d_weights = Matrix.add(Matrix.dot(Matrix.transpose(inputs[data]), delta), d_weights);
                d_bias = Matrix.add(delta, d_bias);
            }
            d_weights = Matrix.addScalar(lr / data_size, d_weights);
            d_bias = Matrix.addScalar(lr / data_size, d_bias);
            weights = Matrix.add(weights, d_weights);
            bias = Matrix.add(bias, d_bias);

            // testing
            outputs = forward(inputs);
            loss = NNUtils.mse(labels, outputs);
            acc = NNUtils.accuracy(labels, outputs);
            System.out.println("epoch: " + (epoch+1) + "/" + num_epochs + " loss: " + loss + " acc: " + acc);
        }
    }

    public void showPredictions() {
        for(int data=0; data<data_size; data++)
            Matrix.show(outputs[data]);
    }

    public void showThresholdPredictions() {
        double[][][] binary_out = new double[data_size][1][1];
        for(int data=0; data<data_size; data++) {
            binary_out[data] = Matrix.fill(1, 1, NNUtils.threshold(outputs[data][0][0]));
            Matrix.show(binary_out[data]);
        }
    }

}
