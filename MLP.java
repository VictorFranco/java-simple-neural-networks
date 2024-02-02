import java.util.Random;

/**
 *
 * @author Franco Hernández Victor Alfonso
 */
public class MLP implements NeuralNetwork {

    int data_size;
    Random rand;
    double[][][] inputs, labels, outputs;
    double[][][] weights, biases;

    public MLP(double[][][] inputs, double[][][] labels, int seed) {
        this.inputs = inputs;
        this.labels = labels;
        this.data_size = labels.length;
        this.rand = new Random(seed);
        this.weights = new double[][][]{
            {{rand.nextDouble(1) - 0.5}, {rand.nextDouble(1) - 0.5}},
            {{rand.nextDouble(1) - 0.5}, {rand.nextDouble(1) - 0.5}},
            {{rand.nextDouble(1) - 0.5}, {rand.nextDouble(1) - 0.5}},
        };
        this.biases = new double[][][]{
            {{rand.nextDouble(1) - 0.5}},
            {{rand.nextDouble(1) - 0.5}},
            {{rand.nextDouble(1) - 0.5}},
        };
        this.outputs = forward(inputs);
    }

    public double[][] forward(double[][] input) {
        double[][][] logit = new double[3][1][1];
        // hidden
        logit[0] = Matrix.add(Matrix.dot(input, weights[0]), biases[0]);
        logit[1] = Matrix.add(Matrix.dot(input, weights[1]), biases[1]);
        double[][] h_outputs = {
            { NNUtils.sigmoid(logit[0])[0][0], NNUtils.sigmoid(logit[1])[0][0] },
        };
        // output
        logit[2] = Matrix.add(Matrix.dot(h_outputs, weights[2]), biases[2]);
        return NNUtils.sigmoid(logit[2]);
    }

    public double[][][] forward(double[][][] inputs) {
        double[][][] outputs = new double[data_size][1][1];
        for(int data=0; data<data_size; data++)
            outputs[data] = forward(inputs[data]);
        return outputs;
    }

    public void train(double lr, int num_epochs) {
        double loss, acc;
        double[][] error, delta, h_output_prime, hadamard, h_deltas, h_delta_1, h_delta_2, h_outputs;
        double[][][] d_weights, d_biases;
        double[][][] logit = new double[3][1][1];

        // training
        for(int epoch=0; epoch<num_epochs; epoch++) {
            d_weights = new double[][][]{
                Matrix.fill(2, 1, 0), Matrix.fill(2, 1, 0), Matrix.fill(2, 1, 0)
            };
            d_biases = new double[][][]{
                Matrix.fill(1, 1, 0), Matrix.fill(1, 1, 0), Matrix.fill(1, 1, 0)
            };
            for(int data=0; data<data_size; data++) {
                // forward pass
                // hidden
                logit[0] = Matrix.add(Matrix.dot(inputs[data], weights[0]), biases[0]);
                logit[1] = Matrix.add(Matrix.dot(inputs[data], weights[1]), biases[1]);
                h_outputs = new double[][]{
                    { NNUtils.sigmoid(logit[0])[0][0], NNUtils.sigmoid(logit[1])[0][0] },
                };
                // output
                logit[2] = Matrix.add(Matrix.dot(h_outputs, weights[2]), biases[2]);
                outputs[data] = NNUtils.sigmoid(logit[2]);

                // backpropagation
                // output
                error = Matrix.sub(labels[data], outputs[data]);
                delta = Matrix.hadamard(error, NNUtils.sigmoidPrime(logit[2]));

                // hidden
                h_output_prime = NNUtils.sigmoidPrime(h_outputs);
                hadamard = Matrix.hadamard(Matrix.transpose(weights[2]), h_output_prime);
                h_deltas = Matrix.dot(delta, hadamard);
                h_delta_1 = new double[][]{{ h_deltas[0][1] }};
                h_delta_2 = new double[][]{{ h_deltas[0][0] }};

                d_weights[2] = Matrix.add(Matrix.dot(Matrix.transpose(h_outputs), delta), d_weights[2]);
                d_biases[2] = Matrix.add(delta, d_biases[2]);
                d_weights[1] = Matrix.add(Matrix.dot(Matrix.transpose(inputs[data]), h_delta_1), d_weights[1]);
                d_biases[1] = Matrix.add(h_delta_1, d_biases[1]);
                d_weights[0] = Matrix.add(Matrix.dot(Matrix.transpose(inputs[data]), h_delta_2), d_weights[0]);
                d_biases[0] = Matrix.add(h_delta_2, d_biases[0]);
            }
            for(int i=0; i<3; i++) {
                d_weights[i] = Matrix.addScalar(lr / data_size, d_weights[i]);
                d_biases[i] = Matrix.addScalar(lr / data_size, d_biases[i]);
                weights[i] = Matrix.add(weights[i], d_weights[i]);
                biases[i] = Matrix.add(biases[i], d_biases[i]);
            }

            // testing
            if(epoch % 10 == 0) {
                outputs = forward(inputs);
                loss = NNUtils.mse(labels, outputs);
                acc = NNUtils.accuracy(labels, outputs);
                System.out.println("epoch: " + (epoch+1) + "/" + num_epochs + " loss: " + loss + " acc: " + acc);
            }
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
