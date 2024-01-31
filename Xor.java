import java.util.Random;

public class Xor {

    public static void main(String args[]) {

        Random rand = new Random(17);
        double[][][] input = {{{0, 0}}, {{0, 1}}, {{1, 0}}, {{1, 1}}}; // 1x2
        double[][][] labels = {{{0}}, {{1}}, {{1}}, {{0}}};
        double[][][] weights = {
            {{rand.nextDouble(1) - 0.5}, {rand.nextDouble(1) - 0.5}},
            {{rand.nextDouble(1) - 0.5}, {rand.nextDouble(1) - 0.5}},
            {{rand.nextDouble(1) - 0.5}, {rand.nextDouble(1) - 0.5}},
        }; // 2x1
        double[][][] bias = {
            {{rand.nextDouble(1) - 0.5}},
            {{rand.nextDouble(1) - 0.5}},
            {{rand.nextDouble(1) - 0.5}},
        };
        int data_size = labels.length;

        // hyperparameters
        double lr = 10;
        int num_epochs = 1000;

        double[][][] outputs = new double[data_size][1][1];
        double[][][] binary_out = new double[data_size][1][1];
        double[][] error, delta, h_output_prime, hadamard, h_deltas, h_delta_1, h_delta_2;
        double[][][] d_weights = new double[3][1][1];
        double[][][] d_bias = new double[3][1][1];
        double[][][] logit = new double[3][1][1];
        double loss, acc;
        double[][] h_outputs = new double[1][2];

        // training
        for(int epoch=0; epoch<num_epochs; epoch++) {
            d_weights = new double[][][]{
                Matrix.fill(2, 1, 0), Matrix.fill(2, 1, 0), Matrix.fill(2, 1, 0)
            };
            d_bias = new double[][][]{
                Matrix.fill(1, 1, 0), Matrix.fill(1, 1, 0), Matrix.fill(1, 1, 0)
            };
            for(int data=0; data<data_size; data++) {
                // forward pass
                // hidden
                logit[0] = Matrix.add(Matrix.dot(input[data], weights[0]), bias[0]);
                logit[1] = Matrix.add(Matrix.dot(input[data], weights[1]), bias[1]);
                h_outputs = new double[][]{
                    { NNUtils.sigmoid(logit[0])[0][0], NNUtils.sigmoid(logit[1])[0][0] },
                };
                // output
                logit[2] = Matrix.add(Matrix.dot(h_outputs, weights[2]), bias[2]);
                outputs[data] = NNUtils.sigmoid(logit[2]);

                // backpropagation
                // output
                error = Matrix.sub(labels[data], outputs[data]);
                delta = Matrix.hadamard(error, NNUtils.sigmoidPrime(logit[2]));

                d_weights[2] = Matrix.add(Matrix.dot(Matrix.transpose(h_outputs), delta), d_weights[2]);
                d_bias[2] = Matrix.add(delta, d_bias[2]);

                // hidden
                h_output_prime = NNUtils.sigmoidPrime(h_outputs);
                hadamard = Matrix.hadamard(Matrix.transpose(weights[2]), h_output_prime);
                h_deltas = Matrix.dot(delta, hadamard);
                h_delta_1 = new double[][]{{ h_deltas[0][1] }};
                h_delta_2 = new double[][]{{ h_deltas[0][0] }};

                d_weights[1] = Matrix.add(Matrix.dot(Matrix.transpose(input[data]), h_delta_1), d_weights[1]);
                d_bias[1] = Matrix.add(h_delta_1, d_bias[1]);
                d_weights[0] = Matrix.add(Matrix.dot(Matrix.transpose(input[data]), h_delta_2), d_weights[0]);
                d_bias[0] = Matrix.add(h_delta_2, d_bias[0]);
            }
            for(int i=0; i<3; i++) {
                d_weights[i] = Matrix.addScalar(lr / data_size, d_weights[i]);
                d_bias[i] = Matrix.addScalar(lr / data_size, d_bias[i]);
                weights[i] = Matrix.add(weights[i], d_weights[i]);
                bias[i] = Matrix.add(bias[i], d_bias[i]);
            }

            // testing
            loss = NNUtils.mse(labels, outputs);
            acc = NNUtils.accuracy(labels, outputs);
            System.out.println("epoch: " + (epoch+1) + "/" + num_epochs + " loss: " + loss + " acc: " + acc);
        }
        System.out.println();
        for(int data=0; data<data_size; data++) {
            // forward pass
            // hidden
            logit[0] = Matrix.add(Matrix.dot(input[data], weights[0]), bias[0]);
            logit[1] = Matrix.add(Matrix.dot(input[data], weights[1]), bias[1]);
            h_outputs = new double[][]{
                { NNUtils.sigmoid(logit[0])[0][0], NNUtils.sigmoid(logit[1])[0][0] },
            };
            // output
            logit[2] = Matrix.add(Matrix.dot(h_outputs, weights[2]), bias[2]);
            outputs[data] = NNUtils.sigmoid(logit[2]);
            binary_out[data] = Matrix.fill(1, 1, NNUtils.threshold(outputs[data][0][0]));
            Matrix.show(outputs[data]);
        }
        System.out.println();
        for(int data=0; data<data_size; data++)
            Matrix.show(binary_out[data]);

    }

}
