import java.util.Random;

public class OR {

    public static double[][] forward(double[][] inputs, double[][] weights, double[][] bias) {
        return NNUtils.sigmoid(Matrix.add(Matrix.dot(inputs, weights), bias));
    }

    public static void main(String args[]) {

        // hyperparameters
        double lr = 0.5;
        int num_epochs = 10;
        Random rand = new Random(1);

        double[][][] inputs = {{{0, 0}}, {{0, 1}}, {{1, 0}}, {{1, 1}}}; // 1x2
        double[][][] labels = {{{0}}, {{1}}, {{1}}, {{1}}};
        double[][] weights = {
            {rand.nextDouble(1) - 0.5},
            {rand.nextDouble(1) - 0.5},
        }; // 2x1
        double[][] bias = {{rand.nextDouble(1) - 0.5}};
        int data_size = labels.length;
        double[][][] outputs = new double[data_size][1][1];
        double[][][] binary_out = new double[data_size][1][1];
        double[][] d_weights, d_bias, logit, error, delta;
        double loss, acc;

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
            for(int data=0; data<data_size; data++)
                outputs[data] = forward(inputs[data], weights, bias);
            loss = NNUtils.mse(labels, outputs);
            acc = NNUtils.accuracy(labels, outputs);
            System.out.println("epoch: " + (epoch+1) + "/" + num_epochs + " loss: " + loss + " acc: " + acc);
        }
        System.out.println();
        for(int data=0; data<data_size; data++) {
            outputs[data] = forward(inputs[data], weights, bias);
            binary_out[data] = Matrix.fill(1, 1, NNUtils.threshold(outputs[data][0][0]));
            Matrix.show(outputs[data]);
        }
        System.out.println();
        for(int data=0; data<data_size; data++)
            Matrix.show(binary_out[data]);

    }

}
