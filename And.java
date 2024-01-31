import java.util.Random;

public class And {

    public static void main(String args[]) {

        Random rand = new Random(43);
        double[][][] input = {{{0, 0}}, {{0, 1}}, {{1, 0}}, {{1, 1}}}; // 1x2
        double[][][] labels = {{{0}}, {{0}}, {{0}}, {{1}}};
        double[][] weights = {
            {rand.nextDouble(1) - 0.5},
            {rand.nextDouble(1) - 0.5},
        }; // 2x1
        double[][] bias = {{rand.nextDouble(1) - 0.5}};
        int data_size = labels.length;

        // hyperparameters
        double lr = 0.5;
        int num_epochs = 20;

        double[][][] outputs = new double[data_size][1][1];
        double[][][] binary_out = new double[data_size][1][1];
        double[][] d_weights, d_bias, logit, error, term_error;
        double loss, acc;

        // training
        for(int epoch=0; epoch<num_epochs; epoch++) {
            d_weights = Matrix.fill(2, 1, 0);
            d_bias = Matrix.fill(1, 1, 0);
            for(int data=0; data<data_size; data++) {
                // forward pass
                logit = Matrix.add(Matrix.dot(input[data], weights), bias);
                outputs[data] = NNUtils.sigmoid(logit);
                // backpropagation
                error = Matrix.sub(labels[data], outputs[data]);
                term_error = Matrix.dot(error, NNUtils.sigmoidPrime(logit));
                d_weights = Matrix.add(Matrix.dot(Matrix.transpose(input[data]), term_error), d_weights);
                d_bias = Matrix.add(term_error, d_bias);
            }
            d_weights = Matrix.addScalar(lr / data_size, d_weights);
            d_bias = Matrix.addScalar(lr / data_size, d_bias);
            weights = Matrix.add(weights, d_weights);
            bias = Matrix.add(bias, d_bias);

            // testing
            loss = NNUtils.mse(labels, outputs);
            acc = NNUtils.accuracy(labels, outputs);
            System.out.println("epoch: " + (epoch+1) + "/" + num_epochs + " loss: " + loss + " acc: " + acc);
        }
        System.out.println();
        for(int data=0; data<data_size; data++) {
            outputs[data] = NNUtils.sigmoid(Matrix.add(Matrix.dot(input[data], weights), bias));
            binary_out[data] = Matrix.fill(1, 1, NNUtils.threshold(outputs[data][0][0]));
            Matrix.showMatrix(outputs[data]);
        }
        System.out.println();
        for(int data=0; data<data_size; data++)
            Matrix.showMatrix(binary_out[data]);

    }

}
