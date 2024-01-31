public class NNUtils {

    public static double[][] sigmoid(double[][] A) {
        double[][] result = new double[A.length][A[0].length];
        for(int i=0; i<A.length; i++)
            for(int j=0; j<A[0].length; j++)
                result[i][j] = 1 / (1 + Math.exp(-A[i][j]));
        return result;
    }

    public static double[][] sigmoidPrime(double[][] A) {
        double[][] aux = Matrix.fill(A.length, A[0].length, 1);
        double[][] sub = Matrix.sub(aux, sigmoid(A));
        return Matrix.hadamard(sigmoid(A), sub);
    }

    public static double mse(double[][][] labels, double[][][] outputs) {
        int data_size = labels.length;
        double sum = 0;
        for(int data=0; data<data_size; data++)
            sum += Math.pow(labels[data][0][0] - outputs[data][0][0], 2);
        return 1.0 / data_size * sum;
    }

    public static double threshold(double num) {
        return num >= 0.5 ? 1 : 0;
    }

    public static double accuracy(double[][][] labels, double[][][] outputs) {
        int data_size = labels.length;
        double acc = 0;
        for(int data=0; data<data_size; data++)
            acc += labels[data][0][0] == threshold(outputs[data][0][0]) ? 1 : 0;
        acc /= data_size;
        return acc;
    }

}
