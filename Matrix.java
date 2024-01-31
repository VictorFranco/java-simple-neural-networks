public class Matrix {

    public static void show(double[][] A) {
        for(int i=0; i<A.length; i++) {
            System.out.print("[");
            for(int j=0; j<A[0].length; j++) {
                System.out.print(A[i][j]);
                if(j+1 != A[0].length) System.out.print(", ");
            }
            System.out.println("]");
        }
    }

    public static double[][] dot(double[][] A, double[][]B) {
        int a_rows = A.length;
        int a_cols = A[0].length;
        int b_rows = B.length;
        int b_cols = B[0].length;
        if(a_cols != b_rows) {
            System.err.println("Error");
            System.exit(-1);
        }
        double[][] result = new double[a_rows][b_cols];
        for(int i=0; i<a_rows; i++)
            for(int j=0; j<b_cols; j++) {
                result[i][j] = 0;
                for(int k=0; k<a_cols; k++)
                    result[i][j] += A[i][k] * B[k][j];
            }
        return result;
    }

    public static double[][] hadamard(double[][] A, double[][]B) {
        int a_rows = A.length;
        int a_cols = A[0].length;
        int b_rows = B.length;
        int b_cols = B[0].length;
        if(a_rows != b_rows && a_cols != b_cols) {
            System.err.println("Error");
            System.exit(-1);
        }
        double[][] result = new double[a_rows][b_cols];
        for(int i=0; i<a_rows; i++)
            for(int j=0; j<b_cols; j++)
                result[i][j] = A[i][j] * B[i][j];
        return result;
    }

    public static double[][] add(double[][] A, double[][] B) {
        if(A.length != B.length && A[0].length != B[0].length) {
            System.err.println("Error");
            System.exit(-1);
        }
        double[][] result = new double[A.length][A[0].length];
        for(int i=0; i<A.length; i++) {
            for(int j=0; j<A[0].length; j++)
                result[i][j] = A[i][j] + B[i][j];
        }
        return result;
    }

    public static double[][] sub(double[][] A, double[][] B) {
        if(A.length != B.length && A[0].length != B[0].length) {
            System.err.println("Error");
            System.exit(-1);
        }
        double[][] result = new double[A.length][A[0].length];
        for(int i=0; i<A.length; i++) {
            for(int j=0; j<A[0].length; j++)
                result[i][j] = A[i][j] - B[i][j];
        }
        return result;
    }

    public static double[][] addScalar(double scalar, double[][] A) {
        double[][] result = new double[A.length][A[0].length];
        for(int i=0; i<A.length; i++) {
            for(int j=0; j<A[0].length; j++)
                result[i][j] = scalar * A[i][j];
        }
        return result;
    }

    public static double[][] transpose(double[][] A) {
        double[][] result = new double[A[0].length][A.length];
        for(int i=0; i<A[0].length; i++) {
            for(int j=0; j<A.length; j++)
                result[i][j] = A[j][i];
        }
        return result;
    }

    public static double[][] fill(int rows, int cols, double num) {
        double[][] result = new double[rows][cols];
        for(int i=0; i<rows; i++)
            for(int j=0; j<cols; j++)
                result[i][j] = num;
        return result;
    }

}
