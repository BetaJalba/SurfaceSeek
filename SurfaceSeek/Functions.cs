using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SurfaceSeek
{
    public static class Functions
    {
        public static void PrintArray(double[] array)
        {
            Console.Write("[");
            foreach (var item in array)
            {
                Console.Write($" {item} ");
            }
            Console.WriteLine("]");
        }

        public static void PrintMatrix(double[,] matrix)
        {
            int rows = matrix.GetLength(0);
            int cols = matrix.GetLength(1);

            Console.WriteLine("[");

            for (int i = 0; i < rows; i++)
            {
                Console.Write("  [");
                for (int j = 0; j < cols; j++)
                {
                    Console.Write(matrix[i, j]);
                    if (j < cols - 1)
                        Console.Write(", ");
                }
                Console.WriteLine("]" + (i < rows - 1 ? "," : ""));
            }

            Console.WriteLine("]");
        }

        public static double[][] ConvertToJagged(double[,] matrix)
        {
            int rows = matrix.GetLength(0);
            int cols = matrix.GetLength(1);

            double[][] jagged = new double[rows][];
            for (int i = 0; i < rows; i++)
            {
                jagged[i] = new double[cols];
                for (int j = 0; j < cols; j++)
                {
                    jagged[i][j] = matrix[i, j];
                }
            }

            return jagged;
        }

        public static double Cost(double[,] yTrue, double[,] yPred)
        {
            double epsilon = 1e-12;
            double sum = 0;

            for (int i = 0; i < yTrue.GetLength(0); i++)
            {
                double y = yTrue[i, 0];
                double p = Math.Clamp(yPred[i, 0], epsilon, 1 - epsilon); // Avoid log(0)

                sum += -(y * Math.Log(p) + (1 - y) * Math.Log(1 - p));
            }

            return sum / yTrue.GetLength(0); // Average per sample
        }

        public static double[,] CostPrime(double[,] yTrue, double[,] yPred)
        {
            double epsilon = 1e-12;
            double[,] grad = new double[yTrue.GetLength(0), yTrue.GetLength(1)];

            for (int i = 0; i < yTrue.GetLength(0); i++)
            {
                double y = yTrue[i, 0];
                double p = Math.Clamp(yPred[i, 0], epsilon, 1 - epsilon);

                grad[i, 0] = -(y / p) + (1 - y) / (1 - p);
            }

            return grad;
        }

        /*public static double Cost(double[,] real, double[,] predicted)
        {
            int rows = real.GetLength(0);
            int cols = real.GetLength(1);
            double sum = 0.0;

            for (int i = 0; i < rows; i++)
                for (int j = 0; j < cols; j++)
                    sum += Math.Pow(predicted[i, j] - real[i, j], 2);

            return sum / (rows * cols);
        }

        public static double[,] CostPrime(double[,] real, double[,] predicted)
        {
            int rows = real.GetLength(0);
            int cols = real.GetLength(1);
            double[,] gradient = new double[rows, cols];
            double scale = 2.0 / (rows * cols);

            for (int i = 0; i < rows; i++)
                for (int j = 0; j < cols; j++)
                    gradient[i, j] = scale * (predicted[i, j] - real[i, j]);

            return gradient;
        }*/

        public static double[,] MatrixLinearMultiply(double n, double[,] m)
        {
            int mRows = m.GetLength(0);
            int mCols = m.GetLength(1);

            double[,] result = new double[mRows, mCols];
            for (int i = 0; i < mRows; i++)
                for (int j = 0; j < mCols; j++)
                    result[i, j] = m[i, j] * n;

            return result;
        }

        public static double[,] HadamardProduct(double[,] a, double[,] b)
        {
            int aRows = a.GetLength(0);
            int aCols = a.GetLength(1);
            int bRows = b.GetLength(0);
            int bCols = b.GetLength(1);
            if (aRows != bRows || aCols != bCols)
                throw new ArgumentException("Matrices must have the same dimensions.");
            double[,] result = new double[aRows, aCols];
            for (int i = 0; i < aRows; i++)
                for (int j = 0; j < aCols; j++)
                    result[i, j] = a[i, j] * b[i, j];
            return result;
        }

        public static double[,] MatrixMultiply(double[,] a, double[,] b)
        {
            int aRows = a.GetLength(0);
            int aCols = a.GetLength(1);
            int bRows = b.GetLength(0);
            int bCols = b.GetLength(1);
            if (aCols != bRows)
                throw new ArgumentException("Number of columns in A must match number of rows in B.");
            double[,] result = new double[aRows, bCols];
            for (int i = 0; i < aRows; i++)
                for (int j = 0; j < bCols; j++)
                    for (int k = 0; k < aCols; k++)
                        result[i, j] += a[i, k] * b[k, j];
            return result;
        }

        public static double[,] MatrixSum(double[,] a, double[,] b)
        {
            int aRows = a.GetLength(0);
            int aCols = a.GetLength(1);
            int bRows = b.GetLength(0);
            int bCols = b.GetLength(1);
            if (aRows != bRows || aCols != bCols)
                throw new ArgumentException("Matrices must have the same dimensions.");
            double[,] result = new double[aRows, aCols];
            for (int i = 0; i < aRows; i++)
                for (int j = 0; j < aCols; j++)
                    result[i, j] = a[i, j] + b[i, j];
            return result;
        }

        public static double[,] MatrixSubtraction(double[,] a, double[,] b)
        {
            int aRows = a.GetLength(0);
            int aCols = a.GetLength(1);
            int bRows = b.GetLength(0);
            int bCols = b.GetLength(1);
            if (aRows != bRows || aCols != bCols)
                throw new ArgumentException("Matrices must have the same dimensions.");
            double[,] result = new double[aRows, aCols];
            for (int i = 0; i < aRows; i++)
                for (int j = 0; j < aCols; j++)
                    result[i, j] = a[i, j] - b[i, j];
            return result;
        }

        public static double[,] Transpose(double[,] matrix)
        {
            int w = matrix.GetLength(0);
            int h = matrix.GetLength(1);

            double[,] result = new double[h, w];

            for (int i = 0; i < w; i++)
            {
                for (int j = 0; j < h; j++)
                {
                    result[j, i] = matrix[i, j];
                }
            }

            return result;
        }

        public static byte[,] Transpose(byte[,] matrix)
        {
            int w = matrix.GetLength(0);
            int h = matrix.GetLength(1);

            byte[,] result = new byte[h, w];

            for (int i = 0; i < w; i++)
            {
                for (int j = 0; j < h; j++)
                {
                    result[j, i] = matrix[i, j];
                }
            }

            return result;
        }
    }
}
