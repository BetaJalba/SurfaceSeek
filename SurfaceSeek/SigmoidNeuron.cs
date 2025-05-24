using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;
using static SurfaceSeek.Tanh;
using static System.Reflection.Metadata.BlobBuilder;

namespace SurfaceSeek
{
    public class NeuronLayer
    {
        protected double[,] inputs;
        protected double[,] outputs;
        double[,] weights; // Inputs to the layer
        double[,] biases; // Weights for each input

        public NeuronLayer()
        {
        }

        public NeuronLayer(NeuronLayer self)
        {
            inputs = self.inputs;
            weights = self.weights;
            biases = self.biases;
        }

        public NeuronLayer(int inputSize, int outputSize)
        {
            Random rand = new();

            weights = new double[outputSize, inputSize];
            biases = new double[outputSize, 1];

            for (int i = 0; i < outputSize; i++)
                for (int j = 0; j < inputSize; j++)
                    weights[i, j] = rand.NextDouble(); // Initialize weights randomly
            for (int i = 0; i < outputSize; i++)
                biases[i, 0] = rand.NextDouble(); // Initialize biases randomly
        }

        public virtual double[,] ForwardPropagation(double[,] inputs)
        {
            this.inputs = inputs; // Save last inputs
            outputs = Functions.MatrixSum(Functions.MatrixMultiply(weights, inputs), biases); // Return output
            return outputs;
        }

        public virtual double[,] BackwardPropagation(double learningRate, double[,] outputGradient)
        {
            // Backward pass

            // Compute the gradient of the loss function with respect to the weights and biases
            var weightsGradient = Functions.MatrixMultiply(outputGradient, Functions.Transpose(inputs));

            // Compute the gradient of the luss function with respect to the inputs
            var r = Functions.MatrixMultiply(Functions.Transpose(weights), outputGradient); // Compute the gradient of the loss function with respect to the inputs

            // Update weights and biases with new values
            weights = Functions.MatrixSubtraction(Functions.MatrixLinearMultiply(learningRate, weightsGradient), weights);
            biases = Functions.MatrixSubtraction(Functions.MatrixLinearMultiply(learningRate, outputGradient), biases);

            return r;
        }


    }

    public class ActivationLayer : NeuronLayer
    {
        Func<double[,], double[,]> activation;
        Func<double[,], double[,]> activationPrime;

        public ActivationLayer() : base()
        {
        }

        public ActivationLayer(Func<double[,], double[,]> activation, Func<double[,], double[,]> activationPrime, NeuronLayer self) : base(self)
        {
            this.activation = activation;
            this.activationPrime = activationPrime; // The derivative of activation
        }

        public ActivationLayer(Func<double[,], double[,]> activation, Func<double[,], double[,]> activationPrime) : base()
        {
            this.activation = activation;
            this.activationPrime = activationPrime; // The derivative of activation
        }

        public override double[,] ForwardPropagation(double[,] inputs)
        {
            this.inputs = activation(inputs);
            return this.inputs;
        }

        public override double[,] BackwardPropagation(double learningRate, double[,] outputGradient)
        {
            return Functions.HadamardProduct(outputGradient, activationPrime(inputs));
        }
    }

    public class Tanh : ActivationLayer
    {
        static Func<double[,], double[,]> tanh = x =>
        {
            double[,] r = new double[x.GetLength(0), x.GetLength(1)];

            for (int i = 0; i < x.GetLength(0); i++)
                for (int j = 0; j < x.GetLength(1); j++)
                    r[i, j] = Math.Tanh(x[i, j]);

            return r;
        };

        static Func<double[,], double[,]> tanhPrime = x =>
        {
            double[,] r = new double[x.GetLength(0), x.GetLength(1)];

            for (int i = 0; i < x.GetLength(0); i++)
                for (int j = 0; j < x.GetLength(1); j++)
                {
                    double th = Math.Tanh(x[i, j]);
                    r[i, j] = 1 - th * th; // tanh'(x) = 1 - tanh(x)^2
                }

            return r;
        };

        public Tanh() : base(tanh, tanhPrime)
        {

        }

        public Tanh(NeuronLayer self) : base(tanh, tanhPrime, self)
        {

        }
    }

    public class ReLU : ActivationLayer
    {
        static Func<double[,], double[,]> forward = x =>
        {
            double[,] r = new double[x.GetLength(0), x.GetLength(1)];

            for (int i = 0; i < x.GetLength(0); i++)
                for (int j = 0; j < x.GetLength(1); j++)
                    r[i, j] = x[i, j] > 0 ? x[i, j] : 0.01 * x[i, j]; // Leaky relu

            return r;
        };

        static Func<double[,], double[,]> backward = x =>
        {
            double[,] r = new double[x.GetLength(0), x.GetLength(1)];

            for (int i = 0; i < x.GetLength(0); i++)
                for (int j = 0; j < x.GetLength(1); j++)
                {
                    r[i, j] = x[i, j] > 0 ? 1 : 0.01; // Blocks the gradient if input is < 0
                }

            return r;
        };

        public ReLU() : base(forward, backward)
        {

        }
    }

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

        public static double Cost(double[,] real, double[,] predicted)
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
        }

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
    }
}
