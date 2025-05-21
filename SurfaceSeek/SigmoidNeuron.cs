using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;

namespace SurfaceSeek
{
    public class NeuronLayer
    {
        public double[,] learningRate; // Learning rate for the layer
        double[,] inputs;
        double[,] weights; // Inputs to the layer
        double[,] biases; // Weights for each input

        public NeuronLayer(int inputSize, int outputSize, double learningRate)
        {
            Random rand = new();

            weights = new double[inputSize, outputSize];
            biases = new double[outputSize, 1];

            for (int i = 0; i < inputSize; i++)
                for (int j = 0; j < outputSize; j++)
                    weights[i, j] = rand.NextDouble(); // Initialize weights randomly
            for (int i = 0; i < outputSize; i++)
                biases[i, 0] = rand.NextDouble(); // Initialize biases randomly
        }

        public virtual double[,] ForwardPropagation(double[,] inputs)
        {
            this.inputs = Functions.MatrixSum(Functions.MatrixMultiply(weights, inputs), biases);
            return this.inputs;
        }

        public virtual double[,] BackwardPropagation(double[,] learningRate, double[,] outputGradient, double[,] inputs)
        {
            // Backward pass

            // Compute the gradient of the loss function with respect to the weights and biases
            var weightsGradient = Functions.MatrixMultiply(outputGradient, Functions.Transpose(inputs));

            // Compute the gradient of the luss function with respect to the inputs
            var r = Functions.MatrixMultiply(Functions.Transpose(weights), outputGradient); // Compute the gradient of the loss function with respect to the inputs

            // Update weights and biases with new values
            weights = Functions.MatrixSubtraction(Functions.MatrixMultiply(learningRate, weightsGradient), weights);
            biases = Functions.MatrixSubtraction(Functions.MatrixMultiply(learningRate, outputGradient), biases);

            return r;
        }

        public class ActivationLayer : NeuronLayer
        {
            Func<double[,], double[,]> activation;
            Func<(double[,], double[,], double[,]), double[,]> activationPrime

            public ActivationLayer(Func<double[,], double[,]> activation, Func<(double[,], double[,], double[,]), double[,]> activationPrime)
            {
                this.activation = activation;
                this.activationPrime = activationPrime;
            }

            public override double[,] ForwardPropagation(double[,] inputs)
            {
                this.inputs = activation(inputs);
                return (activation(inputs));
            }

            public override double[,] BackwardPropagation(double[,] learningRate, double[,] outputGradient, double[,] inputs)
            {
                return base.BackwardPropagation(learningRate, outputGradient, inputs);
            }
        }
    }

    public static class Functions
    {
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
