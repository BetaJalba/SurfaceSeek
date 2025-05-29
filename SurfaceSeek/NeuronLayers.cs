using Newtonsoft.Json;
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
    #region LAYER
    public class NeuronLayer
    {
        protected double[,] inputs;
        protected double[,] outputs;
        public double[,] weights; // Inputs to the layer
        public double[,] biases; // Weights for each input

        public NeuronLayer()
        {

        }

        public NeuronLayer(int inputSize, int outputSize)
        {
            Random rand = new();
            weights = new double[outputSize, inputSize];
            biases = new double[outputSize, 1];

            double stddev = Math.Sqrt(2.0 / inputSize); // He initialization stddev

            for (int i = 0; i < outputSize; i++)
            {
                for (int j = 0; j < inputSize; j++)
                {
                    // Box-Muller transform to generate normal-distributed random numbers
                    double u1 = 1.0 - rand.NextDouble();
                    double u2 = 1.0 - rand.NextDouble();
                    double randStdNormal = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2);

                    weights[i, j] = randStdNormal * stddev;
                }
                biases[i, 0] = 0.0; // Initialize biases to zero
            }
        }

        public virtual double[,] ForwardPropagation(double[,] inputs)
        {
            this.inputs = inputs; // Save last inputs

            outputs = Functions.MatrixBiasSum(Functions.MatrixMultiply(weights, inputs), biases); // Return output
            return outputs;
        }

        public virtual double[,] BackwardPropagation(double learningRate, double[,] outputGradient)
        {
            // Backward pass

            // Compute the gradient of the loss function with respect to the weights and biases
            var weightsGradient = Functions.MatrixMultiply(outputGradient, Functions.Transpose(inputs));

            weightsGradient = Functions.Clip(weightsGradient, -1.0, 1.0);
            //outputGradient = Functions.Clip(outputGradient, -1.0, 1.0);

            // Compute the gradient of the luss function with respect to the inputs
            var inputGradient = Functions.MatrixMultiply(Functions.Transpose(weights), outputGradient); // Compute the gradient of the loss function with respect to the inputs

            // Update weights and biases with new values
            weights = Functions.MatrixSubtraction(weights, Functions.MatrixLinearMultiply(learningRate, weightsGradient));

            var biasGradient = new double[biases.GetLength(0), 1];
            for (int i = 0; i < outputGradient.GetLength(0); i++)
            {
                double sum = 0.0;
                for (int j = 0; j < outputGradient.GetLength(1); j++)
                {
                    sum += outputGradient[i, j];
                }
                biasGradient[i, 0] = sum;
            }
            biasGradient = Functions.Clip(biasGradient, -1.0, 1.0);

            biases = Functions.MatrixSubtraction(biases, Functions.MatrixLinearMultiply(learningRate, biasGradient));

            return inputGradient;
        }
    }

    #endregion

    #region ACTIVATION

    public class ActivationLayer : NeuronLayer
    {
        public Func<double[,], double[,]> activation;
        public Func<double[,], double[,]> activationPrime;

        public ActivationLayer() : base()
        {

        }

        public ActivationLayer(Func<double[,], double[,]> activation, Func<double[,], double[,]> activationPrime) : base()
        {
            this.activation = activation;
            this.activationPrime = activationPrime; // The derivative of activation
        }

        public override double[,] ForwardPropagation(double[,] inputs)
        {
            this.inputs = inputs;
            return activation(inputs);
        }

        public override double[,] BackwardPropagation(double learningRate, double[,] outputGradient)
        {
            return Functions.HadamardProduct(outputGradient, activationPrime(inputs));
        }
    }

    public class Tanh : ActivationLayer
    {
        public Tanh() : base(
            x => 
            {
                double[,] r = new double[x.GetLength(0), x.GetLength(1)];

                for (int i = 0; i < x.GetLength(0); i++)
                    for (int j = 0; j < x.GetLength(1); j++)
                        r[i, j] = Math.Tanh(x[i, j]);

                return r;
            }, 
            x => 
            {
                double[,] r = new double[x.GetLength(0), x.GetLength(1)];

                for (int i = 0; i < x.GetLength(0); i++)
                    for (int j = 0; j < x.GetLength(1); j++)
                    {
                        double th = Math.Tanh(x[i, j]);
                        r[i, j] = 1 - th * th; // tanh'(x) = 1 - tanh(x)^2
                    }

                return r;
            }){}
    }

    public class ReLU : ActivationLayer
    {
        public ReLU() : base(
            x =>
            {
                double[,] r = new double[x.GetLength(0), x.GetLength(1)];

                for (int i = 0; i < x.GetLength(0); i++)
                    for (int j = 0; j < x.GetLength(1); j++)
                        r[i, j] = x[i, j] > 0 ? x[i, j] : 0.01 * x[i, j]; // Leaky relu

                return r;
            },
            x =>
            {
                double[,] r = new double[x.GetLength(0), x.GetLength(1)];

                for (int i = 0; i < x.GetLength(0); i++)
                    for (int j = 0; j < x.GetLength(1); j++)
                    {
                        r[i, j] = x[i, j] > 0 ? 1 : 0.01; // Blocks the gradient if input is < 0
                    }

                return r;
            })
        { }
    }

    public class Sigmoid : ActivationLayer
    {
        public Sigmoid() : base(
            x =>
            {
                double[,] r = new double[x.GetLength(0), x.GetLength(1)];
                for (int i = 0; i < x.GetLength(0); i++)
                    for (int j = 0; j < x.GetLength(1); j++)
                    {
                        r[i, j] = 1.0 / (1.0 + Math.Exp(-x[i, j]));
                        if (r[i, j] > 1)
                            Console.WriteLine("a");
                    }
                        
                return r;
            },
            x =>
            {
                double[,] r = new double[x.GetLength(0), x.GetLength(1)];
                for (int i = 0; i < x.GetLength(0); i++)
                    for (int j = 0; j < x.GetLength(1); j++)
                    {
                        double sig = 1.0 / (1.0 + Math.Exp(-x[i, j]));
                        r[i, j] = sig * (1 - sig);
                    }
                return r;
            }
        )
        { }
    }

    #endregion
}
