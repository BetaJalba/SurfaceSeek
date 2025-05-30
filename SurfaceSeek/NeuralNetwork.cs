using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Threading.Tasks.Dataflow;
using static System.Runtime.InteropServices.JavaScript.JSType;

namespace SurfaceSeek
{
    public class NeuralNetwork
    {
        public NeuronLayer[] network;

        public NeuralNetwork()
        {

        }

        public NeuralNetwork(int activation, params int[] neuronsPerLayer) // 0 ReLU, 1 Tanh, 2 Sigmoid (final)
        {
            network = new NeuronLayer[(neuronsPerLayer.Length - 1) * 2];

            int count = 0;

            for (int i = 1; i < neuronsPerLayer.Length; i++)
            {
                network[count++] = new NeuronLayer(neuronsPerLayer[i - 1], neuronsPerLayer[i]);

                // If last layer, use sigmoid if requested
                if (i == neuronsPerLayer.Length - 1)
                {
                    network[count++] = new Softmax();   
                }
                else
                {
                    // For hidden layers, default to ReLU or Tanh
                    if (activation == 1)
                        network[count++] = new Tanh();
                    else
                        network[count++] = new ReLU();
                }
            }

            Console.WriteLine("Network Generated!");
        }

        public (double[][], double) Learn(double learningRate, double[][] inputs, double[][] outputs, int batchSize) // Learn function returns the accuracy and predicted value
        {
            var error = 0.0;
            double[][] r = new double[inputs.Length][];

            for (int i = 0; i < inputs.Length; i += batchSize)
            {
                // Batch inputs
                int currentBatchSize = Math.Min(batchSize, inputs.Length - i);

                // Batch inputs so they can be fed into the network
                double[,] batchedInputs = new double[inputs[i].Length, currentBatchSize];
                double[,] batchedOutputs = new double[outputs[i].Length, currentBatchSize];

                for (int b = 0; b < currentBatchSize; b++)
                {
                    for (int f = 0; f < inputs[i].Length; f++)
                        batchedInputs[f, b] = inputs[i + b][f];

                    for (int o = 0; o < outputs[i].Length; o++)
                        batchedOutputs[o, b] = outputs[i + b][o];
                }

                // Aggiorna input iniziale
                double[,] output = batchedInputs;
                double[,] realOutput = batchedOutputs;

                // Propagazione avanti
                foreach (var layer in network)
                {
                    output = layer.ForwardPropagation(output);
                }
                    
                r[i] = deBatch(output);

                // Calcolo errore
                error += Functions.Cost(realOutput, output);

                var gradient = Functions.MatrixSubtraction(output, realOutput); // yPred - yTrue

                // Propagazione indietro
                Array.Reverse(network);

                foreach (var layer in network)
                    if (!(layer is Softmax))
                        gradient = layer.BackwardPropagation(learningRate, gradient);

                // Computa accuracy
                //accuracy = (1 - error) * 100;

                // Reset array
                Array.Reverse(network);
            }

            var len = inputs.Length;
            error /= len;

            return (r, error);
        }

        public (double[], double) Test(double[] inputs, double[] outputs) // Learn function returns the accuracy and predicted value
        {
            var error = 0.0;
            double[] r = new double[outputs.Length];

            // Transpose inputs so they can be fed into the network
            // Aggiorna input iniziale
            double[,] output = transposeInputs(inputs);
            double[,] realOutput = transposeInputs(outputs);

            // Propagazione avanti
            foreach (var layer in network)
                output = layer.ForwardPropagation(output);

            r = deMatrix(output);

            // Calcolo errore
            error += Functions.Cost(realOutput, output);

            return (r, error);
        }

        public double[] Test(double[] inputs) // Learn function returns the accuracy and predicted value
        {
            double[] r = new double[inputs.Length];

            // Transpose inputs so they can be fed into the network
            // Aggiorna input iniziale
            double[,] output = transposeInputs(inputs);

            // Propagazione avanti
            foreach (var layer in network)
                output = layer.ForwardPropagation(output);

            r = deMatrix(output);

            return r;
        }

        #region QOL FUNCTIONS

        double[,] transposeInputs(params double[] inputs) 
        {
            double[,] toTranspose = new double[1,1];
            try
            {
                toTranspose = new double[1, inputs.Length];

                for (int i = 0; i < inputs.Length; i++)
                    toTranspose[0, i] = inputs[i];
            }
            catch (ArgumentException e)
            {
                throw new ArgumentException();
            }

            return Functions.Transpose(toTranspose);
        }

        double[] deMatrix(double[,] matrix)
        {
            double[] r = new double[matrix.GetLength(0)];

            for (int i = 0; i < r.Length; i++)
            {
                r[i] = matrix[i, 0];
            }

            return r;
        }

        double[] deBatch(double[,] matrix)
        {
            double[] r = new double[matrix.GetLength(1)];

            for (int i = 0; i < r.Length; i++)
            {
                r[i] = matrix[0, i];
            }

            return r;
        }

        #endregion
    }


}
