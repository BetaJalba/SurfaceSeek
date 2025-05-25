using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static System.Runtime.InteropServices.JavaScript.JSType;

namespace SurfaceSeek
{
    public class NeuralNetwork
    {
        public NeuronLayer[] network;

        public NeuralNetwork()
        {

        }

        public NeuralNetwork(int activation, params int[] neuronsPerLayer) // 0 ReLU, 1 Sigmoid
        {
            network = new NeuronLayer[(neuronsPerLayer.Length - 1) * 2];

            int count = 0;

            for (int i = 1; i < neuronsPerLayer.Length; i++) 
            {
                network[count++] = new NeuronLayer(neuronsPerLayer[i - 1], neuronsPerLayer[i]);

                bool isLastLayer = i == neuronsPerLayer.Length - 1;

                if (!isLastLayer)
                {
                    if (activation == 0)
                        network[count++] = new ReLU();
                    else
                        network[count++] = new Tanh();
                }
                else
                {
                    // Output layer uses sigmoid/tanh for XOR
                    network[count++] = new Tanh(); // or new Sigmoid()
                }
            }

            Console.WriteLine("Network Generated!");
        }

        public (double[][], double) Learn(double learningRate, double[][] inputs, double[][] outputs) // Learn function returns the accuracy and predicted value
        {
            var error = 0.0;
            double[][] r = new double[inputs.Length][];

            for (int i = 0; i < inputs.Length; i++)
            {    

                // Transpose inputs so is can be fed into the network
                double[,] transposedInputs = transposeInputs(inputs[i]);
                double[,] transposedOutputs = transposeInputs(outputs[i]);

                // Aggiorna input iniziale
                double[,] output = transposedInputs;
                double[,] realOutput = transposedOutputs;

                // Propagazione avanti
                foreach (var layer in network)
                    output = layer.ForwardPropagation(output);

                r[i] = deMatrix(output);

                // Calcolo errore
                error += Functions.Cost(realOutput, output);

                // Propagazione indietro
                Array.Reverse(network);

                var gradient = Functions.CostPrime(realOutput, output);
                foreach (var layer in network)
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

        public (double[][], double) Test(double[][] inputs, double[][] outputs) // Learn function returns the accuracy and predicted value
        {
            var error = 0.0;
            double[][] r = new double[inputs.Length][];

            for (int i = 0; i < inputs.Length; i++)
            {

                // Transpose inputs so they can be fed into the network
                double[,] transposedInputs = transposeInputs(inputs[i]);
                double[,] transposedOutputs = transposeInputs(outputs[i]);

                // Aggiorna input iniziale
                double[,] output = transposedInputs;
                double[,] realOutput = transposedOutputs;

                // Propagazione avanti
                foreach (var layer in network)
                    output = layer.ForwardPropagation(output);

                r[i] = deMatrix(output);

                // Calcolo errore
                error += Functions.Cost(realOutput, output);
            }

            var len = inputs.Length;
            error /= len;

            return (r, error);
        }

        #region QOL FUNCTIONS

        double[,] transposeInputs(params double[] inputs) 
        {
            double[,] toTranspose = new double[1, inputs.Length];

            for (int i = 0; i < inputs.Length; i++)
                toTranspose[0, i] = inputs[i];

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

        #endregion
    }


}
