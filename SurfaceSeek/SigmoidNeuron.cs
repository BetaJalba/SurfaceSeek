using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SurfaceSeek
{
    public class SigmoidNeuron
    {
        double[] inputs; // Neuron may have multiple inputs
        double[] weights; // Each input has it's own weight
        double bias; // Each neuron has a bias
        double output;

        public SigmoidNeuron(double[] inputs, double[] weights)
        {
            this.inputs = inputs;
            this.weights = weights;
        }

        public double ComputeOutput()
        {
            return 1 / (1 + Math.Exp(-(inputs.Zip(weights, (input, weight) => input * weight).Sum() + bias))); // 1/(1+e^-sum(zn)) where zn := (xn*wn)+b
        }
    }

    public class NeuronLayer
    {
        SigmoidNeuron[] neurons; // Neurons in the layer
        
        public NeuronLayer()
        {

        }

        public double[] ComputeOutputs()
        {
            double[] r = new double[neurons.Length];

            for (int i = 0; i < neurons.Length; i++)
                r[i] = neurons[i].ComputeOutput();

            return r;
        }
    }

    public static class Functions
    {
        public static double Cost(int trainingInputs, double[] outputs, double[] expectedOutputs)
        {
            return (1 / (2 * trainingInputs)) * outputs.Zip(expectedOutputs, (actual, expected) => Math.Pow(actual - expected, 2)).Sum(); // Altered MSE
        }
    }
}
