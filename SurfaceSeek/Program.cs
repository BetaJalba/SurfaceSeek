// 6 = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0)
using SurfaceSeek;

double[][,] X = new double[4][,];
double[][,] Y = new double[4][,];

// Inputs
X[0] = new double[,] { { 0 }, { 0 } };
X[1] = new double[,] { { 0 }, { 1 } };
X[2] = new double[,] { { 1 }, { 0 } };
X[3] = new double[,] { { 1 }, { 1 } };


// Outputs
Y[0] = new double[,] { { 0 } };
Y[1] = new double[,] { { 1 } };
Y[2] = new double[,] { { 1 } };
Y[3] = new double[,] { { 0 } };


NeuronLayer[] network = 
    { 
        new NeuronLayer(2, 3), // Layer
        new Tanh(), // Activation
        new NeuronLayer(3, 1), // Layer
        new Tanh() // Activation
    };

int epochs = 10000;
double learningRate = 0.1;

for(int i = 0; i < epochs; i++)
{
    var error = 0.0;
    var len = 0;

    for (int j = 0; j < X.Length; j++)
    {
        // Aggiorna input iniziale
        double[,] output = X[j];

        // Propagazione avanti
        foreach (var layer in network)
            output = layer.ForwardPropagation(output);

        // Calcolo errore
        error += Functions.Cost(Y[j], output);

        // Propagazione indietro
        Array.Reverse(network);

        var gradient = Functions.CostPrime(Y[j], output);
        foreach (var layer in network)
            gradient = layer.BackwardPropagation(learningRate, gradient);

        len = X.Length;
        error /= len;

        Console.WriteLine(error);

        // Propagazione avanti
        Array.Reverse(network);
    }
}
