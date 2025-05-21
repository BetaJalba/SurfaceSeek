// 6 = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0)
using SurfaceSeek;
using Numpy;

double[][] X = new double[4][];
double[] Y = new double[4] { 0, 1, 1, 0 };

X[0] = new double[] { 0, 0 };
X[1] = new double[] { 0, 1 };
X[2] = new double[] { 1, 0 };
X[3] = new double[] { 1, 1 };

np.array(X);
np.array(Y);

var X1 = np.reshape(X, (4, 2, 1));
var Y1 = np.reshape(X, (4, 1, 1));


NeuronLayer[] network = 
    { 
        new NeuronLayer(2, 3), 
        new Tanh(), 
        new NeuronLayer(3, 1), 
        new Tanh() 
    };

int epochs = 10000;
double learningRate = 0.1;

for(int i = 0; i < epochs; i++)
{
    var error = 0;

    //foreach(var items in )
    //{
    //    var output = items.First;

    //    foreach (var layer in network)
    //        output = layer.ForwardPropagation(output);

    //    error += Functions.Cost(Y, out)
    //}

}
