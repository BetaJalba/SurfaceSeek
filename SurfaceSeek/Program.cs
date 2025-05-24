// 6 = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0)
using ScottPlot;
using SurfaceSeek;
using System.Diagnostics;

double[][] X = new double[4][];
double[][] Y = new double[4][];

// Inputs
X[0] = new double[] { 0, 0 };
X[1] = new double[] { 0, 1 };
X[2] = new double[] { 1, 0 };
X[3] = new double[] { 1, 1 };


// Outputs
Y[0] = new double[] { 0 };
Y[1] = new double[] { 1 };
Y[2] = new double[] { 1 };
Y[3] = new double[] { 0 };



NeuralNetwork net = new(1, 2, 3, 1);

int epochs = 10000;
double learningRate = 0.01;

double[] xs = new double[epochs];
double[] ys = new double[epochs];


for (int i = 0; i < epochs; i++)
{
    (double[][] output, double accuracy) results;

    
    results = net.Learn(learningRate, X, Y);

    xs[i] = i;
    ys[i] = results.accuracy;
    Console.WriteLine(results.accuracy);
}

// Plot
ScottPlot.Plot myPlot = new();

myPlot.Add.Scatter(xs, ys);
//myPlot.Axes.SetLimits(0, epochs, 0, 120);

string filePath = "error.png";
myPlot.SavePng(filePath, 400, 300);

Process.Start(new ProcessStartInfo
{
    FileName = filePath,
    UseShellExecute = true // Required for opening files in .NET Core
});



for (int i = 0; i < 4; i++)
{
    double[][] res;

    res = net.Learn(learningRate, X, Y).Item1;

    Functions.PrintArray(X[i]);
    Functions.PrintArray(res[i]);
    Console.WriteLine(Y[i][0]);
}

