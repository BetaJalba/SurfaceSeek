// 6 = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0)
using ScottPlot;
using SurfaceSeek;
using System.Diagnostics;

var images = MnistReader.ReadImagesAsDouble("train-images.idx3-ubyte", 1000);
var labels = MnistReader.ReadLabelsAsDouble("train-labels.idx1-ubyte", 1000);

// Display first pixel of first image and label
Console.WriteLine(images[0, 0]); // pixel value (0-255)
Console.WriteLine(labels[0]);    // label (0-9)


NeuralNetwork net = new(0, 784, 256, 1);

int epochs = 4;
double learningRate = 0.001;

double[] xs = new double[epochs];
double[] ys = new double[epochs];


for (int i = 0; i < epochs; i++)
{
    (double[][] output, double accuracy) results;

    
    results = net.Learn(learningRate, Functions.ConvertToJagged(images), labels);

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



