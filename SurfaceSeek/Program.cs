// 6 = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0)
using Newtonsoft.Json;
using ScottPlot;
using SurfaceSeek;
using System.Diagnostics;
using static SkiaSharp.HarfBuzz.SKShaper;


var trainingData = DatasetConverter.Convert("Cleaned_Dataset_No_Flight_Days.csv");

int newSize = (int)(trainingData.inputs.Length / 100);

double[][] reducedInputs = new double[newSize][];
double[][] reducedOutputs = new double[newSize][];

for (int i = 0, j = 0; i < trainingData.inputs.Length && j < newSize; i += 100, j++)
{
    reducedInputs[j] = trainingData.inputs[i];
    reducedOutputs[j] = trainingData.outputs[i];
}

NeuralNetwork net;

//if (!File.Exists("weights.json"))
net = new(0, trainingData.inputs[0].Length, 24, 8, 5);
//else
//    net = Newtonsoft.Json.JsonConvert.DeserializeObject<NeuralNetwork>(File.ReadAllText("weights.json"), new JsonSerializerSettings
//    {
//        TypeNameHandling = TypeNameHandling.Auto
//    });

/*var results = net.Learn(0.001, trainingData.inputs, trainingData.outputs);

Functions.PrintArray(trainingData.outputs[1]);
Functions.PrintArray(results.Item1[1]);*/

int epochs = 1000;
double learningRate = 0.0015;

double[] xs = new double[epochs];
double[] ys = new double[epochs];

for (int i = 0; i < epochs; i++)
{
    (double[][] output, double accuracy) results;

    results = net.Learn(learningRate, reducedInputs, reducedOutputs, 32);

    //learningRate = learningRate * Math.Exp(-i / 200.0);

    //for (int j = 0; j < results.output.Length; j++)
    //{
    //    Functions.PrintArray(reducedOutputs[j]);
    //    Functions.PrintArray(results.output[j]);
    //}

    xs[i] = i;
    ys[i] = results.accuracy;
    Console.WriteLine(results.accuracy);
    if (i == epochs - 1) 
        Console.WriteLine("Finito");
}

string file = "weights.json";
File.WriteAllText(file, Newtonsoft.Json.JsonConvert.SerializeObject(net, Newtonsoft.Json.Formatting.Indented, new JsonSerializerSettings()
{
    ReferenceLoopHandling = Newtonsoft.Json.ReferenceLoopHandling.Ignore,
    TypeNameHandling = TypeNameHandling.All
}));

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

net = Newtonsoft.Json.JsonConvert.DeserializeObject<NeuralNetwork>(File.ReadAllText("weights.json"), new JsonSerializerSettings
{
    TypeNameHandling = TypeNameHandling.Auto
});


