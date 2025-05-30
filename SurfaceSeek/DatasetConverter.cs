using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection.Metadata.Ecma335;
using System.Text;
using System.Threading.Tasks;

namespace SurfaceSeek
{
    public static class DatasetConverter
    {
        static Dictionary<string, int> cityIdMap = new Dictionary<string, int>
        {
            { "Mumbai", 0 },
            { "Delhi", 1 },
            { "Bangalore", 2 },
            { "Kolkata", 3 },
            { "Hyderabad", 4 },
            { "Chennai", 5 }
        };

        static Dictionary<string, int> timeMap = new()
        {
            { "Afternoon", 0 },
            { "Evening", 1 },
            { "Night", 2 },
            { "Morning", 3 },
            { "Early_Morning", 4 },
            { "Late_Night", 5 }
        };

        static Dictionary<string, int> stopsMap = new()
        {
            { "zero", 0 },
            { "one", 1 },
            { "two_or_more", 2 }
        };

        static Dictionary<string, int> airlineMap = new()
        {
            { "Indigo", 0 },
            { "Air_India", 1 },
            { "SpiceJet", 2 },
            { "Vistara", 3 },
            { "GO_FIRST", 4 },
            { "AirAsia", 5 },
            { "Akasa Air", 6 },
            { "Alliance Air", 7 }
        };

        static Dictionary<string, int> pricingMap = new()
        {
            { "Economy", 0 },
            { "Business", 1 }
        };

        const int cities = 6;


        public static (double[][] inputs, double[][] outputs) Convert(string path)
        {
            (double[][], double[][]) r;
            List<double[]> inputs = new List<double[]>();
            List<double[]> outputs = new List<double[]>();

            using (StreamReader sr = new StreamReader(path))
            {
                sr.ReadLine(); // Ignore first line
                while (!sr.EndOfStream)
                {
                    var line = sr.ReadLine();
                    var split = line.Split(',');

                    double[] airline = new double[airlineMap.Count];
                    double[] source_city = new double[cities];
                    double[] departure_time = new double[timeMap.Count];
                    double[] stops = new double[stopsMap.Count];
                    double[] arrival_time = new double[timeMap.Count];
                    double[] destionation_city = new double[cities];
                    double[] classes = new double[pricingMap.Count];
                    double[] duration = new double[1];

                    airline[airlineMap[split[0]]] = 1;

                    source_city[cityIdMap[split[1]]] = 1;

                    departure_time[timeMap[split[2]]] = 1;
                    stops[stopsMap[split[3]]] = 1;
                    arrival_time[timeMap[split[4]]] = 1;

                    destionation_city[cityIdMap[split[5]]] = 1;

                    classes[pricingMap[split[6]]] = 1;

                    duration[0] = double.Parse(split[7]) / 100;

                    inputs.Add(airline.Concat(source_city)
                        .Concat(departure_time)
                        .Concat(stops)
                        .Concat(arrival_time)
                        .Concat(destionation_city)
                        .Concat(classes)
                        .Concat(duration)
                        .ToArray());

                    double[] price = new double[5];
                    //price[] = double.Parse(split[8]) / 100;

                    var cost = double.Parse(split[8]);
                    if (cost < 5000)
                        price[0] = 1;
                    else if (cost < 10000)
                        price[1] = 1;
                    else if (cost < 15000)
                        price[2] = 1;
                    else if (cost < 20000)
                        price[3] = 1;
                    else if (cost < 25000)
                        price[4] = 1;
                    outputs.Add(price);
                }
            }

            r.Item1 = inputs.ToArray();
            r.Item2 = outputs.ToArray();

            return r;
        }

        public static double[] ConvertString(string line)
        {
            List<double[]> inputs = new ();

            var split = line.Split(',');

            double[] airline = new double[airlineMap.Count];
            double[] source_city = new double[cities];
            double[] departure_time = new double[timeMap.Count];
            double[] stops = new double[stopsMap.Count];
            double[] arrival_time = new double[timeMap.Count];
            double[] destionation_city = new double[cities];
            double[] classes = new double[pricingMap.Count];
            double[] duration = new double[1];

            airline[airlineMap[split[0]]] = 1;

            source_city[cityIdMap[split[1]]] = 1;

            departure_time[timeMap[split[2]]] = 1;
            stops[stopsMap[split[3]]] = 1;
            arrival_time[timeMap[split[4]]] = 1;

            destionation_city[cityIdMap[split[5]]] = 1;

            classes[pricingMap[split[6]]] = 1;

            duration[0] = double.Parse(split[7]) / 100;

            inputs.Add(airline.Concat(source_city)
                .Concat(departure_time)
                .Concat(stops)
                .Concat(arrival_time)
                .Concat(destionation_city)
                .Concat(classes)
                .Concat(duration)
                .ToArray());

            return inputs.ToArray()[0];
        }
    }
}
