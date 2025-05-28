using System;
using System.Collections.Generic;
using System.Linq;
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
            { "Morning", 2 },
            { "Early_Morning", 2 }
        };

        static Dictionary<string, int> stopsMap = new()
        {
            { "zero", 0 },
            { "one", 1 },
            { "two", 2 },
            { "three", 2 }
        };

        static Dictionary<string, int> airlineMap = new()
        {
            { "Indigo", 0 },
            { "Air India", 1 },
            { "SpiceJet", 2 },
            { "Vistara", 3 },
            { "GO_FIRST", 4 },
            { "AirAsia India", 5 },
            { "Akasa Air", 6 },
            { "Alliance Air", 7 }
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

                    double[] airline = new double[1];
                    double[] source_city = new double[cities];
                    double[] times = new double[3];
                    double[] destionation_city = new double[cities];
                    double[] duration = new double[1];

                    airline[0] = airlineMap[split[0]];

                    source_city[cityIdMap[split[1]]] = 1;

                    times[0] = timeMap[split[2]];
                    times[1] = stopsMap[split[3]];
                    times[2] = timeMap[split[4]];

                    destionation_city[cityIdMap[split[5]]] = 1;

                    duration[0] = double.Parse(split[6]);

                    inputs.Add(airline.Concat(source_city)
                        .Concat(times)
                        .Concat(destionation_city)
                        .Concat(duration)
                        .ToArray());

                    double[] price = new double[1];
                    price[0] = double.Parse(split[7]);
                    outputs.Add(price);
                }
            }

            r.Item1 = inputs.ToArray();
            r.Item2 = outputs.ToArray();

            return r;
        }
    }
}
