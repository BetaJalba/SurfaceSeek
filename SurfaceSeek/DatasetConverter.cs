using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SurfaceSeek
{
    public static class DatasetConverter
    {
        static Dictionary<string, int> SquadMap = new()
        {
            { "Ancona", 0 },
            { "Ascoli", 1 },
            { "Atalanta", 2 },
            { "Bari", 3 },
            { "Benevento", 4 },
            { "Bologna", 5 },
            { "Brescia", 6 },
            { "Cagliari", 7 },
            { "Carpi", 8 },
            { "Catania", 9 },
            { "Cesena", 10 },
            { "Chievo", 11 },
            { "Como", 12 },
            { "Cremonese", 13 },
            { "Crotone", 14 },
            { "Empoli", 15 },
            { "Fiorentina", 16 },
            { "Foggia", 17 },
            { "Frosinone", 18 },
            { "Genoa", 19 },
            { "Inter", 20 },
            { "Juventus", 21 },
            { "Lazio", 22 },
            { "Lecce", 23 },
            { "Livorno", 24 },
            { "Messina", 25 },
            { "Milan", 26 },
            { "Modena", 27 },
            { "Monza", 28 },
            { "Napoli", 29 },
            { "Novara", 30 },
            { "Padova", 31 },
            { "Palermo", 32 },
            { "Parma", 33 },
            { "Perugia", 34 },
            { "Pescara", 35 },
            { "Piacenza", 36 },
            { "Reggiana", 37 },
            { "Reggina", 38 },
            { "Roma", 39 },
            { "Salernitana", 40 },
            { "Sampdoria", 41 },
            { "Sassuolo", 42 },
            { "Siena", 43 },
            { "Spal", 44 },
            { "Spezia", 45 },
            { "Torino", 46 },
            { "Treviso", 47 },
            { "Udinese", 48 },
            { "Venezia", 49 },
            { "Verona", 50 },
            { "Vicenza", 51 }
        };

        static Dictionary<string, int> ResultsMap = new()
        {
            { "H", 0 },
            { "D", 1 },
            { "A", 2 },
        };

        const int teams = 52;


        public static (double[][] inputs, double[][] outputs) Convert(string path)
        {
            List<List<double>> lastGamesAverage = new List<List<double>>();

            for (int i = 0; i < teams; i++)
            {
                lastGamesAverage.Add(new List<double>());
            }

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

                    double[] home = new double[teams];
                    double[] away = new double[teams];
                    double[] averages = new double[2];

                    // Get Home Team
                    home[SquadMap[split[2]]] = 1;

                    // Get Away Team
                    away[SquadMap[split[3]]] = 1;

                    // Get last 5 games average of home team
                    if (lastGamesAverage[SquadMap[split[2]]].Count == 0)
                        averages[0] = 0;
                    else
                        averages[0] = Normalize(lastGamesAverage[SquadMap[split[2]]].Average());

                    // Get last 5 games average of away team
                    if (lastGamesAverage[SquadMap[split[3]]].Count == 0)
                        averages[1] = 0;
                    else
                        averages[1] = Normalize(lastGamesAverage[SquadMap[split[3]]].Average());
                    
                    // Update average for home
                    if (lastGamesAverage[SquadMap[split[2]]].Count >= 5)
                        lastGamesAverage[SquadMap[split[2]]].RemoveAt(0);
                    lastGamesAverage[SquadMap[split[2]]].Add(double.Parse(split[4]));

                    // Update average for away
                    if (lastGamesAverage[SquadMap[split[3]]].Count >= 5)
                        lastGamesAverage[SquadMap[split[3]]].RemoveAt(0);
                    lastGamesAverage[SquadMap[split[3]]].Add(double.Parse(split[5]));

                    inputs.Add(home.Concat(away).Concat(averages).ToArray()); // Array of size 52 * 2 + 2 = 106

                    double[] win = new double[3];
                    win[ResultsMap[split[6]]] = 1;
                    outputs.Add(win); // Array of size 3
                }
            }

            r.Item1 = inputs.ToArray();
            r.Item2 = outputs.ToArray();

            return r;
        }

        static double Normalize(double val, double min = 0, double max = 5)
        {
            return 2 * ((val - min) / (max - min)) - 1;
        }
    }
}
