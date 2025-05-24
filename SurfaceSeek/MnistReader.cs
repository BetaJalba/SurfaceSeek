using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using System;
using System.IO;

namespace SurfaceSeek
{
    public class MnistReader
    {
        public static byte[,] ReadImages(string filePath, int numberOfImages, int imageSize = 28 * 28)
        {
            byte[,] images = new byte[numberOfImages, imageSize];

            using (var fs = new FileStream(filePath, FileMode.Open))
            using (var br = new BinaryReader(fs))
            {
                int magic = ReverseBytes(br.ReadInt32());
                int numImages = ReverseBytes(br.ReadInt32());
                int numRows = ReverseBytes(br.ReadInt32());
                int numCols = ReverseBytes(br.ReadInt32());

                if (numImages < numberOfImages)
                    throw new Exception("Not enough images in file.");

                for (int i = 0; i < numberOfImages; i++)
                {
                    for (int j = 0; j < imageSize; j++)
                    {
                        images[i, j] = br.ReadByte();
                    }
                }
            }

            return images;
        }

        public static byte[] ReadLabels(string filePath, int numberOfLabels)
        {
            byte[] labels = new byte[numberOfLabels];

            using (var fs = new FileStream(filePath, FileMode.Open))
            using (var br = new BinaryReader(fs))
            {
                int magic = ReverseBytes(br.ReadInt32());
                int numLabels = ReverseBytes(br.ReadInt32());

                if (numLabels < numberOfLabels)
                    throw new Exception("Not enough labels in file.");

                for (int i = 0; i < numberOfLabels; i++)
                {
                    labels[i] = br.ReadByte();
                }
            }

            return labels;
        }

        private static int ReverseBytes(int v)
        {
            byte[] intAsBytes = BitConverter.GetBytes(v);
            Array.Reverse(intAsBytes);
            return BitConverter.ToInt32(intAsBytes, 0);
        }
    }
}
