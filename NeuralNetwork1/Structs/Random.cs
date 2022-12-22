using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork1.Structs
{
    internal class IRandom
    {
        private static readonly Random _random = new Random();
        public static int rand(int from, int to)
        {
            return _random.Next(from, to);
        }
    }
    internal class DRandom
    {
        private static readonly Random _random = new Random();
        public static double rand(double from, double to)
        {
            var next = _random.NextDouble();
            return from + (next * (to - from));
        }
    }
}
