using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork1
{
    class Network
    {
        static double sigmoid(double x)
        {
            return 1.0 / (1.0 + Math.Exp(-x)); 
        }
        static double sigmoidDerivative(double x) // производная, для обратного обХОХОХОда (с наступающим)
        {
            return 1.0 / (1.0 + Math.Exp(-x));
        }
        Network(int[] structure)
        {

        }
    }
}
