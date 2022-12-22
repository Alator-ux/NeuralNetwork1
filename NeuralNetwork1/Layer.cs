using System;
using System.Collections.Generic;
using System.Linq;
using System.Security.Cryptography.X509Certificates;
using System.Text;
using System.Threading.Tasks;
using NeuralNetwork1.Structs;

namespace NeuralNetwork1
{
    internal class Layer
    {
        public Matrix input;
        public Layer()
        {
        }
        public Matrix goForward(Matrix input)
        {
            this.input = input;
            return compute(input);
        }
        public Matrix goBack(Matrix output, double learningRat)
        {
            return backCompute(output, learningRat);
        }
        protected virtual Matrix compute(Matrix input)
        {
            throw new NotImplementedException("Молодой человек, невиртуальная реализация двумя классами ниже");
        }
        protected virtual Matrix backCompute(Matrix output, double learningRat)
        {
            throw new NotImplementedException("Молодой человек, невиртуальная реализация двумя классами ниже");
        }
    }
    internal class LinearLayer : Layer
    {
        private Matrix weights, biases;
        public LinearLayer(int prevNCount, int nextNCount)
        {
            weights = new Matrix(prevNCount, nextNCount, -0.005, 0.005);
            biases = new Matrix(1, nextNCount, 0); // вектор-строка ВАУ
        }
        protected override Matrix compute(Matrix input)
        {
            var temp = input * weights;
            temp.plusVector(biases);
            return temp;
        }
        protected override Matrix backCompute(Matrix output, double learningRat)
        {
            var temp = output * Matrix.transpose(weights);
            Matrix dBiases = Matrix.rowSums(output);
            Matrix dWeights = Matrix.transpose(input) * output;
            weights -= dWeights * learningRat;
            biases -= dBiases * learningRat;
            return temp;
        }
    }
    internal class SigmoidLayer : Layer
    {
        private double alpha; // какая алфа, если мы в сигмоиде ??77?7??
        public SigmoidLayer(double alpha = 1.0)
        {
            this.alpha = alpha;
        }
        protected override Matrix compute(Matrix input)
        {
            var temp = new Matrix(input, sigmoid);
            return temp;
        }
        private double sigmoid(double x)
        {
            return 1.0 / (1.0 + Math.Exp(-alpha * x));
        }
        private double sigmoidDerivative(double x) // производная, для обратного обХОХОХОда (с наступающим)
        {
            return alpha * sigmoid(x) * (1 - sigmoid(x));
        }
        protected override Matrix backCompute(Matrix output, double learningRat)
        {
            var temp = new Matrix(input.rows, input.columns);
            for (int j = 0; j < input.columns; j++)
            {
                temp[0, j] = sigmoidDerivative(input[0, j]) * output[0, j] * learningRat;
            }
            return temp;
        }
    }
}
