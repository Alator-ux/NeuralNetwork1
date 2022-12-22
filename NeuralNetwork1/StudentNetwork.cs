using System;
using System.Diagnostics;
using System.IO;
using System.Linq;
using NeuralNetwork1.Structs;

namespace NeuralNetwork1
{
    public class StudentNetwork : BaseNetwork
    {
        private DoublyLinkedList<Layer> _layers;
        private double learningRat = 0.5; // обучающаяся крыса
        // Часики-то тикают. Вон, все твои друзья уже обучили нейронку, а я когда распознования дождусь?
        public Stopwatch stopWatch = new Stopwatch();
        public StudentNetwork(int[] structure)
        {
            _layers = new DoublyLinkedList<Layer>();
            for(int i = 0; i < structure.Length - 1; i++)
            {
                var linear = new LinearLayer(structure[i], structure[i+1]);
                var sigmoid = new SigmoidLayer();
                _layers.Add(linear);
                _layers.Add(sigmoid);
            }
        }
        private Matrix goForward(Matrix input)
        {
            var res = new Matrix(input);
            var layer = _layers.head;
            while(layer != null)
            {
                res = layer.Data.goForward(res);
                layer = layer.Next;
            }
            return res;
        }
        private void goBack(Matrix output)
        {
            Matrix res = new Matrix(output);
            var layer = _layers.tail;
            while(layer != null)
            {
                res = layer.Data.goBack(res, learningRat);
                layer = layer.Previous;
            }
        }

        public override int Train(Sample sample, double acceptableError, bool parallel)
        {
            int iters = 0;
            double error = double.PositiveInfinity;
            while (error > acceptableError)
            {
                error = trainSample(sample, 1);
                ++iters;
            }

            return iters;
        }
        private double trainSample(Sample sample, int totalCount)
        {
            var pred = Compute(sample.input);
            sample.ProcessPrediction(pred);
            var error = sample.error;
            double loss = sample.EstimatedError() / sample.Output.Length; // IS THIS
            goBack((2.0 / sample.Output.Length / totalCount) * new Matrix(error));
            return loss;
        }
        public override double TrainOnDataSet(SamplesSet samplesSet, int epochsCount, double acceptableError, bool parallel)
        {
            //  Текущий счётчик эпох
            int epoch_to_run = 0;
            double error = double.PositiveInfinity;

#if DEBUG
            StreamWriter errorsFile = File.CreateText("errors.csv");
#endif

            stopWatch.Restart();

            while (epoch_to_run < epochsCount && error > acceptableError)
            {
                epoch_to_run++;
                error = 0.0;
                for(int i = 0; i < samplesSet.Count; i++)
                {
                    error += trainSample(samplesSet[i], samplesSet.Count);
                }
#if DEBUG
                errorsFile.WriteLine(error);
#endif
                OnTrainProgress((epoch_to_run * 1.0) / epochsCount, error, stopWatch.Elapsed);
            }

#if DEBUG
            errorsFile.Close();
#endif

            OnTrainProgress(1.0, error, stopWatch.Elapsed);

            stopWatch.Stop();

            return error;
        }

        protected override double[] Compute(double[] input)
        {
            Matrix mat = new Matrix(input);
            var res = goForward(mat);
            return Matrix.toDoubleArray(res);
        }
    }
}