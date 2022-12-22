using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using Accord.Statistics.Testing;

namespace NeuralNetwork1.Structs
{
    internal class Matrix
    {
        public double[,] data;
        public Matrix(double[,] data)
        {
            this.data = data;
        }
        public Matrix(int rSize, int cSize)
        {
            this.data = new double[rSize, cSize];
        }
        public Matrix(double[] data)
        {
            this.data = new double[1, data.Length];
            for (int i = 0; i < data.Length; i++)
            {
                this.data[0, i] = data[i];
            }
        }
        /// <summary>
        /// Заполняет матрицу случайными числами
        /// </summary>
        /// <param name="rSize"></param>
        /// <param name="cSize"></param>
        /// <param name="rFrom">Левая граница</param>
        /// <param name="rTo">Правая граница</param>
        public Matrix(int rSize, int cSize, double rFrom, double rTo)
        {
            this.data = new double[rSize, cSize];
            for(int i = 0; i < rSize; i++)
            {
                for(int j = 0; j < cSize; j++)
                {
                    this.data[i,j] = DRandom.rand(rFrom, rTo);
                }
            }
        }
        /// <summary>
        /// Заполняет матрицу фиксированным числом
        /// </summary>
        /// <param name="rSize"></param>
        /// <param name="cSize"></param>
        /// <param name="n"></param>
        public Matrix(int rSize, int cSize, double n)
        {
            this.data = new double[rSize, cSize];
            for (int i = 0; i < rSize; i++)
            {
                for (int j = 0; j < cSize; j++)
                {
                    this.data[i, j] = n;
                }
            }
        }
        public Matrix(Matrix other)
        {
            this.data = new double[other.rows, other.columns];
            Parallel.For(0, other.rows, i =>
            {
                for (int j = 0; j < other.columns; j++)
                {
                    this.data[i, j] = other[i, j];
                }
            });
        }
        public Matrix(Matrix other, Func<double, double> f)
        {
            this.data = new double[other.rows, other.columns];
            Parallel.For(0, other.rows, i =>
            {
                for (int j = 0; j < other.columns; j++)
                {
                    this.data[i, j] = f(other[i, j]);
                }
            });
        }
        public double this[int row, int column]
        {
            set => data[row,column] = value;
            get => data[row,column];
        }
        public Matrix this[int row]
        {
            get
            {
                var res = new Matrix(1, columns);
                for (int j = 0; j < columns; j++)
                {
                    res[0, j] = this[row, j];
                }
                return res;
            }
        }
        public int rows
        {
            get => data.GetLength(0);
        }
        public int columns
        {
            get => data.GetLength(1);
        }
        public void plusVector(Matrix vector)
        {
            if (columns != vector.columns)
            {
                throw new ArgumentException(
                    "Матрица и вектор не могут быть сложены, для друг друга они слишком сложны...");
            }
            Parallel.For(0, rows, i =>
            {
                for (int j = 0; j < columns; j++)
                {
                    this[i, j] += vector[0, j];
                }
            });
        }
        public static double[] toDoubleArray(Matrix matrix)
        {
            double[] res = new double[matrix.rows * matrix.columns];
            Parallel.For(0, matrix.rows, i =>
            {
                for (int j = 0; j < matrix.columns; j++)
                {
                    res[i * matrix.columns + j] = matrix[i, j];
                }
            });
            return res;
        }
        public static Matrix transpose(Matrix matrix)
        {
            var res = new Matrix(matrix.columns, matrix.rows);
            Parallel.For(0, matrix.rows, i =>
            {
                for (int j = 0; j < matrix.columns; j++)
                {
                    res[j, i] = matrix[i, j];
                }
            });
            return res;
        }
        public static Matrix rowSums(Matrix matrix)
        {
            var res = new Matrix(matrix.rows, matrix.columns);
            Parallel.For(0, matrix.columns, i =>
            {
                for (int j = 0; j < matrix.rows; j++)
                {
                    res[0, i] += matrix[j, i];
                }
            });
            return res;
        }
        public static Matrix operator *(Matrix first, Matrix second)
        {
            if(first == null || second == null || first.columns != second.rows)
            {
                throw new ArgumentException("Матрицы не могут быть перемножены");
            }
            var res = new Matrix(first.rows, second.columns);
            Parallel.For(0, first.rows, i =>
            {
                for (int j = 0; j < second.columns; j++)
                {
                    for (int k = 0; k < first.columns; k++)
                    {
                        res[i, j] += first[i, k] * second[k, j];
                    }
                }
            });
            return res;
        }
        public static Matrix operator *(Matrix matrix, double n)
        {
            var res = new Matrix(matrix.rows, matrix.columns);
            Parallel.For(0, matrix.rows, i =>
            {
                for (int j = 0; j < matrix.columns; j++)
                {
                    res[i, j] = matrix[i, j] * n;
                }
            });
            return res;
        }
        public static Matrix operator *(double n, Matrix matrix)
        {
            return matrix * n;
        }
        public static Matrix operator +(Matrix first, Matrix second)
        {
            if (first.rows != second.rows || first.columns != second.columns)
            {
                throw new ArgumentException("Матрицы не могут быть сложены");
            }
            var res = new Matrix(first.rows, first.columns);
            Parallel.For(0, first.rows, i =>
            {
                for (int j = 0; j < second.columns; j++)
                {
                    res[i, j] = first[i, j] + second[i, j];
                }
            });
            return res;
        }
        public static Matrix operator -(Matrix first, Matrix second)
        {
            if (first.rows != second.rows || first.columns != second.columns)
            {
                throw new ArgumentException("Матрицы не могут быть сложены");
            }
            var res = new Matrix(first.rows, first.columns);
            Parallel.For(0, first.rows, i =>
            {
                for (int j = 0; j < second.columns; j++)
                {
                    res[i, j] = first[i, j] - second[i, j];
                }
            });
            return res;
        }
        public static Matrix operator -(Matrix matrix)
        {
            return matrix * -1.0;
        }
        public static Matrix operator /(Matrix matrix, double n)
        {
            Matrix res = new Matrix(matrix.rows, matrix.columns);
            Parallel.For(0, matrix.rows, i =>
            {
                for (int j = 0; j < matrix.columns; j++)
                {
                    res[i, j] = matrix[i, j] / n;
                }
            });
            return res;
        }
    }
}
