using System;
using System.Collections.Generic;
using System.Linq;

namespace kNNRegression
{
    public class Classifier
    {
        private List<Point> samples;
        private Point testSample;
        private int k;

        public Classifier(List<Point> samples, Point testSample, int k)
        {
            this.samples = samples;
            this.testSample = testSample;
            this.k = k;
        }

        public List<Point> FindKNearest()
        {
            foreach (var pointTo in samples)
            {
                pointTo.Distance = FindEuclideanDistance(testSample, pointTo);
            }
            
            samples.Sort((o1, o2) => o1.Distance.CompareTo(o2.Distance));

            var kNearest = samples.Take(k).ToList();

            foreach (var pointTo in samples)
            {
                pointTo.Reset();
            }

            return kNearest;
        }
        
        public void SetDistances() {
            foreach (var point in samples)
            {
                point.Distance = FindEuclideanDistance(testSample, point);
            }
        }
        
        /** Classification **/
        public double SimpleClassification()
        {
            var nearest = FindKNearest();
            int one = 0;
            int two = 0;
            int three = 0;

            foreach (var point in nearest)
            {
                if (point.Type == 1)
                {
                    one += 1;
                } else if (point.Type == 2)
                {
                    two += 1;
                }
                else
                {
                    three += 1;
                }
            }

            return one > two ? (one > three ? 1 : 3) : (two > three ? 2 : 3);
        }

        public double ClassifyWithConstantWeights()
        {
            var nearest = FindKNearest();
            double one = 0;
            double two = 0;
            double three = 0;
            int i = 1;
            
            foreach (var point in nearest) 
            {
                if (point.Type == 1) {
                    one += (double) 1 / i;
                } 
                else if (point.Type == 2) {
                    two += (double) 1 / i;
                }
                else
                {
                    three += (double) 1 / i;
                }
                
                i += 1;
            }
            return one > two ? (one > three ? 1 : 3) : (two > three ? 2 : 3);
        }

        public int ClassifyWithDistanceWeights()
        {
            List<Point> nearest = FindKNearest();
            double one = 0;
            double two = 0;
            double three = 0;
            int i = 1;
                
            foreach (var point in nearest) {
                if (point.Type == 1)
                {
                    one += 1 / point.Distance;
                }
                else if (point.Type == 2)
                {
                    two += 1 / point.Distance;
                }
                else
                {
                    three += 1 / point.Distance;
                }

                i += 1;
            }
            
            return one > two ? (one > three ? 1 : 3) : (two > three ? 2 : 3);
        }

        public int ClassifyWithParzenWindow()
        {
            var nearest = new List<Point>(samples);
            double one = 0;
            double two = 0;
            double three = 0;
            /**
             *  change width
             */
            var windowWidth = 0.1;
            
            SetDistances();
            List<Point> list = new List<Point>(nearest);

            list = list.Where(p => p.Distance < windowWidth).ToList();

            foreach (var point in list)
            {
                if (point.Type == 1)
                {
                    one += 1 / point.Distance;
                }
                else if (point.Type == 2)
                {
                    two += 1 / point.Distance;
                }
                else
                {
                    three += 1 / point.Distance;
                }
            }

            return one > two ? (one > three ? 1 : 3) : (two > three ? 2 : 3);
        }

        public int SimpleClassifyWithKernel()
        {
            var nearest = FindKNearest();
            var test = testSample;
            double one = 0;
            double two = 0; 
            double three = 0;

            foreach (var point in nearest)
            {
                var kernelVal = PolynomialKernel(test, point, 2);
                
                if (point.Type == 1)
                {
                    one += kernelVal;
                }
                else if (point.Type == 2)
                {
                    two += kernelVal;
                }
                else
                {
                    three += kernelVal;
                }
            }
            
            return one > two ? (one > three ? 1 : 3) : (two > three ? 2 : 3);
        }

        public int ClassifyWithParzenWindowAndKernel()
        {
            var nearest = new List<Point>(samples);
            double one = 0;
            double two = 0;
            double three = 0;
            /**
             *  change width
             */
            double windowWidth = 0.3;
            SetDistances();
            var list = new List<Point>(nearest);

            list = list.Where(p => p.Distance < windowWidth).ToList();

            foreach (var point in list)
            {
                if (point.Type == 1)
                {
                    one += GaussianKernel(point.Distance);
                }
                else if (point.Type == 2)
                {
                    two += GaussianKernel(point.Distance);
                }
                else
                {
                    three += GaussianKernel(point.Distance);
                }
            }

            return one > two ? (one > three ? 1 : 3) : (two > three ? 2 : 3);
        }
        
        
        
        /** Kernels **/
        private static double GaussianKernel(double distance)
            => Math.Pow(2 * Math.PI, -0.5) * Math.Exp(-0.5 * Math.Pow(distance, 2));

        private static double PolynomialKernel(Point from, Point to, int p)
            => Math.Pow(1 + (from.A * to.A + from.B * to.B + from.C * to.C + from.D * to.D + from.E * to.E), p);

        private static double RadialKernel(Point from, Point to, double s) 
            => Math.Exp(-Math.Pow(Math.Sqrt(Math.Pow(from.A - to.A, 2) + Math.Pow(from.B - to.B, 2) + Math.Pow(from.C - to.C, 2)
                                + Math.Pow(from.D - to.D, 2) + Math.Pow(from.E - to.E, 2)), 2) / s);
        
        
        /** Distances **/
        private static double FindEuclideanDistance(Point from, Point to)
        {
            double A = Math.Pow(from.A - to.A, 2);
            double B = Math.Pow(from.B - to.B, 2);
            double C = Math.Pow(from.C - to.C, 2);
            double D = Math.Pow(from.D - to.D, 2);
            double E = Math.Pow(from.E - to.E, 2);
            
            return Math.Sqrt(A + B + C + D + E);
        }

        private static double FindManhattanDistance(Point from, Point to)
        {
            double A = Math.Abs(from.A - to.A);
            double B = Math.Abs(from.B - to.B);
            double C = Math.Abs(from.C - to.C);
            double D = Math.Abs(from.D - to.D);
            double E = Math.Abs(from.E - to.E);

            return A + B + C + D + E;
        }
    }
}