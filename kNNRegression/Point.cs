using System;

namespace kNNRegression
{
    public class Point
    {
        public double A { get; set; }
        public double B { get; set; }
        public double C { get; set; }
        public double D { get; set; }
        public double E { get; set; }
        public int Type { get; set; }
        public double Distance { get; set; }

        public Point(double a, double b, double c, double d, double e, int  type)
        {
            A = a;
            B = b;
            C = c;
            D = d;
            E = e;
            if (type != 1 && type != 2 && type != 3)
            {
                throw new Exception("WRONG TYPE!");
            }

            Type = type;
        }

        public void Reset()
        {
            Distance = Double.MaxValue;
        }
    }
}