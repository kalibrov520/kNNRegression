﻿using System;
 using System.Collections.Generic;
 using System.IO;

 namespace kNNRegression
{
    class Program
    {
        static void Main(string[] args)
        {
            var samples = new List<Point>();
            
            try
            {
                string[] splitted;
                double a, b, c, d, e;
                int type;

                using (var reader = new StreamReader("teachingAssistant.csv"))
                {
                    reader.ReadLine();

                    while (!reader.EndOfStream)
                    {
                        var line = reader.ReadLine();
                        splitted = line.Split(",");
                        a = double.Parse(splitted[0]);
                        b = double.Parse(splitted[1]);
                        c = double.Parse(splitted[2]);
                        d = double.Parse(splitted[3]);
                        e = double.Parse(splitted[4]);
                        type = int.Parse(splitted[5]);
                        samples.Add(new Point(a, b, c, d, e, type));
                    }
                }
                
                var CV = new CrossValidation(20, samples);
                
                Console.WriteLine(CV.GetF1Measure());
            }
            catch (FileNotFoundException e)
            {
                Console.WriteLine(e);
            }
            catch (IOException e)
            {
                Console.WriteLine(e);
            }
        }
    }
}