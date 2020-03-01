using System;
using System.Collections.Generic;
using System.Linq;

namespace kNNRegression
{
    public class CrossValidation
    {
        int blocksNumber;
        List<Point> samples;
        List<List<Point>> testSamples;
        List<List<Point>> trainingSamples;

        public CrossValidation(int blocksNumber, List<Point> samples)
        {
            this.blocksNumber = blocksNumber;
            this.samples = samples;
            
            testSamples = new List<List<Point>>();
            trainingSamples = new List<List<Point>>();
            
            int pointsInBlock = samples.Count / blocksNumber;
            int cursor = 0;
            
            for (int i = 0; i < blocksNumber; i++)
            {
                testSamples.Add(samples.Skip(i * pointsInBlock).Take(i * pointsInBlock + pointsInBlock).ToList());
                cursor = i * pointsInBlock + pointsInBlock;
                
                List<Point> newTraining = new List<Point>();
                newTraining.AddRange(samples.Skip(0).Take(i * pointsInBlock));
                newTraining.AddRange(samples.Skip(i * pointsInBlock + pointsInBlock).Take(samples.Count));
                
                trainingSamples.Add(newTraining);
            }

            int blocksLeft = samples.Count % blocksNumber;
            testSamples.Add(samples.Skip(cursor).Take(samples.Count).ToList());
            trainingSamples.Add(samples.Skip(0).Take(cursor).ToList());
            Console.WriteLine('h');
        }

        public double GetF1Measure()
        {
            int TP = 0;
            int FP = 0;
            int FN = 0;
            int TN = 0;
            int P = 0;
            int N = 0;
            List<Point> wrongFound = new List<Point>();

            foreach (var p in samples)
            {
                if (p.Type == 0)
                {
                    P += 1;
                }
                else
                {
                    N += 1;
                }
            }

            Console.WriteLine("s");
            
            for (int i = 0; i < testSamples.Count; i++)
            {
                List<Point> testSample = testSamples[i];
                List<Point> trainingSample = trainingSamples[i];

                foreach (var p in testSample)
                {
                    List<Point> hate = new List<Point>(trainingSample);
                    Classifier classifier = new Classifier(hate, p, 3); // chose k
                    
                    /** Classifier **/
                    
                    if (classifier.SimpleClassifyWithKernel() != p.Type)
                    {
                        // chose classification
                        if (p.Type == 0)
                        {
                            FN += 1;
                        }
                        else
                        {
                            FP += 1;
                        }

                        wrongFound.Add(p);
                    }
                    else
                    {
                        if (p.Type == 0)
                        {
                            TN += 1;
                        }
                        else
                        {
                            TP += 1;
                        }
                    }
                }


                foreach (var p in testSample)
                {
                    List<Point> hate = new List<Point>(trainingSample);
                    Classifier classifier = new Classifier(hate, p, 3); // chose k
                    
                    /** Classification **/
                    
                    if (classifier.SimpleClassifyWithKernel() != p.Type)
                    {
                        if (p.Type == 0)
                        {
                            FN += 1;
                        }
                        else if (p.Type == 1)
                        {
                            FP += 1;
                        }

                        wrongFound.Add(p);
                    }
                    else
                    {
                        if (p.Type == 0)
                        {
                            TN += 1;
                        }
                        else
                        {
                            TP += 1;
                        }
                    }
                }
            }

            /*Plot plot1 = new Plot(this.samples, new ArrayList<>());
            plot1.start(this.samples, new ArrayList<>());
            Plot plot = new Plot(this.samples, wrongFound);
            plot.start(this.samples, wrongFound);*/
            
            double prec = (double) TP / (TP + FP);
            double rec = (double) TP / P;
            double f1 = 2 * (prec * rec) / (prec + rec);

            double prec2 = (double) FP / (TP + FP);
            double rec2 = (double) FP / N;
            double f2 = 2 * (prec2 * rec2) / (prec2 + rec2);

            Console.WriteLine("TP: " + TP);
            Console.WriteLine("FP: " + FP);
            Console.WriteLine("TN: " + TN);
            Console.WriteLine("FN: " + FN);
            Console.WriteLine("f1 for 1: " + f1);
            Console.WriteLine("f2 for 2: " + f2);
            
            return f1;
        }
    }
}