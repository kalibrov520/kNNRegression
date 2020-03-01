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

            double x1 = 0, x2 = 0, x3 = 0, y1 = 0, y2 = 0, y3 = 0, z1 = 0, z2 = 0, z3 = 0;
            
            for (int i = 0; i < testSamples.Count; i++)
            {
                List<Point> testSample = testSamples[i];
                List<Point> trainingSample = trainingSamples[i];

                foreach (var p in testSample)
                {
                    List<Point> hate = new List<Point>(trainingSample);
                    Classifier classifier = new Classifier(hate, p, 3); // chose k
                    
                    /** Classifier **/
                    
                    //int classified = classifier.SimpleClassifyWithKernel();
                    int classified = classifier.ClassifyWithParzenWindowAndKernel();  //0.913
                    //int classified = classifier.ClassifyWithParzenWindow();
                    //int classified = classifier.ClassifyWithDistanceWeights();
                    //int classified = classifier.SimpleClassification();  //0.56
                    
                    if (classified != p.Type)
                    {
                        // chose classification
                        if (p.Type == 1)    
                        {
                            if (classified == 2)
                            {
                                y1 += 1;
                            }
                            else
                            {
                                z1 += 1;
                            }
                        }
                        else if (p.Type == 2)
                        {
                            if (classified == 1)
                            {
                                x2 += 1;
                            }
                            else
                            {
                                z2 += 1;
                            }
                        }
                        else
                        {
                            if (classified == 1)
                            {
                                x3 += 1;
                            }
                            else
                            {
                                y3 += 1;
                            }
                        }

                        wrongFound.Add(p);
                    }
                    else
                    {
                        switch (p.Type)
                        {
                            case 1:
                                x1 += 1;
                                break;
                            case 2:
                                y2 += 1;
                                break;
                            case 3:
                                z3 += 1;
                                break;
                        }
                    }
                }
            }

            /*Plot plot1 = new Plot(this.samples, new ArrayList<>());
            plot1.start(this.samples, new ArrayList<>());
            Plot plot = new Plot(this.samples, wrongFound);
            plot.start(this.samples, wrongFound);*/

            double prec1 = x1 / (x1 + y1 + z1);
            double rec1 = x1 / (x1 + x2 + x3);

            double prec2 = y2 / (x2 + y2 + z2);
            double rec2 = y2 / (y1 + y2 + y3);

            double prec3 = z3 / (x3 + y3 + z3);
            double rec3 = z3 / (z1 + z2 + z3);
            
            double f1 = 2 * (prec1 * rec1) / (prec1 + rec1);
            double f2 = 2 * (prec2 * rec2) / (prec2 + rec2);
            double f3 = 2 * (prec3 * rec3) / (prec3 + rec3);

            double Fm = (f1 + f2 + f3) / 3;


            Console.WriteLine("f1 for 1: " + f1);
            Console.WriteLine("f2 for 2: " + f2);
            Console.WriteLine("f3 for 3: " + f3);
            Console.WriteLine("f-macro:" + Fm);
            
            return Fm;
        }
    }
}