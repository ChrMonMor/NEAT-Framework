using System;
using System.Collections.Generic;
using System.Linq;

namespace NEAT
{
    public class Program
    {
        // Global Varibles
        private static int Pop_Size;
        private static int InputNodes;
        private static int HiddenNodes;
        private static int OutputNodes;
        private static float ProcentConnection; // goes between 0 and 1
        private static int SpeciesTarget;
        private static float ComputedThreshold;
        private static float MutationRate;
        private static float MutationChance;

        private static float[] SpeciateCoefficient;

        public static int Generation = 0;
        static void Main(string[] args)
        {
            Pop_Size = 50;
            InputNodes = 2;
            HiddenNodes = 1;
            OutputNodes = 1;
            ProcentConnection = 1f;
            SpeciesTarget = 5;
            ComputedThreshold = 99f;
            MutationRate = 1f;
            MutationChance = 0.8f;
            SpeciateCoefficient = new float[] { 1f, 1f, 0.4f };

            var tests = new (float[] input, float expected)[]
            {
                (new float[] {0, 0, 1}, 0),
                (new float[] {0, 1, 1}, 1),
                (new float[] {1, 0, 1}, 1),
                (new float[] {1, 1, 1}, 0)
            };

            var nodesArr = new int[] { InputNodes, HiddenNodes, OutputNodes };
            List<Node> bias = new List<Node>() { new Node(nodesArr.Length + 1, NodeType.BIAS, 1) };
            List<Brain> brains = Brain.RunInitialies(nodesArr, Pop_Size, bias, ProcentConnection);
            TestRun(brains, tests);
            while (true)
            {
                if (brains.Max(x => x.Fitness) == 4)
                {
                    break;
                }
                brains = Crossover.NextGeneration(brains, nodesArr, Pop_Size, bias, ProcentConnection, ComputedThreshold, SpeciateCoefficient);

                if (Speciate.Species.Count > SpeciesTarget)
                {
                    ComputedThreshold += 0.5f;
                }
                else
                {
                    ComputedThreshold -= 0.5f;
                }
                Generation++;
                TestRun(brains, tests);
            }
            brains = brains.OrderByDescending(x => x.Fitness).ToList();

        }
        public Brain GetBrain(int popSize, int inputNodes, int hiddenNodes, int outputNodes, float connectProcent, int speciesTarget, int computThreshold, float mutationRate, float mutationChance, float[] speciateCoefficient, (float[] input, float expected)[] tests, List<Node> biasNodes, float targetFitness, int MaxGenerations)
        {
            Pop_Size = popSize;
            InputNodes = inputNodes;
            HiddenNodes = hiddenNodes;
            OutputNodes = outputNodes;
            ProcentConnection = connectProcent;
            SpeciesTarget = speciesTarget;
            ComputedThreshold = computThreshold;
            MutationRate = mutationRate;
            MutationChance = mutationChance;
            SpeciateCoefficient = speciateCoefficient;
            Generation = 0;

            var nodesArr = new int[] { InputNodes, HiddenNodes, OutputNodes };
            List<Node> bias = biasNodes;
            List<Brain> brains = Brain.RunInitialies(nodesArr, Pop_Size, bias, ProcentConnection);
            TestRun(brains, tests);
            while (Generation < MaxGenerations)
            {
                if (brains.Max(x => x.Fitness) == targetFitness)
                {
                    break;
                }

                brains = Crossover.NextGeneration(brains, nodesArr, Pop_Size, bias, ProcentConnection, ComputedThreshold, SpeciateCoefficient);

                if (Speciate.Species.Count > SpeciesTarget)
                {
                    ComputedThreshold += 0.5f;
                }
                else
                {
                    ComputedThreshold -= 0.5f;
                }
                Generation++;
                TestRun(brains, tests);
            }
            brains = brains.OrderByDescending(x => x.Fitness).ToList();

            return brains[0];
        }
        public static void TestRun(List<Brain> brains, (float[] input, float expected)[] tests)
        {

            foreach (var brain in brains)
            {
                float fitness = 0f;
                foreach (var (input, expected) in tests)
                {
                    float output = brain.Run(input).Average();
                    fitness += 1 - Math.Abs(expected - output);
                }
                brain.Fitness = fitness;
            }
        }
        public Brain GetBestEver()
        {
            return Crossover.GetBestEver();
        }
    }
}
