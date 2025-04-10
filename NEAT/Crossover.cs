using System.Collections.Generic;
using System.Linq;

namespace NEAT
{
    public static class Crossover
    {
        private static bool Eltism = true;
        private static Brain BestEver = new Brain();
        private static float MutateWeightsChance = 0.8f, MutateWeightsRange = 0.2f, MutateConnectionsChance = 0.05f, MutateConnectionEnablingChance = 0.2f, MutateNodesChance = 0.2f;
        private static bool RecusionAllowed = false;
        private static int Range = 20;
        public static List<Brain> NextGeneration(List<Brain> networks, int[] inHidOut, int popSize, List<Node> biasNodes = null, float connectProcent = 1f, float threshold = 99, float[] coefficients = null)
        {
            List<Brain> nextNetwork = Brain.RunInitialies(inHidOut, popSize, biasNodes, connectProcent);

            var fitest = networks.Max(y => y.Fitness);

            Brain BestInTest = networks.First(x => x.Fitness == fitest);

            if (BestInTest.Fitness > BestEver.Fitness)
            {
                BestEver = BestInTest;
            }

            foreach (var net in networks)
            {
                net.Species = -1;
            }

            List<Species> speciesList = Speciate.GenerationOffspring(networks, threshold, coefficients);

            int n = 0;
            foreach (var last in speciesList)
            {
                for (int i = 0; i < last.AllowedOffspring; i++)
                {
                    Brain parrentA = SoftMaxSelect(last.Members, last.SumFitness);
                    Brain parrentB = SoftMaxSelect(last.Members, last.SumFitness);

                    Brain child = new Brain(parrentA, parrentB);

                    child.Mutate(MutateWeightsChance, MutateWeightsRange, MutateConnectionsChance, MutateConnectionEnablingChance, RecusionAllowed, Range, MutateNodesChance);

                    if (n < popSize)
                    {
                        nextNetwork[n++] = child;
                    }
                    else
                    {
                        nextNetwork.Add(child);
                    }
                }
            }

            if (Eltism)
            {
                if (nextNetwork.Count == 0)
                {
                    nextNetwork.Add(new Brain());
                }
                nextNetwork[0] = BestEver;
            }

            return nextNetwork.Take(popSize).ToList();

        }
        public static void MutationVariables(float mutateWeightsChance = 0.8f, float mutateWeightsRange = 0.2f, float mutateConnectionsChance = 0.05f, float mutateConnectionEnablingChance = 0.2f, bool recusionAllowed = false, int range = 20, float mutateNodesChance = 0.2f)
        {
            MutateWeightsChance = mutateWeightsChance;
            MutateWeightsRange = mutateWeightsRange;
            MutateConnectionsChance = mutateConnectionsChance;
            MutateConnectionEnablingChance = mutateConnectionEnablingChance;
            RecusionAllowed = recusionAllowed;
            Range = range;
            MutateNodesChance = mutateNodesChance;
        }
        public static void EltiOnOff()
        {
            Eltism = !Eltism;
        }
        private static Brain SoftMaxSelect(List<Brain> values, float maxValue)
        {
            float selected = RNG.RanDub() * (maxValue * 2) + 0;
            float sums = 0;

            foreach (var item in values)
            {
                sums += item.Fitness;
                if (sums > selected)
                {
                    return item;
                }
            }

            return values.Last();
        }
        internal static Brain GetBestEver()
        {
            return BestEver;
        }
    }
}
