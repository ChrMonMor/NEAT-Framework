using System;
using System.Collections.Generic;
using System.Linq;

namespace NEAT
{
    internal static class Speciate
    {
        internal static List<Species> Species = new List<Species>();
        internal static float GlobalAdjustedFitness;
        internal static int GenStagnationLimit = 15;
        internal static List<Species> GenerationOffspring(List<Brain> networks, float threshold, float[] coefficient = null)
        {
            GlobalAdjustedFitness = 0;
            int n = 0;
            foreach (var s in Species)
            {

                var rep = s.Members[RNG.Ran(0, s.Members.Count)];
                s.SumFitness = 0;
                foreach (var item in networks)
                {
                    if (item.Species != -1)
                    {
                        continue;
                    }
                    if (SpeciateComparisonCheck(rep, item, threshold, coefficient))
                    {
                        s.SumFitness += item.Fitness;
                    }
                }
            }

            if (Species.Count > 0)
            {
                n = Species.Max(x => x.Id) + 1;
            }

            foreach (var net in networks)
            {
                if (net.Species != -1)
                {
                    continue;
                }
                Species.Add(new Species(n));
                var s = Species.Last();
                s.SumFitness = net.Fitness;
                net.Species = n;
                foreach (var item in networks)
                {
                    if (item.Species != -1)
                    {
                        continue;
                    }
                    if (SpeciateComparisonCheck(net, item, threshold, coefficient))
                    {
                        s.SumFitness += item.Fitness;
                    }
                }
                n++;
            }


            foreach (var brain in networks)
            {
                brain.AdjustedFitness = brain.Fitness / networks.Count(x => x.Species == brain.Species);
                Species.First(x => x.Id == brain.Species).AdjustedFitness += brain.AdjustedFitness;
            }


            float GlobalFitness = networks.Sum(x => x.AdjustedFitness) / networks.Count;

            foreach (var res in Species)
            {
                res.Members = networks.Where(x => x.Species == res.Id).ToList();
                res.AdjustedFitness /= res.Members.Count;
                res.AllowedOffspring = (int)Math.Round(res.AdjustedFitness / GlobalFitness * res.Members.Count, 0);
                res.SumFitness = res.Members.Sum(x => x.Fitness);

                if (res.Fitness < res.SumFitness / res.Members.Count)
                {
                    res.Fitness = res.SumFitness / res.Members.Count;
                    res.GensSinceImproved = 0;
                }
                else
                {
                    res.GensSinceImproved++;
                }
                if (res.GensSinceImproved >= GenStagnationLimit)
                {
                    res.AllowedOffspring = 0;
                }
            }

            for (int i = Species.Count - 1; i >= 0; i--)
            {
                if (Species[i].Members.Count <= 0)
                {
                    Species.Remove(Species[i]);
                }
            }

            GlobalAdjustedFitness = GlobalFitness;

            return Species;

        }
        private static bool SpeciateComparisonCheck(Brain netA, Brain netB, float threshold, float[] coefficient = null)
        {
            coefficient = coefficient ?? new[] { 1f, 1f, 1f };
            float excess = 0, disjoint = 0, avgWeight = 0;
            int n = Math.Max(netA.ArrConnections.Count, netB.ArrConnections.Count), avgCount = 0;
            int aMax = netA.ArrConnections.Max(x => x.Key), bMax = netB.ArrConnections.Max(x => x.Key);
            excess = (float)(netA.ArrConnections.Count(x => x.Key > bMax && x.Value.Enabled) + netB.ArrConnections.Count(x => x.Key > aMax && x.Value.Enabled)) / n * coefficient[0];
            disjoint = (float)(netA.ArrConnections.Count(x => x.Key < bMax && x.Value.Enabled && !netB.ArrConnections.ContainsKey(x.Key)) + netB.ArrConnections.Count(x => x.Key < aMax && x.Value.Enabled && !netA.ArrConnections.ContainsKey(x.Key))) / n * coefficient[1];
            // gives avg of connections of the same InnovationID
            foreach (var aConn in netA.ArrConnections)
            {
                if (aConn.Value.Enabled && netB.ArrConnections.ContainsKey(aConn.Key))
                {
                    if (netB.ArrConnections[aConn.Key].Enabled)
                    {
                        avgWeight += Math.Abs(aConn.Value.ConnectionWeight - netB.ArrConnections[aConn.Key].ConnectionWeight);
                        avgCount++;
                    }
                }
            }

            avgWeight = avgCount != 0 ? (float)(avgWeight / avgCount) * coefficient[2] : 99;

            if (avgWeight == float.NaN)
            {
                avgWeight = 99;
            }

            if (excess + disjoint + avgWeight <= threshold)
            {
                netB.Species = netA.Species;
                return true;
            }

            return false;
        }

    }
}
