using System.Collections.Generic;

namespace NEAT
{
    internal class Species
    {
        internal Species(int id, List<Brain> members = null, int allowedOffspring = 0, float fitness = 0f, float adjustedFitness = 0f, float sumFitness = 0f, float gensSinceImproved = 0f)
        {
            Id = id;
            Members = members ?? new List<Brain>();
            AllowedOffspring = allowedOffspring;
            Fitness = fitness;
            AdjustedFitness = adjustedFitness;
            SumFitness = sumFitness;
            GensSinceImproved = gensSinceImproved;
        }
        internal int Id { get; set; }
        internal List<Brain> Members { get; set; }
        internal int AllowedOffspring { get; set; }
        internal float Fitness { get; set; }
        internal float AdjustedFitness { get; set; }
        internal float SumFitness { get; set; }
        internal float GensSinceImproved { get; set; }
        public override string ToString()
        {
            return "" + AdjustedFitness + ", " + AllowedOffspring + ", " + Members.Count;
        }
    }
}
