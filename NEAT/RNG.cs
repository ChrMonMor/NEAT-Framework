using System;

namespace NEAT
{
    static class RNG
    {
        readonly static Random ran = new Random();
        public static float RanDub()
        {
            return (float)ran.NextDouble();
        }
        public static int Ran(int min, int max)
        {
            return ran.Next(min, max);
        }
    }
}
