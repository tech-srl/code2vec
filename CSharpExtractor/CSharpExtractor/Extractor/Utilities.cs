using CommandLine;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Diagnostics;
using System.Text.RegularExpressions;

namespace Extractor
{
    public class Options
    {
        [Option('t', "threads", Default = 1, HelpText = "How many threads to use <1>")]
        public int Threads { get; set; }

        [Option('p', "path", Default = "./data/", HelpText = "Where to find code files. <.>")]
        public string Path { get; set; }

        [Option('l', "max_length", Default = 9, HelpText = "Max path length")]
        public int MaxLength { get; set; }

        [Option('l', "max_width", Default = 2, HelpText = "Max path length")]
        public int MaxWidth { get; set; }

        [Option('o', "ofile_name", Default = "test.txt", HelpText = "Output file name")]
        public String OFileName { get; set; }

        [Option('h', "no_hash", Default = false, HelpText = "When enabled, prints the whole path strings (not hashed)")]
        public Boolean NoHash { get; set; }

        [Option('l', "max_contexts", Default = 30000, HelpText = "Max number of path contexts to sample. Affects only very large snippets")]
        public int MaxContexts { get; set; }
    }

    public static class Utilities
	{
	    public static String[] NumbericLiteralsToKeep = new String[] { "0", "1", "2", "3", "4", "5", "10" };
        public static IEnumerable<Tuple<T, T>> Choose2<T>(IEnumerable<T> enumerable)
		{
			int index = 0;

			foreach (var e in enumerable)
			{
				++index;
				foreach (var t in enumerable.Skip(index))
					yield return Tuple.Create(e, t);
			}
		}

        /// <summary>
        /// Sample uniform randomly numSamples from an enumerable, using reservoir sampling.
        /// See https://en.wikipedia.org/wiki/Reservoir_sampling
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="input"></param>
        /// <param name="numSamples"></param>
        /// <returns></returns>
        public static IEnumerable<TSource> ReservoirSample<TSource>(this IEnumerable<TSource> input, int numSamples)
        {
            var rng = new Random();
            var sampledElements = new List<TSource>(numSamples);
            int seenElementCount = 0;
            foreach (var element in input)
            {
                seenElementCount++;
                if (sampledElements.Count < numSamples)
                {
                    sampledElements.Add(element);
                }
                else
                {
                    int position = rng.Next(seenElementCount);
                    if (position < numSamples)
                    {
                        sampledElements[position] = element;
                    }
                }
            }
            Debug.Assert(sampledElements.Count <= numSamples);
            return sampledElements;
        }


        public static IEnumerable<T> WeakConcat<T>(IEnumerable<T> enumerable1, IEnumerable<T> enumerable2)
		{
			foreach (T t in enumerable1)
				yield return t;
			foreach (T t in enumerable2)
				yield return t;
		}

        public static IEnumerable<String> SplitToSubtokens(String name)
        {
            return Regex.Split(name.Trim(), "(?<=[a-z])(?=[A-Z])|_|[0-9]|(?<=[A-Z])(?=[A-Z][a-z])|\\s+")
                .Where(s => s.Length > 0)
                .Select(s => NormalizeName(s))
                .Where(s => s.Length > 0);
        }

        private static Regex Whitespaces = new Regex(@"\s");
        private static Regex NonAlphabetic = new Regex("[^A-Za-z]");

        public static String NormalizeName(string s)
        {
            String partiallyNormalized = s.ToLowerInvariant()
                .Replace("\\\\n", String.Empty)
                .Replace("[\"',]", String.Empty);

            partiallyNormalized = Whitespaces.Replace(partiallyNormalized, "");
            partiallyNormalized = Encoding.ASCII.GetString(
                Encoding.Convert(
                    Encoding.UTF8,
                    Encoding.GetEncoding(
                        Encoding.ASCII.EncodingName,
                        new EncoderReplacementFallback(string.Empty),
                        new DecoderExceptionFallback()
                    ),
                    Encoding.UTF8.GetBytes(partiallyNormalized)
                )
            );

            if (partiallyNormalized.Contains('\n'))
            {
                partiallyNormalized = partiallyNormalized.Replace('\n', 'N');
            }
            if (partiallyNormalized.Contains('\r'))
            {
                partiallyNormalized = partiallyNormalized.Replace('\r', 'R');
            }
            if (partiallyNormalized.Contains(','))
            {
                partiallyNormalized = partiallyNormalized.Replace(',', 'C');
            }

            String completelyNormalized = NonAlphabetic.Replace(partiallyNormalized, String.Empty);
            if (completelyNormalized.Length == 0)
            {
                if (Regex.IsMatch(partiallyNormalized, @"^\d+$"))
                {
                    if (NumbericLiteralsToKeep.Contains(partiallyNormalized))
                    {
                        return partiallyNormalized;
                    }
                    else
                    {
                        return "NUM";
                    }
                }

                return String.Empty;
            }
            return completelyNormalized;
            
        }
    }
}
