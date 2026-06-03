using System.Diagnostics;

namespace QuizDotnet;

partial class Program
{
    static void LogStage(string stageName, Stopwatch stopwatch)
    {
        Console.WriteLine($"[TIMING] {stageName} took {FormatElapsed(stopwatch.Elapsed)}.");
        stopwatch.Restart();
    }

    static string FormatElapsed(TimeSpan elapsed)
    {
        if (elapsed.TotalSeconds < 60)
            return $"{elapsed.TotalSeconds:F2}s";
        return $"{(int)elapsed.TotalMinutes}m {elapsed.Seconds}s ({elapsed.TotalSeconds:F1}s)";
    }
}
