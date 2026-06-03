using System.Diagnostics;

namespace QuizDotnet;

partial class Program
{
    static async Task ExtractAudioAsync(string inputPath, string outputPath)
    {
        Console.WriteLine($"Extracting audio via FFmpeg to {outputPath}...");
        var startInfo = new ProcessStartInfo
        {
            FileName = "ffmpeg",
            Arguments = $"-y -i \"{inputPath}\" -vn -acodec pcm_s16le -ar 16000 -ac 1 \"{outputPath}\"",
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            UseShellExecute = false,
            CreateNoWindow = true
        };
        using var process = Process.Start(startInfo);
        if (process == null) throw new Exception("Failed to start FFmpeg.");
        await process.WaitForExitAsync();
        if (process.ExitCode != 0)
        {
            string error = await process.StandardError.ReadToEndAsync();
            throw new Exception($"FFmpeg failed with code {process.ExitCode}: {error}");
        }
    }

    static void SaveTranscript(string inputPath, List<TranscriptSegment> segments, string modelName)
    {
        string dir = "transcripts";
        Directory.CreateDirectory(dir);
        string baseName = Path.GetFileNameWithoutExtension(inputPath);
        string fileName = $"{baseName}_{DateTime.Now:yyyyMMdd_HHmmss}.txt";
        string path = Path.Combine(dir, fileName);

        using var writer = new StreamWriter(path);
        writer.WriteLine($"# Transcript: {baseName}");
        writer.WriteLine($"# Generated: {DateTime.Now:yyyy-MM-dd HH:mm:ss}");
        writer.WriteLine($"# Model: {modelName}");
        writer.WriteLine($"# Segments: {segments.Count}");
        writer.WriteLine();
        foreach (var seg in segments)
        {
            writer.WriteLine($"[{FormatTimestamp(seg.Start)} -> {FormatTimestamp(seg.End)}] {seg.Text.Trim()}");
        }

        Console.WriteLine($"Transcript saved to {path}");
    }

    // Parses a transcript previously written by SaveTranscript. Returns the segments and the model name
    // recorded in the "# Model:" header. Lines look like: [MM:SS.mmm -> MM:SS.mmm] text
    static (List<TranscriptSegment> Segments, string Model) LoadTranscript(string path)
    {
        var segments = new List<TranscriptSegment>();
        string model = "loaded-transcript";

        foreach (var raw in File.ReadAllLines(path))
        {
            string line = raw.Trim();
            if (line.Length == 0) continue;

            if (line.StartsWith('#'))
            {
                if (line.StartsWith("# Model:"))
                    model = line["# Model:".Length..].Trim();
                continue;
            }

            if (!line.StartsWith('[')) continue;
            int close = line.IndexOf(']');
            if (close < 0) continue;

            string timePart = line.Substring(1, close - 1);   // "MM:SS.mmm -> MM:SS.mmm"
            string text = line[(close + 1)..].Trim();

            var bounds = timePart.Split("->", StringSplitOptions.TrimEntries);
            if (bounds.Length != 2) continue;

            segments.Add(new TranscriptSegment(text, ParseTimestamp(bounds[0]), ParseTimestamp(bounds[1]), "SPEAKER_0"));
        }

        return (segments, model);
    }

    static string FormatTimestamp(double seconds)
    {
        var ts = TimeSpan.FromSeconds(seconds);
        return $"{(int)ts.TotalMinutes:D2}:{ts.Seconds:D2}.{ts.Milliseconds:D3}";
    }

    // Inverse of FormatTimestamp: "MM:SS.mmm" where MM is total minutes.
    static double ParseTimestamp(string ts)
    {
        var parts = ts.Split(':');
        if (parts.Length != 2) return 0.0;
        double minutes = double.Parse(parts[0], System.Globalization.CultureInfo.InvariantCulture);
        double seconds = double.Parse(parts[1], System.Globalization.CultureInfo.InvariantCulture);
        return minutes * 60.0 + seconds;
    }
}
