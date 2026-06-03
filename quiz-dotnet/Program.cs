using System.Diagnostics;
using Whisper.net;
using Whisper.net.Ggml;

namespace QuizDotnet;

// Pipeline orchestration. Each stage lives in its own file as a partial of this class:
//   PipelineConfig.cs       - constants / tuning
//   Transcription.cs        - audio extraction + transcript save/load
//   SemanticChunking.cs     - sentence splitting + embedding + chunking
//   ChunkClassification.cs  - classify + embed + store in Qdrant
//   QuizGeneration.cs       - RAG question generation, judging, dedup, quiz output
//   Timing.cs               - stage timing helpers
partial class Program
{
    static async Task Main(string[] args)
    {
        Console.WriteLine("=== Quiz Generator Pipeline (.NET 10 Unified Stack) ===");

        var pipelineStart = DateTimeOffset.Now;
        var totalStopwatch = Stopwatch.StartNew();
        Console.WriteLine($"Pipeline started at: {pipelineStart:yyyy-MM-dd HH:mm:ss}");

        // CLI: [inputAudioOrVideo] [--transcript <path>]
        // --transcript loads a previously saved transcript and skips audio extraction + Whisper entirely.
        string? transcriptPath = null;
        string? positionalInput = null;
        for (int a = 0; a < args.Length; a++)
        {
            if ((args[a] == "--transcript" || args[a] == "-t") && a + 1 < args.Length)
            {
                transcriptPath = args[++a];
            }
            else if (!args[a].StartsWith('-'))
            {
                positionalInput ??= args[a];
            }
        }

        string inputPath = positionalInput ?? "/home/ubuntu/quiz/test01_20s.wav";
        string courseId = "course-123";
        string sourceId = Guid.NewGuid().ToString();
        // Unique collection per run so vectors don't pile up in a shared collection
        string collectionName = $"quiz_chunks_dotnet_{DateTime.Now:yyyyMMdd_HHmmss}";
        Console.WriteLine($"Qdrant collection for this run: {collectionName}");

        var stageStopwatch = Stopwatch.StartNew();
        string whisperModelName;
        List<TranscriptSegment> rawSegments;

        if (transcriptPath != null)
        {
            // Reuse an existing transcript (e.g. from quiz-dotnet/transcripts) — skips the slow Whisper stage.
            if (!File.Exists(transcriptPath))
            {
                Console.WriteLine($"Transcript file not found: {transcriptPath}");
                return;
            }
            Console.WriteLine($"Loading transcript from {transcriptPath} (skipping audio extraction + Whisper)...");
            (rawSegments, whisperModelName) = LoadTranscript(transcriptPath);
            inputPath = transcriptPath; // used only for naming the quiz output file
            Console.WriteLine($"Loaded {rawSegments.Count} segments (model: {whisperModelName}).");
            LogStage("Transcript Load", stageStopwatch);
        }
        else
        {
            // 1. Download Whisper model if not exists
            var whisperModelType = GgmlType.Medium;
            whisperModelName = $"whisper-{whisperModelType.ToString().ToLower()}";
            string modelPath = $"ggml-{whisperModelType.ToString().ToLower()}.bin";
            if (!File.Exists(modelPath))
            {
                Console.WriteLine($"Downloading Whisper GGML {whisperModelType} model to {modelPath}...");
                using var modelStream = await WhisperGgmlDownloader.Default.GetGgmlModelAsync(whisperModelType);
                using var fileStream = File.OpenWrite(modelPath);
                await modelStream.CopyToAsync(fileStream);
                Console.WriteLine("Whisper model downloaded successfully.");
            }
            stageStopwatch.Restart();

            // 2. Audio Extraction
            Directory.CreateDirectory("audio");
            string wavPath = Path.Combine("audio", $"{Path.GetFileNameWithoutExtension(inputPath)}_{DateTime.Now:yyyyMMdd_HHmmss}.wav");
            if (File.Exists(wavPath)) File.Delete(wavPath);
            await ExtractAudioAsync(inputPath, wavPath);
            LogStage("Audio Extraction", stageStopwatch);

            // 3. Whisper Transcription
            Console.WriteLine("Starting Whisper transcription...");
            using var whisperFactory = WhisperFactory.FromPath(modelPath);
            using var processor = whisperFactory.CreateBuilder()
                .WithLanguage("es")
                .Build();

            using var audioStream = File.OpenRead(wavPath);
            rawSegments = new List<TranscriptSegment>();
            await foreach (var result in processor.ProcessAsync(audioStream))
            {
                rawSegments.Add(new TranscriptSegment(
                    result.Text,
                    result.Start.TotalSeconds,
                    result.End.TotalSeconds,
                    "SPEAKER_0" // Default speaker fallback
                ));
            }
            Console.WriteLine($"Transcribed {rawSegments.Count} raw segments.");
            LogStage("Whisper Transcription", stageStopwatch);

            // Save raw transcript for quality monitoring + reuse via --transcript on later runs
            SaveTranscript(inputPath, rawSegments, whisperModelName);
        }

        // 4. Split into sentences/phrases
        var sentences = SplitIntoSentences(rawSegments);
        Console.WriteLine($"Grouped into {sentences.Count} sentences.");
        LogStage("Sentence Splitting", stageStopwatch);

        // 5. Semantic Chunking
        var chunks = await GenerateSemanticChunksAsync(sentences);
        Console.WriteLine($"Generated {chunks.Count} semantic chunks.");
        LogStage("Semantic Chunking", stageStopwatch);

        // 6. Classification, Summarization & Embedding Storage
        var processedChunks = await ProcessChunksAndStoreAsync(chunks, courseId, sourceId, collectionName);
        LogStage("Classification & Embedding Storage", stageStopwatch);

        // 7. RAG & Quiz Generation (numQuestions <= 0 => adaptive based on academic chunk count)
        var quizStopwatch = Stopwatch.StartNew();
        var generatedQuestions = await GenerateQuizQuestionsAsync(courseId, 0, "comprender", collectionName);
        quizStopwatch.Stop();
        LogStage("RAG & Quiz Generation", stageStopwatch);

        Console.WriteLine("\n=== Pipeline Execution Completed Successfully ===");
        Console.WriteLine($"Generated {generatedQuestions.Count} valid quiz questions.");
        SaveQuiz(inputPath, generatedQuestions, whisperModelName, quizStopwatch.Elapsed, totalStopwatch.Elapsed);
        foreach (var q in generatedQuestions)
        {
            Console.WriteLine($"\nPregunta: {q.Question}");
            foreach (var opt in q.Options)
            {
                Console.WriteLine($"  {opt.Key}) {opt.Value}");
            }
            Console.WriteLine($"Respuesta correcta: {q.CorrectOption}");
            Console.WriteLine($"Justificación: {q.Justification}");
        }

        totalStopwatch.Stop();
        var pipelineEnd = DateTimeOffset.Now;
        Console.WriteLine($"\nPipeline started at:  {pipelineStart:yyyy-MM-dd HH:mm:ss}");
        Console.WriteLine($"Pipeline finished at: {pipelineEnd:yyyy-MM-dd HH:mm:ss}");
        Console.WriteLine($"Total elapsed time:   {FormatElapsed(totalStopwatch.Elapsed)}");
    }
}
