using System.Diagnostics;
using System.Numerics.Tensors;
using System.Text.Json;
using System.Text.Json.Serialization;
using Microsoft.Extensions.AI;
using Qdrant.Client;
using Qdrant.Client.Grpc;
using Whisper.net;
using Whisper.net.Ggml;

namespace QuizDotnet
{
    public record TranscriptSegment(string Text, double Start, double End, string Speaker);

    public record Sentence(string Text, double Start, double End, string Speaker);

    public record SemanticChunk(int ChunkIndex, double TsStart, double TsEnd, string Speaker, string Content)
    {
        public string Category { get; set; } = "ACADEMICO";
        public double ConfidenceScore { get; set; } = 0.8;
        public string TopicSummary { get; set; } = "";
        public Guid QdrantVectorId { get; set; }
    }

    public class ClassificationResult
    {
        [JsonPropertyName("categoria")]
        public string Categoria { get; set; } = "ACADEMICO";

        [JsonPropertyName("confidence")]
        public double Confidence { get; set; } = 0.8;

        [JsonPropertyName("razon")]
        public string Razon { get; set; } = "";
    }

    public class QuizQuestion
    {
        [JsonPropertyName("question")]
        public string Question { get; set; } = "";

        [JsonPropertyName("options")]
        public Dictionary<string, string> Options { get; set; } = new();

        [JsonPropertyName("correct_option")]
        public string CorrectOption { get; set; } = "";

        [JsonPropertyName("bloom_level")]
        public string BloomLevel { get; set; } = "";

        [JsonPropertyName("justification")]
        public string Justification { get; set; } = "";
    }

    public class ValidationResult
    {
        [JsonPropertyName("score")]
        public double Score { get; set; } = 0.0;

        [JsonPropertyName("reason")]
        public string Reason { get; set; } = "";
    }

    class Program
    {
        // Single source of truth for the Ollama models used across the pipeline.
        const string LlmModel = "qwen2.5:7b-instruct";          // classification, summarization, quiz generation
        const string JudgeModel = "llama3.1:8b-instruct-q4_K_M"; // LLM-as-a-judge (different family to reduce self-bias)
        const string EmbeddingModel = "bge-m3";                  // multilingual embeddings (1024-dim)
        const int EmbeddingDim = 1024;

        // Semantic chunking tuning. Higher MinChunkWords + lower similarity threshold => fewer, larger chunks
        // (fewer classification LLM calls, more context per question).
        const float ChunkSimilarityThreshold = 0.45f; // split only on clearer topic shifts (was 0.5)
        const int MinChunkWords = 80;                  // don't split until a chunk has this many words (was 30)
        const int MaxChunkWords = 400;                 // hard cap before a forced split (was 350)

        // Adaptive quiz size: ~1 question per QuestionsPerChunkDivisor academic chunks, clamped to [Min, Max].
        const double QuestionsPerChunkDivisor = 5.0;   // higher => fewer questions
        const int MinQuestions = 5;
        const int MaxQuestions = 25;

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

        // Inverse of FormatTimestamp: "MM:SS.mmm" where MM is total minutes.
        static double ParseTimestamp(string ts)
        {
            var parts = ts.Split(':');
            if (parts.Length != 2) return 0.0;
            double minutes = double.Parse(parts[0], System.Globalization.CultureInfo.InvariantCulture);
            double seconds = double.Parse(parts[1], System.Globalization.CultureInfo.InvariantCulture);
            return minutes * 60.0 + seconds;
        }

        static void SaveQuiz(string inputPath, List<QuizQuestion> questions, string transcriptModel, TimeSpan quizElapsed, TimeSpan totalElapsed)
        {
            string dir = "quizzes";
            Directory.CreateDirectory(dir);
            string baseName = Path.GetFileNameWithoutExtension(inputPath);
            string fileName = $"{baseName}_{DateTime.Now:yyyyMMdd_HHmmss}.json";
            string path = Path.Combine(dir, fileName);

            var output = new
            {
                generated_at = DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss"),
                source = baseName,
                models = new
                {
                    transcript = transcriptModel,
                    processing = LlmModel,
                    embedding = EmbeddingModel,
                    llm_as_judge = JudgeModel
                },
                timings = new
                {
                    quiz_generation = FormatElapsed(quizElapsed),
                    quiz_generation_seconds = Math.Round(quizElapsed.TotalSeconds, 2),
                    pipeline_total = FormatElapsed(totalElapsed),
                    pipeline_total_seconds = Math.Round(totalElapsed.TotalSeconds, 2)
                },
                questions
            };

            var json = JsonSerializer.Serialize(output, new JsonSerializerOptions
            {
                WriteIndented = true,
                Encoder = System.Text.Encodings.Web.JavaScriptEncoder.UnsafeRelaxedJsonEscaping
            });
            File.WriteAllText(path, json);

            Console.WriteLine($"Quiz saved to {path}");
        }

        static string FormatTimestamp(double seconds)
        {
            var ts = TimeSpan.FromSeconds(seconds);
            return $"{(int)ts.TotalMinutes:D2}:{ts.Seconds:D2}.{ts.Milliseconds:D3}";
        }

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

        static List<Sentence> SplitIntoSentences(List<TranscriptSegment> segments)
        {
            var sentences = new List<Sentence>();
            var currentWords = new List<string>();
            double? startTime = null;
            var speakers = new List<string>();

            foreach (var seg in segments)
            {
                string text = seg.Text.Trim();
                if (string.IsNullOrWhiteSpace(text)) continue;

                if (startTime == null) startTime = seg.Start;
                currentWords.Add(text);
                speakers.Add(seg.Speaker);

                bool endsWithPunc = text.EndsWith('.') || text.EndsWith('?') || text.EndsWith('!');
                int wordCount = currentWords.Sum(w => w.Split(' ', StringSplitOptions.RemoveEmptyEntries).Length);

                if (endsWithPunc || wordCount >= 15)
                {
                    string dominantSpeaker = speakers.GroupBy(s => s)
                                                     .OrderByDescending(g => g.Count())
                                                     .First().Key;
                    sentences.Add(new Sentence(
                        string.Join(" ", currentWords),
                        startTime.Value,
                        seg.End,
                        dominantSpeaker
                    ));
                    currentWords.Clear();
                    startTime = null;
                    speakers.Clear();
                }
            }

            if (currentWords.Count > 0 && segments.Count > 0)
            {
                string dominantSpeaker = speakers.GroupBy(s => s)
                                                 .OrderByDescending(g => g.Count())
                                                 .FirstOrDefault()?.Key ?? "SPEAKER_0";
                sentences.Add(new Sentence(
                    string.Join(" ", currentWords),
                    startTime ?? segments[0].Start,
                    segments[^1].End,
                    dominantSpeaker
                ));
            }

            return sentences;
        }

        static async Task<List<SemanticChunk>> GenerateSemanticChunksAsync(List<Sentence> sentences)
        {
            if (sentences.Count == 0) return new List<SemanticChunk>();

            Console.WriteLine($"Computing sentence embeddings using Ollama ({EmbeddingModel})...");
            IEmbeddingGenerator<string, Embedding<float>> embeddingGenerator =
                new OllamaEmbeddingGenerator(new Uri("http://127.0.0.1:11434"), EmbeddingModel);

            // Embed sentences in batches (fast) with per-item fallback. bge-m3 returns a NaN vector for some
            // degenerate inputs, which Ollama cannot JSON-encode and which would kill an all-in-one batch call.
            var (sentencesEmbedded, embeddings) = await EmbedSentencesAsync(embeddingGenerator, sentences);
            if (sentencesEmbedded.Count == 0) return new List<SemanticChunk>();
            sentences = sentencesEmbedded; // keep sentences and embeddings index-aligned after any drops

            var similarities = new List<float>();
            for (int i = 0; i < embeddings.Count - 1; i++)
            {
                float sim = TensorPrimitives.CosineSimilarity(embeddings[i].Vector.Span, embeddings[i + 1].Vector.Span);
                similarities.Add(sim);
            }

            var chunks = new List<SemanticChunk>();
            var currentSentences = new List<Sentence> { sentences[0] };
            int currentWordCount = sentences[0].Text.Split(' ', StringSplitOptions.RemoveEmptyEntries).Length;
            int chunkIndex = 0;

            for (int i = 0; i < similarities.Count; i++)
            {
                float sim = similarities[i];
                var nextSentence = sentences[i + 1];
                int nextWordCount = nextSentence.Text.Split(' ', StringSplitOptions.RemoveEmptyEntries).Length;

                bool shouldSplit = false;
                if (sim < ChunkSimilarityThreshold)
                {
                    if (currentWordCount >= MinChunkWords)
                    {
                        shouldSplit = true;
                    }
                }

                if (currentWordCount + nextWordCount > MaxChunkWords)
                {
                    shouldSplit = true;
                }

                if (shouldSplit)
                {
                    chunks.Add(BuildChunk(currentSentences, chunkIndex++));
                    currentSentences = new List<Sentence> { nextSentence };
                    currentWordCount = nextWordCount;
                }
                else
                {
                    currentSentences.Add(nextSentence);
                    currentWordCount += nextWordCount;
                }
            }

            if (currentSentences.Count > 0)
            {
                chunks.Add(BuildChunk(currentSentences, chunkIndex));
            }

            return chunks;
        }

        // Embeds sentences in batches with a per-item fallback. Returns the surviving sentences and their
        // embeddings, kept strictly index-aligned (degenerate inputs that fail to embed are dropped from both).
        static async Task<(List<Sentence> Sentences, List<Embedding<float>> Embeddings)> EmbedSentencesAsync(
            IEmbeddingGenerator<string, Embedding<float>> generator, List<Sentence> sentences)
        {
            const int batchSize = 32;
            var keptSentences = new List<Sentence>();
            var embeddings = new List<Embedding<float>>();

            // Drop empty/whitespace sentences up front: bge-m3 yields NaN vectors for them, which Ollama
            // cannot serialize ("json: unsupported value: NaN").
            var clean = sentences.Where(s => !string.IsNullOrWhiteSpace(s.Text)).ToList();

            for (int i = 0; i < clean.Count; i += batchSize)
            {
                var batch = clean.GetRange(i, Math.Min(batchSize, clean.Count - i));
                try
                {
                    var res = await generator.GenerateAsync(batch.Select(s => s.Text).ToList());
                    for (int j = 0; j < batch.Count; j++)
                    {
                        keptSentences.Add(batch[j]);
                        embeddings.Add(res[j]);
                    }
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"Batch embedding failed ({ex.Message}); retrying items individually...");
                    foreach (var s in batch)
                    {
                        try
                        {
                            var single = await generator.GenerateAsync(new[] { s.Text });
                            keptSentences.Add(s);
                            embeddings.Add(single[0]);
                        }
                        catch (Exception exItem)
                        {
                            Console.WriteLine($"Skipping sentence that failed to embed: {exItem.Message}");
                        }
                    }
                }
            }

            return (keptSentences, embeddings);
        }

        static SemanticChunk BuildChunk(List<Sentence> chunkSents, int index)
        {
            string content = string.Join(" ", chunkSents.Select(s => s.Text));
            string dominantSpeaker = chunkSents.GroupBy(s => s.Speaker)
                                               .OrderByDescending(g => g.Count())
                                               .First().Key;
            return new SemanticChunk(index, chunkSents[0].Start, chunkSents[^1].End, dominantSpeaker, content);
        }

        static async Task<List<SemanticChunk>> ProcessChunksAndStoreAsync(List<SemanticChunk> chunks, string courseId, string sourceId, string collectionName)
        {
            IChatClient chatClient = new OllamaChatClient(new Uri("http://127.0.0.1:11434"), LlmModel);
            IEmbeddingGenerator<string, Embedding<float>> embeddingGenerator =
                new OllamaEmbeddingGenerator(new Uri("http://127.0.0.1:11434"), EmbeddingModel);

            // Connect to Qdrant using the official gRPC client
            Console.WriteLine("Connecting to Qdrant...");
            using var qdrantClient = new QdrantClient("127.0.0.1", 6334); // gRPC default port is 6334

            // Initialize collection if it doesn't exist
            var collections = await qdrantClient.ListCollectionsAsync();
            if (!collections.Contains(collectionName))
            {
                Console.WriteLine($"Creating Qdrant collection '{collectionName}'...");
                await qdrantClient.CreateCollectionAsync(
                    collectionName: collectionName,
                    vectorsConfig: new VectorParams { Size = EmbeddingDim, Distance = Distance.Cosine }
                );
            }

            var processedChunks = new List<SemanticChunk>();

            foreach (var chunk in chunks)
            {
                // 1. Classify relevance
                string classPrompt = $@"Eres un clasificador de contenido educativo universitario.
Clasifica el siguiente fragmento de clase grabada.

SPEAKER: {chunk.Speaker}
TEXTO: ""{chunk.Content}""

Categorías:
- ACADEMICO: contenido del tema de la materia, explicaciones, definiciones, ejemplos académicos.
- OFF_TOPIC: conversación no relacionada al tema (clima, anécdotas personales, organización, saludos).
- ERROR_ESTUDIANTIL: respuesta incorrecta de un estudiante no corregida explícitamente.
- ANECDOTA: historia ilustrativa, puede ser útil como contexto pero no es conocimiento clave.

Responde únicamente con un objeto JSON válido con este formato:
{{""categoria"": ""ACADEMICO"", ""confidence"": 0.92, ""razon"": ""explicación sobre termodinámica""}}";

                var classOptions = new ChatOptions
                {
                    ResponseFormat = ChatResponseFormat.ForJsonSchema(typeof(ClassificationResult))
                };

                string category = "ACADEMICO";
                double confidence = 0.8;
                try
                {
                    var classResponse = await chatClient.GetResponseAsync(classPrompt, classOptions);
                    var classResult = JsonSerializer.Deserialize<ClassificationResult>(classResponse.Text);
                    if (classResult != null)
                    {
                        category = classResult.Categoria.ToUpper();
                        confidence = classResult.Confidence;
                    }
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"Classification error for chunk {chunk.ChunkIndex}: {ex.Message}");
                }

                // Summary generation removed: it cost a second LLM call per chunk for no downstream value.
                chunk.Category = category;
                chunk.ConfidenceScore = confidence;
                chunk.QdrantVectorId = Guid.NewGuid();

                // 3. Compute embedding vector
                var embRes = await embeddingGenerator.GenerateAsync(new[] { chunk.Content });
                var vector = embRes[0].Vector.ToArray();

                // 4. Index in Qdrant
                var point = new PointStruct
                {
                    Id = new PointId { Uuid = chunk.QdrantVectorId.ToString() },
                    Vectors = vector
                };
                point.Payload.Add("course_id", courseId);
                point.Payload.Add("source_id", sourceId);
                point.Payload.Add("content", chunk.Content);
                point.Payload.Add("speaker", chunk.Speaker);
                point.Payload.Add("category", chunk.Category);
                point.Payload.Add("confidence_score", chunk.ConfidenceScore);
                point.Payload.Add("topic_summary", chunk.TopicSummary);
                point.Payload.Add("ts_start", chunk.TsStart);
                point.Payload.Add("ts_end", chunk.TsEnd);

                await qdrantClient.UpsertAsync(collectionName, new[] { point });
                Console.WriteLine($"Indexed chunk {chunk.ChunkIndex} in Qdrant (ID: {chunk.QdrantVectorId}, Category: {chunk.Category}, Topic: {chunk.TopicSummary}).");
                processedChunks.Add(chunk);
            }

            return processedChunks;
        }

        static async Task<List<QuizQuestion>> GenerateQuizQuestionsAsync(string courseId, int numQuestions, string bloomLevel, string collectionName)
        {
            IChatClient chatClient = new OllamaChatClient(new Uri("http://127.0.0.1:11434"), LlmModel);
            IChatClient judgeClient = new OllamaChatClient(new Uri("http://127.0.0.1:11434"), JudgeModel);
            IEmbeddingGenerator<string, Embedding<float>> embeddingGenerator =
                new OllamaEmbeddingGenerator(new Uri("http://127.0.0.1:11434"), EmbeddingModel);
            using var qdrantClient = new QdrantClient("127.0.0.1", 6334);

            // Retrieve course chunks from Qdrant
            Console.WriteLine("Retrieving academic chunks from Qdrant for RAG context...");
            // Filter search for this course_id and ACADEMICO category
            var searchFilter = new Filter();
            searchFilter.Must.Add(new Condition { Field = new FieldCondition { Key = "course_id", Match = new Match { Text = courseId } } });
            searchFilter.Must.Add(new Condition { Field = new FieldCondition { Key = "category", Match = new Match { Text = "ACADEMICO" } } });

            // Bumped scroll limit so 2h-class chunk counts (often 60-150+) are not truncated.
            var scrollResult = await qdrantClient.ScrollAsync(
                collectionName: collectionName,
                filter: searchFilter,
                limit: 500
            );

            var retrievedChunks = scrollResult.Result;
            if (retrievedChunks.Count == 0)
            {
                Console.WriteLine("No academic chunks found in Qdrant.");
                return new List<QuizQuestion>();
            }

            // Adaptive target: stays dynamic with content size but biased smaller (see consts).
            // Caller can override by passing numQuestions > 0.
            int targetQuestions = numQuestions > 0
                ? numQuestions
                : Math.Clamp((int)Math.Ceiling(retrievedChunks.Count / QuestionsPerChunkDivisor), MinQuestions, MaxQuestions);
            Console.WriteLine($"Academic chunks retrieved: {retrievedChunks.Count}. Target questions: {targetQuestions}.");

            // ---- Phase 0: keep only the most significant chunks (classifier confidence × concept density). ----
            // Generating from every chunk is the dominant time sink and dilutes quality; the densest chunks
            // hold the actual concepts, so we generate from those only.
            var candidates = retrievedChunks
                .Select(p => new
                {
                    Content = p.Payload.TryGetValue("content", out var contentVal) ? contentVal.StringValue : "",
                    Confidence = p.Payload.TryGetValue("confidence_score", out var confVal) ? confVal.DoubleValue : 0.8
                })
                .Where(x => !string.IsNullOrWhiteSpace(x.Content))
                .Select(x => new
                {
                    x.Content,
                    x.Confidence,
                    WordCount = x.Content.Split(' ', StringSplitOptions.RemoveEmptyEntries).Length
                })
                .ToList();

            // Skip thin chunks (they yield trivial questions); relax only if the filter removes everything.
            var dense = candidates.Where(x => x.WordCount >= 30).ToList();
            if (dense.Count == 0) dense = candidates;

            var selectedChunks = dense
                .OrderByDescending(x => x.Confidence * x.WordCount)
                .Take(targetQuestions)
                .Select(x => x.Content)
                .ToList();
            Console.WriteLine($"Selected {selectedChunks.Count} significant chunks for question generation.");

            // ---- Phase 1: generate every question first, so the generator model stays resident (no per-question swap). ----
            var quizOptions = new ChatOptions
            {
                ResponseFormat = ChatResponseFormat.ForJsonSchema(typeof(QuizQuestion))
            };
            var candidateQuestions = new List<(QuizQuestion Question, string Source)>();
            foreach (var chunkContent in selectedChunks)
            {
                string quizPrompt = $@"Eres un experto en diseño instruccional universitario.
Basándote ÚNICAMENTE en el siguiente contenido de clase, genera UNA pregunta de opción múltiple con 4 opciones (a, b, c, d).

REQUISITOS:
- Nivel de Taxonomía de Bloom objetivo: {bloomLevel}
- La respuesta correcta debe poder verificarse y justificarse de manera directa y factual con el contenido dado.
- Cada pregunta debe tener 3 distractores que vayan acorde al tema.
- No generes preguntas sobre detalles triviales, saludos, o anécdotas personales.
- No inventes información externa, pero entiende el contexto si la transcripción está mal.
- Ten en cuenta que la transcripcion puede tener ruido o conceptos que no se interpretaron de forma exacta.

Responde únicamente con un objeto JSON válido que siga exactamente esta estructura:
{{
  ""question"": ""texto de la pregunta..."",
  ""options"": {{
    ""a"": ""opción a..."",
    ""b"": ""opción b..."",
    ""c"": ""opción c...""
    ""d"": ""opción d...""
  }},
  ""correct_option"": ""a"",
  ""bloom_level"": ""{bloomLevel}"",
  ""justification"": ""justificación basada en el texto...""
}}

CONTENIDO DE CLASE:
""{chunkContent}""

";

                try
                {
                    var quizResponse = await chatClient.GetResponseAsync(quizPrompt, quizOptions);
                    var question = JsonSerializer.Deserialize<QuizQuestion>(quizResponse.Text);
                    if (question != null && !string.IsNullOrWhiteSpace(question.Question))
                    {
                        candidateQuestions.Add((question, chunkContent));
                    }
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"Error generating question: {ex.Message}");
                }
            }

            // ---- Phase 2: judge every candidate in one pass (judge model loads once, not once per question). ----
            var valOptions = new ChatOptions
            {
                ResponseFormat = ChatResponseFormat.ForJsonSchema(typeof(ValidationResult))
            };
            var vettedQuestions = new List<QuizQuestion>();
            foreach (var (question, source) in candidateQuestions)
            {
                string valPrompt = $@"Eres un evaluador de preguntas de examen universitario. Tu objetivo es juzgar la calidad y fidelidad factual de la pregunta generada a partir de un fragmento de clase y contrasta con el conocimiento que tengas del tema.

FRAGMENTO DE CLASE:
""{source}""

PREGUNTA EVALUADA:
Pregunta: {question.Question}
Opciones:
a) {question.Options.GetValueOrDefault("a")}
b) {question.Options.GetValueOrDefault("b")}
c) {question.Options.GetValueOrDefault("c")}
d) {question.Options.GetValueOrDefault("d")}
Respuesta correcta: {question.CorrectOption}

REGLAS DE EVALUACIÓN:
1. Grounding factual (0.0 a 1.0): ¿La respuesta correcta está totalmente respaldada y demostrada de forma directa por el fragmento de clase? Si requiere suposiciones externas, dale puntaje bajo.
2. Calidad de distractores (0.0 a 1.0): ¿Los distractores son plausibles pero indiscutiblemente falsos según el fragmento?
3. Relevancia (0.0 a 1.0): ¿La pregunta evalúa conceptos clave y no detalles insignificantes?

Calcula el promedio general de estas 3 reglas como un valor entre 0.0 y 1.0.
Responde únicamente con un objeto JSON válido con este formato:
{{""score"": 0.85, ""reason"": ""explicación breve de la evaluación""}}";

                double valScore = 0.0;
                try
                {
                    var valResponse = await judgeClient.GetResponseAsync(valPrompt, valOptions);
                    var valResult = JsonSerializer.Deserialize<ValidationResult>(valResponse.Text);
                    valScore = valResult?.Score ?? 0.0;
                    Console.WriteLine($"LLM-as-a-judge score: {valScore} (Reason: {valResult?.Reason})");
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"Error validating question: {ex.Message}");
                }

                if (valScore < 0.75)
                {
                    Console.WriteLine("Discarding question due to low validation score.");
                    continue;
                }

                vettedQuestions.Add(question);
            }

            // ---- Phase 3: drop near-duplicate questions. Embed per item with a guard: bge-m3 emits a NaN
            // vector for some inputs, which Ollama cannot serialize and would otherwise kill a batch call. ----
            var generatedQuestions = new List<QuizQuestion>();
            var keptEmbeddings = new List<Embedding<float>>();
            foreach (var question in vettedQuestions)
            {
                Embedding<float>? qEmb = null;
                try
                {
                    var res = await embeddingGenerator.GenerateAsync(new[] { question.Question });
                    qEmb = res[0];
                }
                catch (Exception ex)
                {
                    // Keep the (already judge-vetted) question; just can't dedup-check this one.
                    Console.WriteLine($"Dedup embedding failed, accepting without duplicate check: {ex.Message}");
                }

                if (qEmb != null)
                {
                    bool isDuplicate = false;
                    for (int j = 0; j < keptEmbeddings.Count; j++)
                    {
                        if (TensorPrimitives.CosineSimilarity(qEmb.Vector.Span, keptEmbeddings[j].Vector.Span) > 0.92f)
                        {
                            isDuplicate = true;
                            break;
                        }
                    }

                    if (isDuplicate)
                    {
                        Console.WriteLine("Discarding question as a semantic duplicate of an existing question.");
                        continue;
                    }

                    keptEmbeddings.Add(qEmb);
                }

                generatedQuestions.Add(question);
                Console.WriteLine($"Accepted question: {question.Question}");
            }

            return generatedQuestions;
        }
    }
}
