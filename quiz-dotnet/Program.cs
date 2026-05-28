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
        static async Task Main(string[] args)
        {
            Console.WriteLine("=== Quiz Generator Pipeline (.NET 10 Unified Stack) ===");

            var pipelineStart = DateTimeOffset.Now;
            var totalStopwatch = Stopwatch.StartNew();
            Console.WriteLine($"Pipeline started at: {pipelineStart:yyyy-MM-dd HH:mm:ss}");

            string inputPath = args.Length > 0 ? args[0] : "/home/ubuntu/quiz/test01_20s.wav";
            string courseId = "course-123";
            string sourceId = Guid.NewGuid().ToString();
            // Unique collection per run so vectors don't pile up in a shared collection
            string collectionName = $"quiz_chunks_dotnet_{DateTime.Now:yyyyMMdd_HHmmss}";
            Console.WriteLine($"Qdrant collection for this run: {collectionName}");

            // 1. Download Whisper model if not exists
            var whisperModelType = GgmlType.Small;
            string whisperModelName = $"whisper-{whisperModelType.ToString().ToLower()}";
            string modelPath = "ggml-small.bin";
            if (!File.Exists(modelPath))
            {
                Console.WriteLine($"Downloading Whisper GGML {whisperModelType} model to {modelPath}...");
                using var modelStream = await WhisperGgmlDownloader.Default.GetGgmlModelAsync(whisperModelType);
                using var fileStream = File.OpenWrite(modelPath);
                await modelStream.CopyToAsync(fileStream);
                Console.WriteLine("Whisper model downloaded successfully.");
            }

            var stageStopwatch = Stopwatch.StartNew();

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
            var rawSegments = new List<TranscriptSegment>();
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

            // Save raw transcript for quality monitoring
            SaveTranscript(inputPath, rawSegments, whisperModelName);

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

            // 7. RAG & Quiz Generation
            var generatedQuestions = await GenerateQuizQuestionsAsync(courseId, 3, "comprender", collectionName);
            LogStage("RAG & Quiz Generation", stageStopwatch);

            Console.WriteLine("\n=== Pipeline Execution Completed Successfully ===");
            Console.WriteLine($"Generated {generatedQuestions.Count} valid quiz questions.");
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

            Console.WriteLine("Computing sentence embeddings using Ollama (nomic-embed-text)...");
            IEmbeddingGenerator<string, Embedding<float>> embeddingGenerator =
                new OllamaEmbeddingGenerator(new Uri("http://127.0.0.1:11434"), "nomic-embed-text");

            var embeddings = new List<Embedding<float>>();
            foreach (var s in sentences)
            {
                var response = await embeddingGenerator.GenerateAsync(new[] { s.Text });
                embeddings.Add(response[0]);
            }

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
                if (sim < 0.5f)
                {
                    if (currentWordCount >= 30)
                    {
                        shouldSplit = true;
                    }
                }

                if (currentWordCount + nextWordCount > 350)
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
            IChatClient chatClient = new OllamaChatClient(new Uri("http://127.0.0.1:11434"), "llama3.2:3b");
            IEmbeddingGenerator<string, Embedding<float>> embeddingGenerator =
                new OllamaEmbeddingGenerator(new Uri("http://127.0.0.1:11434"), "nomic-embed-text");

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
                    vectorsConfig: new VectorParams { Size = 768, Distance = Distance.Cosine }
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

                // 2. Generate summary
                string summaryPrompt = $"Proporciona un título o resumen del tema de máximo 5 palabras para el siguiente fragmento. Responde únicamente con el resumen sin comillas ni texto adicional:\n\n{chunk.Content}";
                string summary = "Resumen de clase";
                try
                {
                    var summaryResponse = await chatClient.GetResponseAsync(summaryPrompt);
                    summary = summaryResponse.Text.Trim().Replace("\"", "");
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"Summary error for chunk {chunk.ChunkIndex}: {ex.Message}");
                }

                chunk.Category = category;
                chunk.ConfidenceScore = confidence;
                chunk.TopicSummary = summary;
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
            IChatClient chatClient = new OllamaChatClient(new Uri("http://127.0.0.1:11434"), "llama3.2:3b");
            IEmbeddingGenerator<string, Embedding<float>> embeddingGenerator =
                new OllamaEmbeddingGenerator(new Uri("http://127.0.0.1:11434"), "nomic-embed-text");
            using var qdrantClient = new QdrantClient("127.0.0.1", 6334);

            // Retrieve course chunks from Qdrant
            Console.WriteLine("Retrieving academic chunks from Qdrant for RAG context...");
            // Filter search for this course_id and ACADEMICO category
            var searchFilter = new Filter();
            searchFilter.Must.Add(new Condition { Field = new FieldCondition { Key = "course_id", Match = new Match { Text = courseId } } });
            searchFilter.Must.Add(new Condition { Field = new FieldCondition { Key = "category", Match = new Match { Text = "ACADEMICO" } } });

            // We can search or scroll
            var scrollResult = await qdrantClient.ScrollAsync(
                collectionName: collectionName,
                filter: searchFilter,
                limit: 50
            );

            var retrievedChunks = scrollResult.Result;
            if (retrievedChunks.Count == 0)
            {
                Console.WriteLine("No academic chunks found in Qdrant.");
                return new List<QuizQuestion>();
            }

            var generatedQuestions = new List<QuizQuestion>();
            var questionEmbeddings = new List<Embedding<float>>();

            foreach (var point in retrievedChunks)
            {
                if (generatedQuestions.Count >= numQuestions) break;

                string chunkContent = point.Payload.TryGetValue("content", out var contentVal) ? contentVal.StringValue : "";
                if (string.IsNullOrWhiteSpace(chunkContent)) continue;

                // 1. Generate Question via RAG
                string quizPrompt = $@"Eres un experto en diseño instruccional universitario.
Basándote ÚNICAMENTE en el siguiente contenido de clase, genera UNA pregunta de opción múltiple con 4 opciones (a, b, c, d).

CONTENIDO DE CLASE:
""{chunkContent}""

REQUISITOS:
- Nivel de Taxonomía de Bloom objetivo: {bloomLevel}
- La respuesta correcta debe poder verificarse y justificarse de manera directa y factual con el contenido dado.
- Los 3 distractores deben ser plausibles pero claramente incorrectos según el contenido de clase.
- No generes preguntas sobre detalles triviales, saludos, o anécdotas personales.

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
}}";

                var quizOptions = new ChatOptions
                {
                    ResponseFormat = ChatResponseFormat.ForJsonSchema(typeof(QuizQuestion))
                };

                QuizQuestion? question = null;
                try
                {
                    var quizResponse = await chatClient.GetResponseAsync(quizPrompt, quizOptions);
                    question = JsonSerializer.Deserialize<QuizQuestion>(quizResponse.Text);
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"Error generating question: {ex.Message}");
                    continue;
                }

                if (question == null || string.IsNullOrWhiteSpace(question.Question)) continue;

                // 2. Validate Question (LLM-as-a-judge)
                string valPrompt = $@"Eres un evaluador de preguntas de examen universitario. Tu objetivo es juzgar la calidad y fidelidad factual de la pregunta generada a partir de un fragmento de clase.

FRAGMENTO DE CLASE:
""{chunkContent}""

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

                var valOptions = new ChatOptions
                {
                    ResponseFormat = ChatResponseFormat.ForJsonSchema(typeof(ValidationResult))
                };

                double valScore = 0.0;
                try
                {
                    var valResponse = await chatClient.GetResponseAsync(valPrompt, valOptions);
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

                // 3. Deduplication Check
                var qEmbRes = await embeddingGenerator.GenerateAsync(new[] { question.Question });
                var qEmb = qEmbRes[0];

                bool isDuplicate = false;
                foreach (var existingEmb in questionEmbeddings)
                {
                    float sim = TensorPrimitives.CosineSimilarity(qEmb.Vector.Span, existingEmb.Vector.Span);
                    if (sim > 0.92f)
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

                generatedQuestions.Add(question);
                questionEmbeddings.Add(qEmb);
                Console.WriteLine($"Accepted question: {question.Question}");
            }

            return generatedQuestions;
        }
    }
}
